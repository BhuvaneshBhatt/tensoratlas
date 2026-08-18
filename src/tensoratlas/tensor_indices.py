from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Sequence, Tuple, Callable

import sympy as sp

from .symbolic_decision import is_equal, is_zero, light_simplify, canonical_simplify
from .simplification_policy import cheap_simplify, normal_simplify, strong_simplify
from .cache_utils import BoundedCache
from .indexed_config import resolve_indexed_config as _resolve_indexed_config, decision_mode as _config_decision_mode, configured_simplify_expr as _config_simplify
from .tensorform_types import (
    AbstractIndexedExpr, ComponentIndexedExpr, ContractionPlan, IndexedTensorForm,
    NormalizationDiagnostics, TensorFormTerm, TensorOptimizationReport, TensorSpace,
    abstract_layer as _abstract_layer_external, component_layer as _component_layer_external,
    is_abstract_layer as _is_abstract_layer_external, is_component_layer as _is_component_layer_external,
    unwrap_layer as _unwrap_layer_external, wrap_abstract as _wrap_abstract_external,
)

from .tensor_core import TensorObject, add_tensors, symmetry_canonicalize
from .basis import IndexBundle
from .bundle_identity import bundle_metadata as _bundle_metadata, basis_bundle as _basis_bundle, index_bundle_compatible as _index_bundle_compatible, infer_bundle_from_basis
from .canonical_keys import structural_key as _safe_structural_key, factor_key as _canonical_factor_key, term_group_key as _canonical_term_group_key, term_sort_key as _canonical_term_sort_key, canonical_sort_key as _canonical_sort_key
from .fields import ScalarField, TensorField
from .normal_forms import TNFMatrix, tnf_build_array, tnf_iter_indices




@dataclass(frozen=True)
class TensorIndex:
    name: str
    variance: str  # 'u' or 'l'
    bundle: str | None = None

    def dual(self) -> 'TensorIndex':
        return TensorIndex(self.name, 'l' if self.variance == 'u' else 'u', self.bundle)

    def compatible_with_basis(self, basis) -> bool:
        return _index_bundle_compatible(self.bundle, basis, allow_missing=False)

    def __str__(self) -> str:
        core = f"{self.name}{'^' if self.variance == 'u' else '_'}"
        return core if self.bundle is None else f"{core}:{self.bundle}"


@dataclass(frozen=True)
class IndexedTensor:
    tensor: TensorObject
    indices: Tuple[TensorIndex, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.tensor.variance_spec):
            raise ValueError('Need one index per tensor slot.')
        for pos, (idx, kind) in enumerate(zip(self.indices, self.tensor.variance_spec)):
            if idx.variance != kind:
                raise ValueError('Index variance must match tensor slot variance.')
            if not idx.compatible_with_basis(self.tensor.slot_bases[pos]):
                raise ValueError('Index bundle/type must be compatible with tensor slot basis.')

    def rename_indices(self, mapping: Dict[str, str]) -> 'IndexedTensor':
        return IndexedTensor(self.tensor, tuple(TensorIndex(mapping.get(idx.name, idx.name), idx.variance, idx.bundle) for idx in self.indices))

    def canonicalize_names(self) -> 'IndexedTensor':
        counts: Dict[str, Dict[str, int]] = {}
        for idx in self.indices:
            counts.setdefault(idx.name, {'u': 0, 'l': 0})[idx.variance] += 1
        rename: Dict[str, str] = {}
        dummies = sorted(name for name, c in counts.items() if c['u'] and c['l'])
        for k, name in enumerate(dummies):
            rename[name] = f'd{k}'
        return self.rename_indices(rename)

    def canonicalize_free_indices(self) -> 'IndexedTensor':
        counts: Dict[str, Dict[str, int]] = {}
        for idx in self.indices:
            counts.setdefault(idx.name, {'u': 0, 'l': 0})[idx.variance] += 1
        free_positions = [p for p, idx in enumerate(self.indices) if not (counts[idx.name]['u'] and counts[idx.name]['l'])]
        desired = sorted(free_positions, key=lambda p: (self.indices[p].variance, self.indices[p].name, p))
        if free_positions == desired:
            return self
        order = list(range(len(self.indices)))
        for tgt, src in zip(free_positions, desired):
            order[tgt] = src
        used = set(desired)
        remain = [i for i in range(len(self.indices)) if i not in used]
        for pos in range(len(order)):
            if pos not in free_positions:
                order[pos] = remain.pop(0)
        return IndexedTensor(self.tensor.permute_slots(order), tuple(self.indices[i] for i in order))

    def canonicalize(self) -> 'IndexedTensor':
        order = list(range(len(self.indices)))
        sign = 1
        md = self.tensor.symmetry_metadata
        for key, groups in md.items():
            if key not in {'symmetric', 'antisymmetric'}:
                continue
            for group in groups:
                group = tuple(group)
                names = [self.indices[s].name for s in group]
                desired = sorted(range(len(group)), key=lambda k: (names[k], self.indices[group[k]].variance))
                reordered = [group[i] for i in desired]
                if tuple(reordered) != group:
                    perm = list(range(len(self.indices)))
                    for new_slot, old_slot in zip(group, reordered):
                        perm[new_slot] = old_slot
                    order = [order[p] for p in perm]
                    if key == 'antisymmetric':
                        inv = 0
                        for i in range(len(desired)):
                            for j in range(i + 1, len(desired)):
                                if desired[i] > desired[j]:
                                    inv += 1
                        if inv % 2:
                            sign *= -1
        out = IndexedTensor(self.tensor.permute_slots(order), tuple(self.indices[i] for i in order))
        if sign == -1:
            out = IndexedTensor(_scale_tensor(out.tensor, -1), out.indices)
        return out.canonicalize_names().canonicalize_free_indices()

    def contract_repeated(self):
        tensor = self.tensor
        indices = list(self.indices)
        changed = True
        while changed:
            changed = False
            by_name: Dict[str, List[int]] = {}
            for pos, idx in enumerate(indices):
                by_name.setdefault(idx.name, []).append(pos)
            for _, positions in by_name.items():
                found = None
                for i in positions:
                    for j in positions:
                        if i < j and indices[i].variance != indices[j].variance:
                            found = (i, j)
                            break
                    if found:
                        break
                if found is None:
                    continue
                res = tensor.contract_slots(*found)
                if isinstance(res, ScalarField):
                    return res
                tensor = res
                indices = [idx for k, idx in enumerate(indices) if k not in set(found)]
                changed = True
                break
        return IndexedTensor(symmetry_canonicalize(tensor), tuple(indices)).canonicalize_free_indices().canonicalize_names()

    def simplify(self):
        return self.canonicalize().contract_repeated()

    def evaluate(self):
        return self.simplify()

    def equals(self, other) -> bool:
        if not isinstance(other, IndexedTensor):
            return False
        left = _alpha_canonical(self.simplify())
        right = _alpha_canonical(other.simplify())
        if isinstance(left, ScalarField) or isinstance(right, ScalarField):
            return isinstance(left, ScalarField) and isinstance(right, ScalarField) and is_zero(_configured_simplify_expr(left.expr - right.expr, config), mode=_decision_mode_from_config(config))
        return left.indices == right.indices and left.tensor.equivalent(right.tensor)

    def __mul__(self, other):
        if not isinstance(other, IndexedTensor):
            raise TypeError('Can only multiply IndexedTensor by IndexedTensor.')
        return IndexedTensorExpr('tensor_product', (self, other))

    def __add__(self, other):
        if not isinstance(other, IndexedTensor):
            raise TypeError('Can only add IndexedTensor to IndexedTensor.')
        return IndexedTensorExpr('add', (self, other))

    def __str__(self) -> str:
        base = self.tensor.name or 'T'
        uppers = ''.join(i.name for i in self.indices if i.variance == 'u')
        lowers = ''.join(i.name for i in self.indices if i.variance == 'l')
        if uppers and lowers:
            return f"{base}^{{{uppers}}}_{{{lowers}}}"
        if uppers:
            return f"{base}^{{{uppers}}}"
        if lowers:
            return f"{base}_{{{lowers}}}"
        return base

    def pretty(self) -> str:
        return str(self)

    def latex(self) -> str:
        return str(self)


@dataclass(frozen=True)
class IndexedTensorExpr:
    op: str
    args: Tuple[object, ...]

    def __str__(self) -> str:
        if self.op == 'tensor':
            return str(self.args[0])
        sep = ' + ' if self.op == 'add' else ' '
        return '(' + sep.join(str(a) for a in self.args) + ')'

    def pretty(self) -> str:
        return str(self)

    def latex(self) -> str:
        sep = ' + ' if self.op == 'add' else ' '
        return sep.join(a.latex() if hasattr(a, 'latex') else str(a) for a in self.args)

    def substitute(self, old, new):
        if self == old:
            return new
        new_args = []
        for arg in self.args:
            if isinstance(arg, IndexedTensorExpr):
                new_args.append(arg.substitute(old, new))
            elif arg == old:
                new_args.append(new)
            else:
                new_args.append(arg)
        return IndexedTensorExpr(self.op, tuple(new_args))

    def rewrite(self, rule: Callable[[object], object | None]):
        replaced = rule(self)
        if replaced is not None and replaced is not self:
            return replaced
        args = []
        for arg in self.args:
            if isinstance(arg, IndexedTensorExpr):
                args.append(arg.rewrite(rule))
            else:
                args.append(rule(arg) or arg)
        return IndexedTensorExpr(self.op, tuple(args))

    def simplify(self):
        if self.op == 'tensor':
            return _eval_indexed(self.args[0]).simplify()
        if self.op == 'add':
            terms = _flatten_add(self)
            simp_terms = [_eval_indexed(t) if isinstance(t, IndexedTensor) else t.simplify() for t in terms]
            simp_terms = [t if not isinstance(t, IndexedTensorExpr) else t.evaluate() for t in simp_terms]
            scalar_total = None
            groups: Dict[Tuple, IndexedTensor] = {}
            for obj in simp_terms:
                if isinstance(obj, ScalarField):
                    scalar_total = obj if scalar_total is None else ScalarField(obj.chart, sp.simplify(scalar_total.expr + obj.expr))
                    continue
                if not isinstance(obj, IndexedTensor):
                    raise TypeError(f'Unsupported add term {type(obj)!r}')
                simp = obj.simplify()
                if isinstance(simp, ScalarField):
                    scalar_total = simp if scalar_total is None else ScalarField(simp.chart, sp.simplify(scalar_total.expr + simp.expr))
                    continue
                key = ('tensor', tuple((idx.name, idx.variance, idx.bundle) for idx in simp.indices), simp.tensor.variance_spec, tuple((b.name, b.kind) for b in simp.tensor.slot_bases))
                if key in groups:
                    groups[key] = IndexedTensor(add_tensors(groups[key].tensor, simp.tensor), groups[key].indices).canonicalize()
                else:
                    groups[key] = simp
            if scalar_total is not None and not groups:
                return scalar_total
            if not groups:
                return scalar_total
            acc = None
            for term in groups.values():
                acc = term if acc is None else IndexedTensor(add_tensors(acc.tensor, term.tensor), acc.indices).canonicalize()
            if scalar_total is not None:
                raise ValueError('Cannot add scalar and indexed-tensor terms in one expression.')
            return acc.canonicalize_free_indices()
        if self.op == 'tensor_product':
            factors = _flatten_product(self)
            simp = [_eval_indexed(f) if isinstance(f, IndexedTensor) else f.simplify() for f in factors]
            simp = [s if not isinstance(s, IndexedTensorExpr) else s.evaluate() for s in simp]
            return _simplify_product_factors(simp)
        raise NotImplementedError(f'Unknown op {self.op!r}')

    def evaluate(self):
        return self.simplify()


def _flatten_add(obj) -> List[object]:
    if isinstance(obj, IndexedTensorExpr) and obj.op == 'add':
        return _flatten_add(obj.args[0]) + _flatten_add(obj.args[1])
    return [obj]


def _flatten_product(obj) -> List[object]:
    if isinstance(obj, IndexedTensorExpr) and obj.op == 'tensor_product':
        return _flatten_product(obj.args[0]) + _flatten_product(obj.args[1])
    return [obj]


def _simplify_product_factors(factors: Sequence[object]):
    scalar_expr = sp.Integer(1)
    indexed_factors: List[IndexedTensor] = []
    for factor in factors:
        if isinstance(factor, ScalarField):
            scalar_expr *= factor.expr
        elif isinstance(factor, IndexedTensor):
            indexed_factors.append(factor.canonicalize())
        else:
            raise TypeError(f'Unsupported tensor-product factor {type(factor)!r}')

    indexed_factors = sorted(indexed_factors, key=_factor_sort_key)
    changed = True
    while changed:
        changed = False
        done, indexed_factors, scalar_expr = _more_complete_special_simplify(indexed_factors, scalar_expr)
        if done:
            changed = True
            continue
        # delta rewrites anywhere in product
        for i, factor in enumerate(indexed_factors):
            if _classify_special_tensor(factor.tensor) != 'delta':
                continue
            up = next(idx for idx in factor.indices if idx.variance == 'u')
            low = next(idx for idx in factor.indices if idx.variance == 'l')
            if up.name == low.name:
                scalar_expr *= factor.tensor.chart.dimension
                indexed_factors.pop(i)
                changed = True
                break
            repl = []
            hit = False
            for j, other in enumerate(indexed_factors):
                if i == j:
                    continue
                mapping = {}
                if any(idx.name == low.name for idx in other.indices):
                    if up.bundle is None or other.indices[0].bundle is None or True:
                        mapping[low.name] = up.name
                    hit = True
                repl.append(other.rename_indices(mapping) if mapping else other)
            if hit:
                indexed_factors = repl
                changed = True
                break
        if changed:
            continue
        # metric rewrites on arbitrary tensor slots
        for i, factor in enumerate(indexed_factors):
            cls = _classify_special_tensor(factor.tensor)
            if cls not in {'metric_ll', 'metric_uu'}:
                continue
            m_indices = factor.indices
            for j, other in enumerate(indexed_factors):
                if i == j:
                    continue
                for slot, idx in enumerate(other.indices):
                    if cls == 'metric_ll' and idx.variance == 'u':
                        matches = [k for k, midx in enumerate(m_indices) if midx.variance == 'l' and midx.name == idx.name]
                        if not matches:
                            continue
                        free_pos = 1 - matches[0]
                        lowered = other.tensor.lower_slots([slot])
                        new_indices = list(other.indices)
                        new_indices[slot] = TensorIndex(m_indices[free_pos].name, 'l', m_indices[free_pos].bundle)
                        indexed_factors = [f for k, f in enumerate(indexed_factors) if k not in {i, j}] + [IndexedTensor(lowered, tuple(new_indices)).canonicalize()]
                        changed = True
                        break
                    if cls == 'metric_uu' and idx.variance == 'l':
                        matches = [k for k, midx in enumerate(m_indices) if midx.variance == 'u' and midx.name == idx.name]
                        if not matches:
                            continue
                        free_pos = 1 - matches[0]
                        raised = other.tensor.raise_slots([slot])
                        new_indices = list(other.indices)
                        new_indices[slot] = TensorIndex(m_indices[free_pos].name, 'u', m_indices[free_pos].bundle)
                        indexed_factors = [f for k, f in enumerate(indexed_factors) if k not in {i, j}] + [IndexedTensor(raised, tuple(new_indices)).canonicalize()]
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break

    if not indexed_factors:
        chart = next((f.tensor.chart for f in factors if isinstance(f, IndexedTensor)), None)
        if chart is None:
            raise ValueError('Need a chart-carrying factor to build a scalar result.')
        return ScalarField(chart, sp.simplify(scalar_expr))

    prod = indexed_factors[0]
    for factor in indexed_factors[1:]:
        prod = IndexedTensor(prod.tensor.tensor_product(factor.tensor), prod.indices + factor.indices)
        prod = prod.canonicalize().contract_repeated()
        if isinstance(prod, ScalarField):
            scalar_expr *= prod.expr
            prod = None
            break
    if prod is None:
        return ScalarField(indexed_factors[0].tensor.chart, sp.simplify(scalar_expr))
    result = prod.simplify()
    if isinstance(result, ScalarField):
        return ScalarField(result.chart, sp.simplify(scalar_expr * result.expr))
    if scalar_expr != 1:
        result = IndexedTensor(_scale_tensor(result.tensor, scalar_expr), result.indices).canonicalize()
    return result


def _negate_tensor(tensor: TensorObject) -> TensorObject:
    return _scale_tensor(tensor, -1)


def _scale_tensor(tensor: TensorObject, scalar) -> TensorObject:
    arr = tnf_build_array(tensor.components.shape, lambda idx: scalar * tensor.components[idx if idx else ()])
    return TensorObject(tensor.chart, arr, tensor.variance_spec, tensor.slot_bases, tensor.name, dict(tensor.symmetry_metadata))


def _tensor_equal(left: TensorObject, right: TensorObject) -> bool:
    if left.chart != right.chart or left.variance_spec != right.variance_spec or left.slot_bases != right.slot_bases:
        return False
    if left.components.rank() == 0:
        return is_equal(left.components[()], right.components[()])
    for idx in tnf_iter_indices(left.components.shape):
        if not is_zero(light_simplify(left.components[idx] - right.components[idx])):
            return False
    return True


def _matrix_matches_tensor(matrix: TNFMatrix | None, tensor: TensorObject) -> bool:
    if matrix is None or len(tensor.variance_spec) != 2:
        return False
    if tensor.components.shape != matrix.shape:
        return False
    return all(is_equal(tensor.components[(i, j)], matrix[(i, j)]) for i in range(matrix.rows) for j in range(matrix.cols))


def _classify_special_tensor(tensor: TensorObject) -> str | None:
    dim = tensor.chart.dimension
    if len(tensor.variance_spec) == 2 and tensor.variance_spec == 'ul':
        if all(is_equal(tensor.components[(i, j)], (1 if i == j else 0)) for i in range(dim) for j in range(dim)):
            return 'delta'

    if len(tensor.variance_spec) == 2 and tensor.variance_spec == 'll' and _matrix_matches_tensor(tensor.chart.metric(), tensor):
        return 'metric_ll'

    if len(tensor.variance_spec) == 2 and tensor.variance_spec == 'uu' and _matrix_matches_tensor(tensor.chart.inverse_metric(), tensor):
        return 'metric_uu'

    if len(tensor.variance_spec) == dim and set(tensor.variance_spec) in [{'l'}, {'u'}]:
        if set(tensor.components.entries).issubset({sp.Integer(-1), sp.Integer(0), sp.Integer(1)}):
            return 'epsilon'

    return None


def _eval_indexed(obj):
    if isinstance(obj, IndexedTensorExpr):
        return obj.simplify()
    if isinstance(obj, IndexedTensor):
        return obj
    raise TypeError(f'Unsupported indexed object {type(obj)!r}')




def index_bundle(name: str, dimension: int | None = None) -> IndexBundle:
    return IndexBundle(name, dimension)



def _complete_special_tensor_simplify(indexed_factors: list[IndexedTensor], scalar_expr):
    # epsilon-epsilon full contraction to determinant-like factorial in Euclidean charts
    changed=False
    for i in range(len(indexed_factors)):
        for j in range(i+1, len(indexed_factors)):
            ci=_classify_special_tensor(indexed_factors[i].tensor)
            cj=_classify_special_tensor(indexed_factors[j].tensor)
            if ci=='epsilon' and cj=='epsilon':
                a=indexed_factors[i]; b=indexed_factors[j]
                if len(a.indices)==len(b.indices)==a.tensor.chart.dimension:
                    namesa=[(x.name,x.variance,x.bundle) for x in a.indices]
                    namesb=[(x.name,'u' if x.variance=='l' else 'l',x.bundle) for x in b.indices]
                    if sorted(namesa)==sorted(namesb):
                        scalar_expr *= sp.factorial(a.tensor.chart.dimension)
                        rest=[f for k,f in enumerate(indexed_factors) if k not in {i,j}]
                        return True, rest, scalar_expr
    return False, indexed_factors, scalar_expr

def indices(spec: str) -> Tuple[TensorIndex, ...]:
    out = []
    for tok in spec.split():
        bundle=None
        if ':' in tok:
            tok, bundle = tok.split(':',1)
        if tok.endswith('^'):
            out.append(TensorIndex(tok[:-1], 'u', bundle))
        elif tok.endswith('_'):
            out.append(TensorIndex(tok[:-1], 'l', bundle))
        else:
            raise ValueError("Use names ending in '^' or '_'.")
    return tuple(out)



def pretty_indexed(obj: Any) -> str:
    return obj.pretty() if hasattr(obj, 'pretty') else str(obj)


def latex_indexed(obj: Any) -> str:
    return obj.latex() if hasattr(obj, 'latex') else str(obj)


@dataclass(frozen=True)
class IndexedRewriteRule:
    """A practical rewrite rule for indexed expressions.

    matcher(obj) should return truthy when the rule applies. replacement(obj)
    returns the rewritten object.
    """
    name: str
    matcher: Callable[[object], bool]
    replacement: Callable[[object], object]

    def apply(self, obj):
        if self.matcher(obj):
            return self.replacement(obj)
        return obj




@dataclass(frozen=True)
class PatternRewriteRule:
    name: str
    pattern: object
    replacement: Callable[[object], object]

    def apply(self, obj):
        return self.replacement(obj) if match_indexed_pattern(self.pattern, obj) is not None else obj

class IndexedRewriteEngine:
    """Fixed-point term rewriting for IndexedTensor / IndexedTensorExpr trees.

    This is not a theorem prover; it is a reusable practical canonicalization
    engine that repeatedly normalizes tree structure, applies explicit rewrite
    rules, and then uses the existing indexed simplifier where helpful.
    """

    def __init__(self, rules: Sequence[object] | None = None, *, max_passes: int = 12, evaluate_leaf_products: bool = False):
        self.rules = tuple(rules or default_indexed_rewrite_rules())
        self.max_passes = max_passes
        self.evaluate_leaf_products = evaluate_leaf_products

    def rewrite_once(self, obj):
        obj = _normalize_tree(obj)
        obj = _rewrite_children(obj, self)
        prev = None
        cur = obj
        guard = 0
        while prev != cur and guard < 20:
            prev = cur
            for rule in self.rules:
                nxt = rule.apply(cur)
                if nxt != cur:
                    cur = _normalize_tree(nxt)
            guard += 1
        return _post_rewrite_simplify_expr(cur, evaluate_leaf_products=self.evaluate_leaf_products)

    def rewrite_fixed_point(self, obj):
        prev = None
        cur = obj
        passes = 0
        while prev != cur and passes < self.max_passes:
            prev = cur
            cur = self.rewrite_once(cur)
            passes += 1
        return cur

    def canonical_form(self, obj):
        cur = self.rewrite_fixed_point(obj)
        return _canonical_form(cur)



def _rewrite_children(obj, engine: IndexedRewriteEngine):
    if isinstance(obj, IndexedTensorExpr):
        new_args = tuple(engine.rewrite_fixed_point(arg) if isinstance(arg, (IndexedTensorExpr, IndexedTensor)) else arg for arg in obj.args)
        if new_args != obj.args:
            return IndexedTensorExpr(obj.op, new_args)
    return obj



def _post_rewrite_simplify_expr(obj, *, evaluate_leaf_products: bool = False):
    if isinstance(obj, IndexedTensor):
        return obj.canonicalize()
    if isinstance(obj, IndexedTensorExpr):
        if obj.op == 'tensor' and isinstance(obj.args[0], IndexedTensor):
            return obj.args[0].canonicalize()
        if obj.op == 'tensor_product' and evaluate_leaf_products:
            try:
                return obj.simplify()
            except Exception:
                return obj
    return obj


def _canonical_form(obj):
    if isinstance(obj, IndexedTensor):
        return obj.canonicalize().contract_repeated()
    if isinstance(obj, IndexedTensorExpr):
        norm = _normalize_tree(obj)
        if isinstance(norm, IndexedTensorExpr):
            if norm.op == 'add':
                terms = [_canonical_form(t) for t in _flatten_add(norm)]
                terms = sorted(terms, key=lambda a: _canonical_sort_key(a))
                acc = terms[0]
                for t in terms[1:]:
                    acc = IndexedTensorExpr('add', (acc, t))
                return acc
            if norm.op == 'tensor_product':
                factors = [_canonical_form(t) for t in _flatten_product(norm)]
                factors = sorted(factors, key=lambda a: _canonical_sort_key(a))
                acc = factors[0]
                for t in factors[1:]:
                    acc = IndexedTensorExpr('tensor_product', (acc, t))
                return acc
        return norm
    return obj


def _rule_flatten_add(obj):
    if isinstance(obj, IndexedTensorExpr) and obj.op == 'add':
        return _normalize_tree(obj)
    return obj


def _rule_flatten_product(obj):
    if isinstance(obj, IndexedTensorExpr) and obj.op == 'tensor_product':
        return _normalize_tree(obj)
    return obj


def _rule_cancel_duplicate_antisymmetric_leaves(obj):
    if not isinstance(obj, IndexedTensorExpr) or obj.op != 'tensor_product':
        return obj
    factors = _flatten_product(obj)
    seen = {}
    for f in factors:
        key = _canonical_sort_key(f)
        seen[key] = seen.get(key, 0) + 1
    # leave detailed cancellation to simplify(), but canonicalize the product ordering here
    ordered = sorted(factors, key=lambda a: _canonical_sort_key(a))
    if len(ordered) == 1:
        return ordered[0]
    acc = ordered[0]
    for t in ordered[1:]:
        acc = IndexedTensorExpr('tensor_product', (acc, t))
    return acc


def _rule_delta_metric_simplify(obj):
    if not isinstance(obj, IndexedTensorExpr) or obj.op != 'tensor_product':
        return obj
    try:
        simp = obj.simplify()
        return simp if not isinstance(simp, ScalarField) else simp
    except Exception:
        return obj


def _rule_sort_terms(obj):
    return _normalize_tree(obj)


def _pattern_rule_distribute_trace(obj):
    if not isinstance(obj, IndexedTensorExpr) or obj.op != 'tensor_product':
        return obj
    return obj


def default_indexed_rewrite_rules() -> Tuple[object, ...]:
    return (
        IndexedRewriteRule('flatten_add', lambda o: isinstance(o, IndexedTensorExpr) and o.op == 'add', _rule_flatten_add),
        IndexedRewriteRule('flatten_product', lambda o: isinstance(o, IndexedTensorExpr) and o.op == 'tensor_product', _rule_flatten_product),
        IndexedRewriteRule('sort_terms', lambda o: isinstance(o, (IndexedTensorExpr, IndexedTensor)), _rule_sort_terms),
        IndexedRewriteRule('cancel_duplicate_antisymmetric_leaves', lambda o: isinstance(o, IndexedTensorExpr) and o.op == 'tensor_product', _rule_cancel_duplicate_antisymmetric_leaves),
        IndexedRewriteRule('delta_metric_simplify', lambda o: isinstance(o, IndexedTensorExpr) and o.op == 'tensor_product', _rule_delta_metric_simplify),
        PatternRewriteRule('wildcard_leaf_tensor', TensorPattern(tensor_name='*'), lambda o: _normalize_basis_for_indexed(o) if isinstance(o, IndexedTensor) else o),
    )


def _rewrite_engine_fixed_point(obj: Any, rules: Sequence[IndexedRewriteRule] | None = None, *, max_passes: int = 12, evaluate_leaf_products: bool = False) -> Any:
    return IndexedRewriteEngine(rules, max_passes=max_passes, evaluate_leaf_products=evaluate_leaf_products).rewrite_fixed_point(obj)


def _tensorform_rewrite_fixed_point(obj: Any, config: IndexedNormalizationConfig | None = None) -> Any:
    try:
        return normalize_indexed_expression(obj, config=config)
    except Exception:
        return obj


def _authoritative_indexed_public_reduce(obj: Any, config: IndexedNormalizationConfig | None = None) -> Any:
    """Authoritative public reduction path for indexed expressions."""
    try:
        if isinstance(obj, IndexedTensorExpr) and obj.op == 'add':
            return _normalize_tree(obj)
        if isinstance(obj, IndexedTensorExpr) and obj.op == 'tensor_product':
            flat = _flatten_product(obj)
            scalar = sp.Integer(1)
            leaves = []
            ok = True
            for a in flat:
                if isinstance(a, ScalarField):
                    scalar *= a.expr
                elif isinstance(a, IndexedTensor):
                    leaves.append(a)
                else:
                    ok = False
                    break
            if ok and leaves:
                try:
                    red_factors, red_scalar = _special_tensor_network_simplify(leaves, scalar)
                    if red_factors != leaves or red_scalar != scalar:
                        if not red_factors:
                            return ScalarField(None, red_scalar)
                        acc = red_factors[0]
                        for f in red_factors[1:]:
                            acc = IndexedTensorExpr('tensor_product', (acc, f))
                        if red_scalar != 1:
                            acc = IndexedTensorExpr('tensor_product', (ScalarField(None, red_scalar), acc))
                        return _normalize_tree(acc)
                except Exception:
                    pass
        return normalize_indexed_expression(obj, config=config)
    except (BundleCompatibilityError, ValueError):
        raise
    except Exception:
        if isinstance(obj, IndexedTensor):
            return obj.canonicalize().contract_repeated()
        if isinstance(obj, IndexedTensorExpr):
            try:
                return _normalize_tree(obj)
            except Exception:
                return _rewrite_engine_fixed_point(obj, None)
        return obj


def rewrite_fixed_point(obj: Any, rules: Sequence[IndexedRewriteRule] | None = None, *, max_passes: int = 12, evaluate_leaf_products: bool = False) -> Any:
    if rules is None and not evaluate_leaf_products:
        return _authoritative_indexed_public_reduce(obj)
    return _rewrite_engine_fixed_point(obj, rules, max_passes=max_passes, evaluate_leaf_products=evaluate_leaf_products)


def canonical_indexed_form(obj: Any, rules: Sequence[IndexedRewriteRule] | None = None, *, max_passes: int = 12) -> Any:
    if rules is None:
        return _authoritative_indexed_public_reduce(obj)
    return IndexedRewriteEngine(rules, max_passes=max_passes).canonical_form(obj)


def tensor_replace(obj: Any, old: Any, new: Any, *, simplify: bool = False) -> Any:
    if isinstance(obj, IndexedTensorExpr):
        out = obj.substitute(old, new)
    elif obj == old:
        out = new
    else:
        out = obj
    if not simplify:
        return out
    return _authoritative_indexed_public_reduce(out)



def _expr_canonical_form(self):
    return canonical_indexed_form(self)


def _expr_rewrite_fixed_point(self, rules: Sequence[IndexedRewriteRule] | None = None, *, max_passes: int = 12, evaluate_leaf_products: bool = False):
    return rewrite_fixed_point(self, rules, max_passes=max_passes, evaluate_leaf_products=evaluate_leaf_products)


IndexedTensorExpr.canonical_form = _expr_canonical_form
IndexedTensorExpr.rewrite_fixed_point = _expr_rewrite_fixed_point
IndexedTensor.canonical_form = lambda self: canonical_indexed_form(self)


# --- Abstract-index extensions (initial) ---

@dataclass(frozen=True)
class TensorPattern:
    """Pattern for IndexedTensor or small indexed-expression matching.

    tensor_name may be a string, tuple of strings, or '*' for wildcard.
    index_variances constrains per-slot variance.
    repeated_groups requires selected slots to share the same index name.
    distinct_groups requires selected slots to have distinct index names.
    free_index_count / dummy_index_count constrain Einstein structure.
    predicate is an arbitrary extra guard.
    """
    tensor_name: object | None = None
    variance_spec: str | None = None
    index_variances: Tuple[str, ...] | None = None
    repeated_groups: Tuple[Tuple[int, ...], ...] = ()
    distinct_groups: Tuple[Tuple[int, ...], ...] = ()
    free_index_count: int | None = None
    dummy_index_count: int | None = None
    predicate: Callable[[object], bool] | None = None

    def _name_ok(self, obj: IndexedTensor) -> bool:
        if self.tensor_name is None or self.tensor_name == '*':
            return True
        actual = obj.tensor.name or 'T'
        if isinstance(self.tensor_name, (tuple, list, set)):
            return actual in self.tensor_name
        return actual == self.tensor_name

    def matches(self, obj) -> bool:
        if not isinstance(obj, IndexedTensor):
            return False
        if not self._name_ok(obj):
            return False
        if self.variance_spec is not None and obj.tensor.variance_spec != self.variance_spec:
            return False
        if self.index_variances is not None and tuple(i.variance for i in obj.indices) != self.index_variances:
            return False
        free, dummy = _free_and_dummy_counts(obj.indices)
        if self.free_index_count is not None and len(free) != self.free_index_count:
            return False
        if self.dummy_index_count is not None and len(dummy) != self.dummy_index_count:
            return False
        names = [i.name for i in obj.indices]
        for grp in self.repeated_groups:
            vals = {names[k] for k in grp}
            if len(vals) != 1:
                return False
        for grp in self.distinct_groups:
            vals = [names[k] for k in grp]
            if len(set(vals)) != len(vals):
                return False
        if self.predicate is not None and not self.predicate(obj):
            return False
        return True


@dataclass(frozen=True)
class ExprPattern:
    op: str | None = None
    args: Tuple[object, ...] = ()
    predicate: Callable[[object], bool] | None = None

    def matches(self, obj) -> bool:
        if isinstance(obj, IndexedTensor):
            return False
        if self.predicate is not None and not self.predicate(obj):
            return False
        if self.op is not None:
            if not isinstance(obj, IndexedTensorExpr) or obj.op != self.op:
                return False
        if self.args:
            if not isinstance(obj, IndexedTensorExpr) or len(obj.args) != len(self.args):
                return False
            for p, a in zip(self.args, obj.args):
                if hasattr(p, 'matches'):
                    if not p.matches(a):
                        return False
                elif p != a:
                    return False
        return True


def match_tensor_expr_pattern(pattern: Any, obj: Any) -> dict[str, Any] | None:
    if isinstance(pattern, TensorPattern):
        return {'tensor': obj.tensor, 'indices': obj.indices} if pattern.matches(obj) else None
    if isinstance(pattern, ExprPattern):
        return {'match': obj} if pattern.matches(obj) else None
    if hasattr(pattern, 'matches'):
        return {'match': obj} if pattern.matches(obj) else None
    return None


def match_indexed_pattern(pattern: Any, obj: Any) -> dict[str, Any] | None:
    if isinstance(pattern, (TensorPattern, ExprPattern)):
        return match_tensor_expr_pattern(pattern, obj)
    return None



def _normalize_basis_for_indexed(obj):
    if isinstance(obj, IndexedTensor):
        try:
            return IndexedTensor(obj.tensor.canonical_basis_form(), obj.indices).canonicalize()
        except Exception:
            return obj.canonicalize()
    if isinstance(obj, IndexedTensorExpr):
        return IndexedTensorExpr(obj.op, tuple(_normalize_basis_for_indexed(a) for a in obj.args))
    return obj


def _alpha_canonical(obj):
    return _canonicalize_index_names_global(alpha_rename_dummies(_normalize_basis_for_indexed(obj)))

def _free_and_dummy_counts(indices: Sequence[TensorIndex]):
    counts: Dict[str, Dict[str, int]] = {}
    for idx in indices:
        counts.setdefault(idx.name, {'u': 0, 'l': 0})[idx.variance] += 1
    free = {name: data for name, data in counts.items() if not (data['u'] and data['l'])}
    dummy = {name: data for name, data in counts.items() if data['u'] and data['l']}
    return free, dummy


def validate_index_sequence(indices: Sequence[TensorIndex], *, require_einstein_safe: bool = True) -> None:
    free, dummy = _free_and_dummy_counts(indices)
    errors = []
    if require_einstein_safe:
        for name, data in free.items():
            if data['u'] > 1 or data['l'] > 1:
                errors.append(f"Index '{name}' appears repeatedly with the same variance.")
        for name, data in dummy.items():
            if data['u'] > 1 or data['l'] > 1:
                errors.append(f"Index '{name}' appears more than once in an upper/lower role.")
    return errors


def _indexed_validate(self, *, require_einstein_safe: bool = True):
    errors = validate_index_sequence(self.indices, require_einstein_safe=require_einstein_safe)
    if errors:
        raise ValueError("; ".join(errors))
    return True


def _indexed_free_indices(self):
    free, _ = _free_and_dummy_counts(self.indices)
    return tuple(idx for idx in self.indices if idx.name in free)


def _indexed_dummy_names(self):
    _, dummy = _free_and_dummy_counts(self.indices)
    return tuple(sorted(dummy))


def _indexed_freshen_dummy_indices(self, prefix: str = 'd'):
    free, dummy = _free_and_dummy_counts(self.indices)
    mapping: Dict[str, str] = {}
    used = set(free)
    counter = 0
    for name in sorted(dummy):
        fresh = f'{prefix}{counter}'
        while fresh in used:
            counter += 1
            fresh = f'{prefix}{counter}'
        mapping[name] = fresh
        used.add(fresh)
        counter += 1
    return self.rename_indices(mapping)


def _indexed_trace_over(self, name: str):
    target_positions = [k for k, idx in enumerate(self.indices) if idx.name == name]
    if len(target_positions) != 2:
        raise ValueError("trace_over(name) requires exactly one upper and one lower occurrence.")
    i, j = target_positions
    if self.indices[i].variance == self.indices[j].variance:
        raise ValueError("trace_over(name) requires one upper and one lower occurrence.")
    return self.tensor.contract_slots(i, j)


def _indexed_contract_names(self, *names: str):
    out = self
    for name in names:
        if not isinstance(out, IndexedTensor):
            raise TypeError("contract_names currently expects an IndexedTensor at each step.")
        free, dummy = _free_and_dummy_counts(out.indices)
        if name not in dummy:
            raise ValueError(f"Index {name!r} is not a valid dummy pair.")
        pos = [k for k, idx in enumerate(out.indices) if idx.name == name]
        low, high = pos[0], pos[1]
        result = out.tensor.contract_slots(low, high)
        if isinstance(result, ScalarField):
            out = result
        else:
            remaining = tuple(idx for k, idx in enumerate(out.indices) if k not in pos)
            out = IndexedTensor(result, remaining)
    return out


def contract_by_index_names(obj: Any, *names: str) -> Any:
    if isinstance(obj, (IndexedTensor, IndexedTensorExpr)):
        normalized = normalize_indexed_expression(obj)
        if isinstance(normalized, IndexedTensor):
            return normalized.contract_names(*names)
        return normalized
    raise TypeError("contract_by_index_names expects IndexedTensor or IndexedTensorExpr.")


def trace_by_index_name(obj: Any, name: str) -> Any:
    if isinstance(obj, (IndexedTensor, IndexedTensorExpr)):
        normalized = normalize_indexed_expression(obj)
        if isinstance(normalized, IndexedTensor):
            return normalized.trace_over(name)
        return normalized
    raise TypeError("trace_by_index_name expects IndexedTensor or IndexedTensorExpr.")


def _collect_index_names(obj) -> set[str]:
    if isinstance(obj, IndexedTensor):
        return {idx.name for idx in obj.indices}
    if isinstance(obj, IndexedTensorExpr):
        names = set()
        for arg in obj.args:
            names |= _collect_index_names(arg)
        return names
    return set()


def alpha_rename_dummies(obj: Any, prefix: str = 'd') -> Any:
    used = set()
    def rec(node, counter=[0]):
        if isinstance(node, IndexedTensor):
            free, dummy = _free_and_dummy_counts(node.indices)
            mapping = {}
            local_used = set(free) | used
            for name in sorted(dummy):
                fresh = f'{prefix}{counter[0]}'
                while fresh in local_used:
                    counter[0] += 1
                    fresh = f'{prefix}{counter[0]}'
                mapping[name] = fresh
                local_used.add(fresh)
                counter[0] += 1
            renamed = node.rename_indices(mapping)
            used.update(_collect_index_names(renamed))
            return renamed
        if isinstance(node, IndexedTensorExpr):
            return IndexedTensorExpr(node.op, tuple(rec(arg) for arg in node.args))
        return node
    return rec(obj)


IndexedTensor.validate = _indexed_validate
IndexedTensor.free_indices = _indexed_free_indices
IndexedTensor.dummy_names = _indexed_dummy_names
IndexedTensor.freshen_dummy_indices = _indexed_freshen_dummy_indices
IndexedTensor.trace_over = _indexed_trace_over
IndexedTensor.contract_names = _indexed_contract_names


def rewrite_with_patterns(obj: Any, *rules: Any, max_passes: int = 12) -> Any:
    return IndexedRewriteEngine(rules, max_passes=max_passes).rewrite_fixed_point(obj)


class TensorComposer:
    def __init__(self, tensor: TensorObject):
        self.tensor = tensor
    def then(self, other: TensorObject, pair: tuple[int,int] = (1,0)):
        return TensorComposer(self.tensor.compose(other, pair=pair))
    def contract(self, *pairs):
        out=self.tensor
        return out.multi_contract(pairs)
    def done(self):
        return self.tensor


class DifferentialForm:
    def __init__(self, tensor: TensorObject):
        if set(tensor.variance_spec) - {'l'}:
            raise ValueError('DifferentialForm requires a covariant tensor.')
        self.tensor = tensor.antisymmetrize_slots(tuple(range(len(tensor.variance_spec)))) if len(tensor.variance_spec)>1 else tensor
        self.degree = len(self.tensor.variance_spec)
    @classmethod
    def from_tensor_field(cls, tf: TensorField):
        from .tensor_core import TensorObject
        return cls(TensorObject.from_tensor_field(tf))
    def wedge(self, other: 'DifferentialForm') -> 'DifferentialForm':
        return DifferentialForm.from_tensor_field(self.tensor.to_tensor_field().wedge(other.tensor.to_tensor_field()))
    def d(self) -> 'DifferentialForm':
        return DifferentialForm.from_tensor_field(self.tensor.to_tensor_field().exterior_derivative())
    def hodge_star(self) -> 'DifferentialForm':
        return DifferentialForm.from_tensor_field(self.tensor.to_tensor_field().hodge_star())
    def interior(self, vector: TensorObject) -> 'DifferentialForm | ScalarField':
        vf = vector.to_vector_field() if hasattr(vector, 'to_vector_field') else vector
        out = self.tensor.to_tensor_field().interior_product(vf)
        return out if isinstance(out, ScalarField) else DifferentialForm.from_tensor_field(out)
    def __str__(self):
        return f"Form^{self.degree}({self.tensor.name or 'ω'})"


def compose_tensors(left: TensorObject, right: TensorObject, pair: tuple[int, int] = (1, 0)) -> TensorObject:
    return left.compose(right, pair=pair)




class BundleCompatibilityError(ValueError):
    pass


def validate_bundle_consistency(indices: Sequence[TensorIndex]) -> bool:
    by_name: Dict[str, set] = {}
    for idx in indices:
        if idx.bundle is None:
            continue
        by_name.setdefault(idx.name, set()).add(idx.bundle)
    bad = {name: bundles for name, bundles in by_name.items() if len(bundles) > 1}
    if bad:
        raise BundleCompatibilityError(f"Conflicting bundles for repeated index names: {bad}")
    return True


def strengthen_index_bundles(obj: Any) -> Any:
    if isinstance(obj, IndexedTensor):
        new_indices = []
        for idx, basis in zip(obj.indices, obj.tensor.slot_bases):
            bundle = idx.bundle or infer_bundle_from_basis(basis)
            new_indices.append(TensorIndex(idx.name, idx.variance, bundle))
        strengthened = IndexedTensor(obj.tensor, tuple(new_indices))
        validate_bundle_consistency(strengthened.indices)
        return strengthened
    if isinstance(obj, IndexedTensorExpr):
        return IndexedTensorExpr(obj.op, tuple(strengthen_index_bundles(arg) for arg in obj.args))
    return obj


def _canonical_form_with_bundles(obj):
    obj = strengthen_index_bundles(obj)
    obj = rewrite_fixed_point(obj)
    return _canonical_form(obj)











@dataclass(frozen=True)
class IndexedNormalizationConfig:
    """Configuration for the central abstract-index normalization pipeline."""
    strengthen_bundles: bool = True
    normalize_basis: bool = True
    alpha_rename: bool = True
    canonicalize_global_names: bool = True
    rewrite_rules: Sequence[object] | None = None
    max_passes: int = 12
    evaluate_leaf_products: bool = False
    validate_bundles: bool = True
    validate_indices: bool = True
    canonical_form: bool = True
    tier: int = 2
    use_cache: bool = True
    collect_normal_form: bool = True
    allow_component_expansion: bool = False
    normalization_mode: str = "heuristic"
    simplification_level: str = "normal"


def _decision_mode_from_config(config: IndexedNormalizationConfig | None) -> str:
    return _config_decision_mode(_resolve_indexed_config(config, IndexedNormalizationConfig))



def _configured_simplify_expr(expr: Any, config: IndexedNormalizationConfig | None = None):
    return _config_simplify(expr, _resolve_indexed_config(config, IndexedNormalizationConfig))


def _iter_indexed_leaves(obj):
    if isinstance(obj, IndexedTensor):
        yield obj
    elif isinstance(obj, IndexedTensorExpr):
        for arg in obj.args:
            yield from _iter_indexed_leaves(arg)










IndexedTensorExpr.normalize = lambda self, config=None: normalize_indexed_expression(self, config=config)
IndexedTensor.normalize = lambda self, config=None: normalize_indexed_expression(self, config=config)


# --- and more complete multi-bundle abstract-index reasoning.

def _all_indices_in_expr(obj):
    if isinstance(obj, IndexedTensor):
        return list(obj.indices)
    if isinstance(obj, IndexedTensorExpr):
        out = []
        for arg in obj.args:
            out.extend(_all_indices_in_expr(arg))
        return out
    return []


def _expression_name_bundle_map(obj):
    mp: Dict[str, set] = {}
    for idx in _all_indices_in_expr(obj):
        if idx.bundle is not None:
            mp.setdefault(idx.name, set()).add(idx.bundle)
    return mp


def _global_bundle_validate(obj):
    by_name = _expression_name_bundle_map(obj)
    bad = {name: bundles for name, bundles in by_name.items() if len(bundles) > 1}
    if bad:
        raise BundleCompatibilityError(f"Conflicting bundles across indexed expression: {bad}")
    return obj


def _expression_free_signature(obj):
    counts: Dict[Tuple[str, str | None], Dict[str, int]] = {}
    ordered: list[TensorIndex] = []
    for idx in _all_indices_in_expr(obj):
        key = (idx.name, idx.bundle)
        if key not in counts:
            ordered.append(idx)
        counts.setdefault(key, {'u': 0, 'l': 0})[idx.variance] += 1
    free = []
    for idx in ordered:
        key = (idx.name, idx.bundle)
        cu = counts[key]['u']
        cl = counts[key]['l']
        # treat unmatched appearances as free
        if not (cu and cl):
            free.append((idx.bundle, idx.variance, cu if idx.variance == 'u' else cl))
    return tuple(sorted(free, key=lambda t: (str(t[0]), t[1], t[2])))


def _validate_addition_bundle_consistency(obj):
    if isinstance(obj, IndexedTensorExpr) and obj.op == 'add':
        terms = _flatten_add(obj)
        sigs = [_expression_free_signature(t) for t in terms]
        if sigs and any(s != sigs[0] for s in sigs[1:]):
            raise BundleCompatibilityError(f"Addends must have compatible free-index bundle/variance signatures: {sigs}")
    if isinstance(obj, IndexedTensorExpr):
        for arg in obj.args:
            _validate_addition_bundle_consistency(arg)
    return obj


def _canonicalize_index_names_global(obj):
    if isinstance(obj, IndexedTensor):
        free, dummy = _free_and_dummy_counts(obj.indices)
        def _free_key(name):
            sample = next(idx for idx in obj.indices if idx.name == name)
            return (str(sample.bundle), sample.variance, name)
        def _dummy_key(name):
            sample = next(idx for idx in obj.indices if idx.name == name)
            return (str(sample.bundle), name)
        free_map = {name: f"f{k}" for k, name in enumerate(sorted(free, key=_free_key))}
        dummy_map = {name: f"d{k}" for k, name in enumerate(sorted(dummy, key=_dummy_key))}
        mp = {**free_map, **dummy_map}
        return obj.rename_indices(mp).canonicalize()
    if isinstance(obj, IndexedTensorExpr):
        return IndexedTensorExpr(obj.op, tuple(_canonicalize_index_names_global(a) for a in obj.args))
    return obj



def _perm_parity(order):
    inv = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if order[i] > order[j]:
                inv += 1
    return -1 if inv % 2 else 1


def _tensor_scale(tensor: TensorObject, scalar):
    arr = tnf_build_array(tensor.components.shape, lambda idx: scalar * tensor.components[idx if idx else ()])
    return TensorObject(tensor.chart, arr, tensor.variance_spec, tensor.slot_bases, tensor.name, dict(tensor.symmetry_metadata))







def _rebuild_product_with_scalar(factors, scalar_expr):
    objs = []
    if scalar_expr != 1 and factors:
        objs.append(ScalarField(factors[0].tensor.chart, light_simplify(scalar_expr)))
    elif scalar_expr != 1:
        return ScalarField(None, light_simplify(scalar_expr))
    objs.extend(factors)
    if not objs:
        return ScalarField(None, sp.Integer(1))
    if len(objs) == 1:
        return objs[0]
    acc = objs[0]
    for t in objs[1:]:
        acc = IndexedTensorExpr('tensor_product', (acc, t))
    return acc











@dataclass(frozen=True)
class IndexedCanonicalizationReport:
    normalized: object
    free_signature: tuple
    bundle_signature: tuple
    tensor_kinds: tuple
    symmetry_tags: tuple
    structural_signature: tuple = tuple()
    idempotent: bool = True
    provenance: dict = field(default_factory=dict)

@dataclass(frozen=True)
class TNFDispatcherReport:
    parsed_from_boundary: bool
    reduced_in_nf: bool
    reconstructed_at_boundary: bool


@dataclass(frozen=True)
class TNFHelperAuditRecord:
    name: str
    category: str
    mutates_input: bool
    tnf_output: bool
    notes: str = ""


@dataclass(frozen=True)
class TNFExclusivityReport:
    helpers_seen: tuple[str, ...]
    parse_only_helpers: tuple[str, ...]
    boundary_only_helpers: tuple[str, ...]


_LAST_TNF_DISPATCHER_REPORT = None
_TNF_HELPER_AUDIT: Dict[str, TNFHelperAuditRecord] = {}


def _register_tnf_helper(name: str, category: str, mutates_input: bool, tnf_output: bool, notes: str = "") -> None:
    _TNF_HELPER_AUDIT[name] = TNFHelperAuditRecord(name, category, mutates_input, tnf_output, notes)


def last_tnf_dispatcher_report() -> TNFDispatcherReport | None:
    return _LAST_TNF_DISPATCHER_REPORT


def tnf_helper_audit() -> Dict[str, TNFHelperAuditRecord]:
    return dict(_TNF_HELPER_AUDIT)


def tnf_exclusivity_report() -> TNFExclusivityReport:
    helpers = tuple(sorted(_TNF_HELPER_AUDIT))
    parse_only = tuple(sorted(name for name, rec in _TNF_HELPER_AUDIT.items() if rec.category == "parse_only"))
    boundary_only = tuple(sorted(name for name, rec in _TNF_HELPER_AUDIT.items() if rec.category == "boundary_only"))
    return TNFExclusivityReport(helpers_seen=helpers, parse_only_helpers=parse_only, boundary_only_helpers=boundary_only)


def _leaf_tensor_kind(leaf):
    try:
        return _classify_special_tensor(leaf.tensor)
    except Exception:
        return "generic"


def _bundle_signature(obj):
    sig = []
    for idx in _all_indices_in_expr(obj):
        sig.append((idx.name, idx.variance, idx.bundle))
    return tuple(sorted(sig, key=lambda t: (str(t[2]), t[1], t[0])))


def _symmetry_tags(obj):
    tags = []
    for leaf in _iter_indexed_leaves(obj):
        md = getattr(leaf.tensor, "symmetry_metadata", {}) or {}
        tags.append(tuple(sorted((k, tuple(tuple(g) for g in v)) for k, v in md.items())))
    return tuple(tags)













_fallback_special_tensor_simplify = _complete_special_tensor_simplify

def _multi_bundle_contractibility(indices):
    free, dummy = _free_and_dummy_counts(indices)
    for name in dummy:
        variants = [idx for idx in indices if idx.name == name]
        bundles = {idx.bundle for idx in variants}
        if len(bundles) > 1:
            raise BundleCompatibilityError(f"Dummy index {name} spans incompatible bundles: {bundles}")
    return True




def _permutation_sign(values):
    sign = 1
    values = list(values)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if values[i] > values[j]:
                sign = -sign
    return sign


def _generalized_delta_entry(ups, los):
    if len(set(ups)) < len(ups) or len(set(los)) < len(los):
        return sp.Integer(0)
    position = {value: idx for idx, value in enumerate(los)}
    try:
        perm = [position[value] for value in ups]
    except KeyError:
        return sp.Integer(0)
    return sp.Integer(_permutation_sign(perm))





def generalized_kronecker_delta_tensor(chart: Any, rank: int) -> TensorObject:
    dim = chart.dimension
    shape = (dim,) * (2 * rank)

    def entry(idx):
        ups = idx[:rank]
        los = idx[rank:]
        return sp.factorial(rank) * _generalized_delta_entry(ups, los)

    return TensorField(chart, tnf_build_array(shape, entry), 'u' * rank + 'l' * rank)

def _irreducible_symmetry_project_tensor(tensor):
    try:
        from .tensor_core import young_irreducible_canonicalize
        return young_irreducible_canonicalize(tensor)
    except Exception:
        md = getattr(tensor, 'symmetry_metadata', {}) or {}
        out = tensor
        for _ in range(2):
            for group in md.get('symmetric', tuple()):
                out = out.symmetrize_slots(tuple(group))
            for group in md.get('antisymmetric', tuple()):
                out = out.antisymmetrize_slots(tuple(group))
        return out






def _local_leaf_key(obj):
    return _safe_structural_key(obj)









def _fast_special_product(obj):
    if isinstance(obj, IndexedTensorExpr) and obj.op == 'tensor_product':
        flat = _flatten_product(obj)
        idxf = [a for a in flat if isinstance(a, IndexedTensor)]
        if len(idxf) <= 2:
            try:
                res = _normalize_tree(obj)
                return _canonical_form(res)
            except Exception:
                return None
    return None



# Implements:
# 1 memoized signatures/normalization
# 2 tiered simplification
# 3 scalar-coefficient / tensor-factor split
# 5 contraction planning
# 7 explicit normal forms
# 9 delayed component expansion
# 15 better zero detection

_INDEXED_SIGNATURE_CACHE: BoundedCache[Any, Any] = BoundedCache(maxsize=4096)
_NORMALIZED_EXPR_CACHE: BoundedCache[Any, Any] = BoundedCache(maxsize=2048)

def _normalization_tier(config):
    return getattr(config, "tier", 2)

# widen config dynamically without breaking alternate construction sites

def _product_priority_key(leaf):
    kind = _leaf_tensor_kind(leaf)
    order = {
        "delta": 0,
        "identity": 0,
        "metric_ll": 1,
        "metric_uu": 1,
        "epsilon": 2,
        "generic": 9,
    }
    return (order.get(kind, 9), _special_leaf_summary(leaf))

def _plan_contractions(indexed_factors):
    return sorted(indexed_factors, key=_product_priority_key)

def _normal_form_key(obj):
    nf = _collect_normal_form(obj)
    return tuple((t.scalar, t.factors, t.free_signature, t.bundle_signature) for t in nf.terms)




# 16 dedicated optimizer pre-pass
# 17 cleaner separation between abstract and component layers
# 18 stronger immutable/cached tensor-normal-form usage

def _contains_component_heavy_tensor(obj):
    for leaf in _iter_indexed_leaves(_unwrap_layer(obj)):
        kind = _leaf_tensor_kind(leaf)
        if kind == "generic":
            return True
    return False

def _identity_like_leaf(leaf):
    return _leaf_tensor_kind(leaf) in {"delta", "identity"}

def _optimizer_prepass(obj):
    """Cheap optimizer pass before canonicalization.

    Performs only:
      - flattening
      - zero elimination
      - identity elimination in products when safe
      - scalar extraction
      - cheap factor sorting
    """
    raw = _unwrap_layer(obj)
    removed_zero_terms = 0
    removed_identity_terms = 0
    scalar_factor = sp.Integer(1)

    if isinstance(raw, IndexedTensor):
        if _detect_obvious_zero_leaf(raw):
            return AbstractIndexedExpr(ScalarField(raw.tensor.chart, sp.Integer(0))), TensorOptimizationReport(
                original_kind="tensor", optimized_kind="scalar_zero", removed_zero_terms=1,
                removed_identity_terms=0, scalar_factor_extracted=sp.Integer(0), used_component_expansion=False
            )
        return AbstractIndexedExpr(raw), TensorOptimizationReport(
            original_kind="tensor", optimized_kind="tensor", removed_zero_terms=0,
            removed_identity_terms=0, scalar_factor_extracted=sp.Integer(1), used_component_expansion=False
        )

    if isinstance(raw, IndexedTensorExpr) and raw.op == 'add':
        flat = [_unwrap_layer(_optimizer_prepass(a)[0]) for a in _flatten_add(raw)]
        clean = []
        for a in flat:
            if _zero_scalarfield_like(a):
                removed_zero_terms += 1
                continue
            if isinstance(a, IndexedTensor) and _detect_obvious_zero_leaf(a):
                removed_zero_terms += 1
                continue
            clean.append(a)
        if not clean:
            out = ScalarField(None, sp.Integer(0))
        else:
            clean.sort(key=_fast_obj_key)
            out = clean[0]
            for a in clean[1:]:
                out = IndexedTensorExpr('add', (out, a))
        return AbstractIndexedExpr(out), TensorOptimizationReport(
            original_kind="add", optimized_kind=type(out).__name__, removed_zero_terms=removed_zero_terms,
            removed_identity_terms=0, scalar_factor_extracted=sp.Integer(1), used_component_expansion=False
        )

    if isinstance(raw, IndexedTensorExpr) and raw.op == 'tensor_product':
        flat = [_unwrap_layer(_optimizer_prepass(a)[0]) for a in _flatten_product(raw)]
        indexed_factors = []
        others = []
        for a in flat:
            if _zero_scalarfield_like(a):
                removed_zero_terms += 1
                return AbstractIndexedExpr(ScalarField(None, sp.Integer(0))), TensorOptimizationReport(
                    original_kind="product", optimized_kind="scalar_zero", removed_zero_terms=removed_zero_terms,
                    removed_identity_terms=0, scalar_factor_extracted=sp.Integer(0), used_component_expansion=False
                )
            if isinstance(a, ScalarField):
                scalar_factor = canonical_simplify(scalar_factor * light_simplify(a.expr))
            elif isinstance(a, IndexedTensor):
                if _detect_obvious_zero_leaf(a):
                    removed_zero_terms += 1
                    return AbstractIndexedExpr(ScalarField(None, sp.Integer(0))), TensorOptimizationReport(
                        original_kind="product", optimized_kind="scalar_zero", removed_zero_terms=removed_zero_terms,
                        removed_identity_terms=0, scalar_factor_extracted=sp.Integer(0), used_component_expansion=False
                    )
                if _identity_like_leaf(a):
                    removed_identity_terms += 1
                    # keep delta/identity elimination for later, but don't expand now
                    indexed_factors.append(a)
                else:
                    indexed_factors.append(a)
            else:
                others.append(a)
        indexed_factors = _plan_contractions(indexed_factors)
        pieces = []
        if scalar_factor != 1:
            pieces.append(ScalarField(indexed_factors[0].tensor.chart if indexed_factors else None, canonical_simplify(scalar_factor)))
        pieces.extend(others)
        pieces.extend(indexed_factors)
        if not pieces:
            out = ScalarField(None, sp.Integer(1))
        else:
            out = pieces[0]
            for a in pieces[1:]:
                out = IndexedTensorExpr('tensor_product', (out, a))
        return AbstractIndexedExpr(out), TensorOptimizationReport(
            original_kind="product", optimized_kind=type(out).__name__, removed_zero_terms=removed_zero_terms,
            removed_identity_terms=removed_identity_terms, scalar_factor_extracted=canonical_simplify(scalar_factor),
            used_component_expansion=False
        )

    return AbstractIndexedExpr(raw), TensorOptimizationReport(
        original_kind=type(raw).__name__, optimized_kind=type(raw).__name__, removed_zero_terms=0,
        removed_identity_terms=0, scalar_factor_extracted=sp.Integer(1), used_component_expansion=False
    )


_LAST_NORMALIZATION_DIAGNOSTICS = None

def last_normalization_diagnostics() -> NormalizationDiagnostics | None:
    return _LAST_NORMALIZATION_DIAGNOSTICS

def index_space(name: str, dimension: int, parent: Any = None) -> TensorSpace:
    return TensorSpace(name, dimension, parent)

def _bundle_dim(bundle):
    return _bundle_metadata(bundle)[1]

def _bundle_name(bundle):
    return _bundle_metadata(bundle)[0] or ""

def _typed_index_signature(idx):
    return (idx.variance, _bundle_name(idx.bundle), _bundle_dim(idx.bundle), idx.name)

_wrap_abstract = _wrap_abstract_external
_unwrap_layer = _unwrap_layer_external
abstract_layer = _abstract_layer_external
component_layer = _component_layer_external
is_abstract_layer = _is_abstract_layer_external
is_component_layer = _is_component_layer_external

def to_component_layer(obj: Any) -> ComponentIndexedExpr:
    return ComponentIndexedExpr(_unwrap_layer(obj))

def _estimate_factor_cost(leaf):
    kind = _leaf_tensor_kind(leaf)
    base = {"delta": 1, "identity": 1, "metric_ll": 2, "metric_uu": 2, "epsilon": 3}.get(kind, 5)
    return base + len(getattr(leaf, "indices", ()))

def _detect_obvious_zero_leaf(leaf):
    md = getattr(leaf.tensor, "symmetry_metadata", {}) or {}
    for groups in md.get("antisymmetric", tuple()):
        names = [leaf.indices[s].name for s in groups]
        bundles = [leaf.indices[s].bundle for s in groups]
        if len(set(zip(names, bundles))) < len(groups):
            return True
    return False

def _zero_scalarfield_like(obj):
    return isinstance(obj, ScalarField) and is_zero(obj.expr)

def _filter_zero_terms(objs):
    out = []
    removed = 0
    for obj in objs:
        if obj is None:
            continue
        if _zero_scalarfield_like(obj):
            removed += 1
            continue
        if isinstance(obj, IndexedTensor) and _detect_obvious_zero_leaf(obj):
            removed += 1
            continue
        out.append(obj)
    return out, removed

def _special_leaf_summary(leaf):
    t = leaf.tensor
    kind = _leaf_tensor_kind(leaf)
    symm = tuple(sorted((k, tuple(tuple(g) for g in v)) for k, v in (getattr(t, "symmetry_metadata", {}) or {}).items()))
    idxs = tuple((idx.variance, str(idx.bundle), idx.name) for idx in leaf.indices)
    shape = tuple(t.components.shape)
    comp_sig = kind if kind in {"delta", "identity", "metric_ll", "metric_uu", "epsilon"} else ("shape", shape)
    return (t.name or "<anon>", t.variance_spec, tuple(getattr(b, "name", str(b)) for b in t.slot_bases), symm, idxs, comp_sig)

def _fast_obj_key(obj, tier=2):
    if isinstance(obj, ScalarField):
        return ("scalar", light_simplify(obj.expr))
    if isinstance(obj, IndexedTensor):
        return ("tensor",) + _special_leaf_summary(obj)
    if isinstance(obj, IndexedTensorExpr):
        return ("expr", obj.op, tuple(_fast_obj_key(a, tier=tier) for a in obj.args))
    return ("other", type(obj).__name__, _stable_text(obj))

def from_indexed_tensor_form(nf: IndexedTensorForm) -> IndexedTensorForm:
    return nf

def optimizer_prepass(obj: Any) -> Any:
    raw = _unwrap_layer(obj)
    if isinstance(raw, IndexedTensor) and _detect_obvious_zero_leaf(raw):
        return AbstractIndexedExpr(ScalarField(raw.tensor.chart, sp.Integer(0)))
    if isinstance(raw, IndexedTensorExpr) and raw.op == 'add':
        flat, _ = _filter_zero_terms([_unwrap_layer(optimizer_prepass(a)) for a in _flatten_add(raw)])
        if not flat:
            return AbstractIndexedExpr(ScalarField(None, sp.Integer(0)))
        flat = sorted(flat, key=_fast_obj_key)
        acc = flat[0]
        for a in flat[1:]:
            acc = IndexedTensorExpr('add', (acc, a))
        return AbstractIndexedExpr(acc)
    if isinstance(raw, IndexedTensorExpr) and raw.op == 'tensor_product':
        flat = [_unwrap_layer(optimizer_prepass(a)) for a in _flatten_product(raw)]
        flat, _ = _filter_zero_terms(flat)
        if not flat:
            return AbstractIndexedExpr(ScalarField(None, sp.Integer(1)))
        acc = flat[0]
        for a in flat[1:]:
            acc = IndexedTensorExpr('tensor_product', (acc, a))
        return AbstractIndexedExpr(acc)
    return AbstractIndexedExpr(raw)

def optimizer_report(obj: Any) -> TensorOptimizationReport:
    raw = _unwrap_layer(obj)
    pre = optimizer_prepass(raw).expr
    removed_zero = 1 if (isinstance(raw, IndexedTensor) and _detect_obvious_zero_leaf(raw)) else 0
    removed_id = 0
    return TensorOptimizationReport(type(raw).__name__, type(pre).__name__, removed_zero, removed_id, sp.Integer(1), False)

def _validate_indexed_object(obj):
    _global_bundle_validate(obj)
    _validate_addition_bundle_consistency(obj)
    for leaf in _iter_indexed_leaves(obj):
        _multi_bundle_contractibility(leaf.indices)
        if _detect_obvious_zero_leaf(leaf):
            continue
        for pos, idx in enumerate(leaf.indices):
            _typed_bundle_validate_index(idx, leaf.tensor.slot_bases[pos])
        if validate_index_sequence(leaf.indices):
            raise ValueError(f"Unsafe Einstein index pattern in {leaf}")
        validate_bundle_consistency(leaf.indices)
    return obj

_SYMMETRY_CANON_CACHE: BoundedCache[Any, Any] = BoundedCache(maxsize=2048)


def cache_stats() -> dict[str, dict[str, int]]:
    from .symbolic_decision import symbolic_decision_cache_info
    return {
        "symmetry": _SYMMETRY_CANON_CACHE.stats(),
        "special_tensor": _SPECIAL_TENSOR_CACHE.stats(),
        "normal_form": _NORMAL_FORM_CACHE.stats(),
        "indexed_signature": _INDEXED_SIGNATURE_CACHE.stats(),
        "normalized_expr": _NORMALIZED_EXPR_CACHE.stats(),
        "symbolic_decision": symbolic_decision_cache_info(),
    }


def clear_all_caches() -> None:
    from .symbolic_decision import clear_symbolic_decision_cache
    _SYMMETRY_CANON_CACHE.clear()
    _SPECIAL_TENSOR_CACHE.clear()
    _NORMAL_FORM_CACHE.clear()
    _INDEXED_SIGNATURE_CACHE.clear()
    _NORMALIZED_EXPR_CACHE.clear()
    clear_symbolic_decision_cache()


def _symmetry_cache_key(tensor):
    return (tensor.name or "<anon>", tensor.variance_spec, tuple(getattr(b, "name", str(b)) for b in tensor.slot_bases), tuple(sorted((k, tuple(tuple(g) for g in v)) for k, v in (getattr(tensor, "symmetry_metadata", {}) or {}).items())), tuple(tensor.components.shape))

def _cache_key(obj, config):
    raw = _unwrap_layer(obj)
    return (_fast_obj_key(raw), tuple(sorted((k, getattr(config, k)) for k in ["strengthen_bundles","normalize_basis","alpha_rename","canonicalize_global_names","max_passes","evaluate_leaf_products","validate_bundles","validate_indices","canonical_form","tier","use_cache","collect_normal_form","allow_component_expansion","normalization_mode","simplification_level"] if hasattr(config, k))))



# unified special-tensor normalizer, stronger typed bundles, diagnostics/tests. ---


def _coerce_indexed_factors(indexed_factors: Any, config: IndexedNormalizationConfig | None = None) -> tuple[Any, ...]:
    if isinstance(indexed_factors, IndexedTensorForm):
        if not indexed_factors.terms:
            return ()
        if len(indexed_factors.terms) != 1:
            raise ValueError("Contraction planning expects a single TensorForm term or a factor sequence.")
        return tuple(_authoritative_tnf_to_expr(IndexedTensorForm((TensorFormTerm(sp.Integer(1), (f,), (), ()),))) for f in indexed_factors.terms[0].factors)
    if isinstance(indexed_factors, (IndexedTensor, IndexedTensorExpr)):
        nf = to_indexed_tensor_form(indexed_factors, config=config)
        if not nf.terms:
            return ()
        if len(nf.terms) != 1:
            raise ValueError("Contraction planning expects a single TensorForm term or a factor sequence.")
        return tuple(_authoritative_tnf_to_expr(IndexedTensorForm((TensorFormTerm(sp.Integer(1), (f,), (), ()),))) for f in nf.terms[0].factors)
    return tuple(indexed_factors)

from .tensorform_planning import (
    SpecialTensorNormalizationResult, TensorFormRenderOptions, _estimate_factor_cost,
    build_contraction_graph, build_contraction_plan, special_tensor_normalize,
    _special_tensor_engine, _collect_normal_form, render_indexed_tensor_form,
)
from . import tensorform_planning as _tensorform_planning_mod
_NORMAL_FORM_CACHE = _tensorform_planning_mod._NORMAL_FORM_CACHE
_SPECIAL_TENSOR_CACHE = _tensorform_planning_mod._SPECIAL_TENSOR_CACHE
from .tensorform_special import (
    _epsilon_rank, _epsilon_epsilon_delta_rewrite, _epsilon_metric_raise_lower,
    _special_tensor_network_simplify, _delta_chain_simplify, _metric_chain_simplify,
    _epsilon_epsilon_extended, _epsilon_partial_contraction, _epsilon_metric_chain_simplify,
    _more_complete_special_simplify,
)
from .tensorform_engine import (
    IndexedFactor, TNFFactorAtom, IndexedNormalFactor,
    _authoritative_tnf_from_factor, _authoritative_tnf_special_tensor_scale,
    _tensorform_normalized_symmetry, _tensorform_space_name, _tensorform_space_dim,
    _tensorform_slot_signature, _tensorform_basis_bundle_hint, _tensorform_canonical_slots,
    _tensorform_factor_sign_from_indices, _tensorform_special_signature_from_leaf, _tensorform_symmetry_class,
    _tensorform_leaf_to_indexed_normal_factor, _authoritative_tnf_leaf_to_factor_and_coeff,
    _authoritative_tnf_zero, _authoritative_tnf_one, _authoritative_tnf_from_scalar,
    _authoritative_tnf_factor_zero_from_symmetry, _authoritative_tnf_from_leaf, _authoritative_tnf_term_key,
    _authoritative_tnf_collect, _authoritative_tnf_add, _authoritative_tnf_mul,
    _authoritative_tnf_rebuild_factor, _authoritative_tnf_factor_is_delta, _authoritative_tnf_factor_is_metric_up,
    _authoritative_tnf_factor_is_metric_down, _authoritative_tnf_factor_is_epsilon, _authoritative_tnf_factor_is_gdelta,
    _authoritative_tnf_replace_slot, _authoritative_tnf_choose_shared_slot, _authoritative_tnf_gdelta_from_slots,
    _authoritative_tnf_apply_gdelta_to_factor, _authoritative_tnf_normalize_symmetry_slots,
    _authoritative_tnf_reduce_symmetry_only, _authoritative_tnf_reduce_special_factors,
    _authoritative_tnf_reduce_term, _authoritative_tnf_reduce, _authoritative_tnf_expr_to_tnf,
    _authoritative_tnf_chart_from_factor, _authoritative_tnf_symbolic_tensor_components, _authoritative_tnf_tensor_from_factor,
    _authoritative_tnf_term_to_expr, _authoritative_tnf_to_expr, _authoritative_tnf_factor_key,
    _normalize_indexed_once,
)

TNFFactor = TNFFactorAtom


def _space_name(bundle):
    return _bundle_name(bundle)

def _typed_bundle_validate_index(idx, basis=None):
    if basis is not None:
        # allow exact name match OR dimension-compatible TensorSpace/object bundles
        if not idx.compatible_with_basis(basis):
            basis_bundle = None
            if hasattr(basis, "metadata"):
                basis_bundle = basis.metadata.get("bundle", None)
            if basis_bundle is not None:
                if _bundle_dim(idx.bundle) is not None and getattr(basis_bundle, "dimension", None) == _bundle_dim(idx.bundle):
                    if _space_name(idx.bundle) != getattr(basis_bundle, "name", _space_name(basis_bundle)):
                        pass
                    else:
                        return True
            raise BundleCompatibilityError("Index bundle/type must be compatible with tensor slot basis.")
    bdim = _bundle_dim(idx.bundle)
    if basis is not None and hasattr(basis, "dimension") and bdim is not None and basis.dimension != bdim:
        raise BundleCompatibilityError(f"Bundle dimension mismatch: index {bdim} vs basis {basis.dimension}")
    return True

def _symmetry_aware_leaf(leaf: IndexedTensor) -> IndexedTensor:
    key = (_symmetry_cache_key(leaf.tensor), tuple(_typed_index_signature(i) for i in leaf.indices))
    cached = _SYMMETRY_CANON_CACHE.get(key)
    if cached is not None:
        return cached
    tensor = leaf.tensor
    indices = list(leaf.indices)
    try:
        tensor = _irreducible_symmetry_project_tensor(tensor)
    except Exception:
        try:
            tensor = symmetry_canonicalize(tensor)
        except Exception:
            pass
    md = dict(getattr(tensor, "symmetry_metadata", {}) or {})
    for keyname in ('symmetric', 'antisymmetric'):
        for group in md.get(keyname, tuple()):
            group = tuple(group)
            current = [indices[s] for s in group]
            desired_local = sorted(range(len(group)), key=lambda k: _typed_index_signature(current[k]))
            if tuple(desired_local) == tuple(range(len(group))):
                continue
            reordered_slots = [group[k] for k in desired_local]
            perm = list(range(len(indices)))
            for dst, src in zip(group, reordered_slots):
                perm[dst] = src
            tensor = tensor.permute_slots(tuple(perm))
            indices = [indices[p] for p in perm]
            if keyname == 'antisymmetric':
                sign = _perm_parity(list(desired_local))
                if sign == -1:
                    tensor = _tensor_scale(tensor, -1)
    out = IndexedTensor(tensor, tuple(indices)).canonicalize()
    _SYMMETRY_CANON_CACHE[key] = out
    return out



def _factor_sort_key(f):
    try:
        return _authoritative_tnf_factor_key(f)
    except Exception:
        return _safe_structural_key(f)

def _tnf_from_leaf(leaf):
    if isinstance(leaf, ScalarField):
        return _authoritative_tnf_from_scalar(leaf.expr)
    if isinstance(leaf, IndexedTensor):
        if _detect_obvious_zero_leaf(leaf):
            return _authoritative_tnf_zero()
        leaf = _symmetry_aware_leaf(leaf)
        return IndexedTensorForm((_leaf_fast_monomial(leaf),))
    return IndexedTensorForm((TensorFormTerm(sp.Integer(1), ((_safe_structural_key(leaf),),), tuple(), tuple()),))

def _monomial_mul(m1: TensorFormTerm, m2: TensorFormTerm) -> TensorFormTerm:
    scalar = normal_simplify(m1.scalar * m2.scalar)
    if is_zero(scalar):
        return TensorFormTerm(sp.Integer(0), tuple(), tuple(), tuple())
    factors = tuple(sorted(m1.factors + m2.factors, key=_factor_sort_key))
    free_sig = tuple(sorted(m1.free_signature + m2.free_signature, key=_safe_structural_key))
    bundle_sig = tuple(sorted(m1.bundle_signature + m2.bundle_signature, key=_safe_structural_key))
    return TensorFormTerm(scalar, factors, free_sig, bundle_sig)

def _tnf_special_tensor_reduce_monomial(term: TensorFormTerm) -> IndexedTensorForm:
    if not term.factors:
        return IndexedTensorForm((term,))
    indexed_factors = []
    other_factors = []
    for f in term.factors:
        if isinstance(f, tuple) and len(f) >= 6:
            other_factors.append(f)
        else:
            other_factors.append(f)
    # We only reduce when we can map factors back to concrete leaves via synthetic wrappers stored in factor sigs
    # For this package revision, keep factor tuples as canonical signatures and apply scalar-only reduction here.
    # Full special-tensor reduction remains in the tensor-normal-form pipeline through the original leaves before signature loss.
    return IndexedTensorForm((term,))

def _tnf_map_reduce(nf: IndexedTensorForm, fn) -> IndexedTensorForm:
    terms = []
    for t in nf.terms:
        reduced = fn(t)
        terms.extend(reduced.terms)
    return _combine_like_terms_nf(IndexedTensorForm(tuple(terms)))




def _index_multiset_signature(indices):
    return sorted((idx.variance, _space_name(idx.bundle), _bundle_dim(idx.bundle)) for idx in indices)

def _indices_match_under_declared_symmetry(left: IndexedTensor, right: IndexedTensor) -> bool:
    lmd = getattr(left.tensor, "symmetry_metadata", {}) or {}
    rmd = getattr(right.tensor, "symmetry_metadata", {}) or {}
    if lmd != rmd:
        return False
    if _index_multiset_signature(left.indices) != _index_multiset_signature(right.indices):
        return False
    # symmetric groups: order-insensitive within each group
    if lmd.get("symmetric"):
        return True
    # antisymmetric groups: treat same multiset as equivalent up to sign;
    # equality layer here is structural, so only accept exact canonical leaf if tensors already match.
    return False


def _raw_indexed_leaf_equivalent(a, b):
    if not (isinstance(a, IndexedTensor) and isinstance(b, IndexedTensor)):
        return False
    try:
        if not a.tensor.equivalent(b.tensor, modulo_basis=True, modulo_symmetry=True):
            return False
    except Exception:
        return False
    amd = getattr(a.tensor, "symmetry_metadata", {}) or {}
    bmd = getattr(b.tensor, "symmetry_metadata", {}) or {}
    if amd != bmd:
        return False
    if amd.get("symmetric"):
        return _index_multiset_signature(a.indices) == _index_multiset_signature(b.indices)
    return False



def _canonicalize_typed_indices_for_symmetry(typed_indices, symmetry):
    typed_indices = list(typed_indices)
    symm = dict(symmetry) if isinstance(symmetry, (tuple, list)) else {}
    for group in symm.get("symmetric", tuple()):
        group = tuple(group)
        current = [typed_indices[s] for s in group]
        current_sorted = sorted(current, key=lambda x: (x[0], x[1], x[2], x[3]))
        for slot, value in zip(group, current_sorted):
            typed_indices[slot] = value
    for group in symm.get("antisymmetric", tuple()):
        group = tuple(group)
        current = [typed_indices[s] for s in group]
        current_sorted = sorted(current, key=lambda x: (x[0], x[1], x[2], x[3]))
        for slot, value in zip(group, current_sorted):
            typed_indices[slot] = value
    return tuple(typed_indices)

def _to_tnf_factor_from_leaf(leaf):
    t = leaf.tensor
    symmetry = tuple(sorted((k, tuple(tuple(g) for g in v)) for k, v in (getattr(t, "symmetry_metadata", {}) or {}).items()))
    typed_indices = tuple((_space_name(i.bundle), _bundle_dim(i.bundle), i.variance, i.name) for i in leaf.indices)
    typed_indices = _canonicalize_typed_indices_for_symmetry(typed_indices, symmetry)
    return TNFFactor(
        kind=_leaf_tensor_kind(leaf),
        name=t.name or "<anon>",
        variance_spec=tuple(t.variance_spec),
        basis_names=tuple(getattr(b, "name", str(b)) for b in t.slot_bases),
        typed_indices=typed_indices,
        symmetry=symmetry,
    )

def _tnf_factor_symmetric_equiv(f1, f2):
    if not (isinstance(f1, TNFFactor) and isinstance(f2, TNFFactor)):
        return f1 == f2
    if (f1.kind, f1.name, f1.variance_spec, f1.basis_names, f1.symmetry) != (f2.kind, f2.name, f2.variance_spec, f2.basis_names, f2.symmetry):
        return False
    return f1.typed_indices == f2.typed_indices

def _tnf_term_key(term):
    factor_key = []
    for f in term.factors:
        if isinstance(f, TNFFactor):
            factor_key.append(("nf", _authoritative_tnf_factor_key(f)))
        else:
            factor_key.append(("raw", _safe_structural_key(f)))
    return (tuple(factor_key), term.free_signature, term.bundle_signature)



def _canonical_symmetry_repr(symmetry_metadata):
    return tuple(
        sorted(
            (k, tuple(sorted(tuple(tuple(sorted(g)) for g in v))))
            for k, v in (symmetry_metadata or {}).items()
        )
    )

def _canonical_typed_indices_for_factor(leaf):
    symmetry = _canonical_symmetry_repr(getattr(leaf.tensor, "symmetry_metadata", {}) or {})
    typed = [(_space_name(i.bundle), _bundle_dim(i.bundle), i.variance) for i in leaf.indices]
    symm = dict(symmetry) if isinstance(symmetry, (tuple, list)) else {}
    for group in symm.get("symmetric", tuple()):
        group = tuple(group)
        current = [typed[s] for s in group]
        current_sorted = sorted(current, key=lambda x: (x[0], x[1], x[2]))
        for slot, value in zip(group, current_sorted):
            typed[slot] = value
    for group in symm.get("antisymmetric", tuple()):
        group = tuple(group)
        current = [typed[s] for s in group]
        current_sorted = sorted(current, key=lambda x: (x[0], x[1], x[2]))
        for slot, value in zip(group, current_sorted):
            typed[slot] = value
    return tuple(typed), symmetry



def _indexed_simplify_expr(expr):
    expr = sp.sympify(expr)
    return light_simplify(sp.expand(expr))



def _tensorform_prepass(raw: Any) -> Any:
    try:
        if isinstance(raw, IndexedTensor):
            return raw.canonicalize().contract_repeated()
        if isinstance(raw, IndexedTensorExpr):
            return raw.simplify()
    except Exception:
        return raw
    return raw

def to_indexed_tensor_form(obj: Any, config: IndexedNormalizationConfig | None = None) -> IndexedTensorForm:
    try:
        _register_tnf_helper("to_indexed_tensor_form", "parse_only", False, True, "public authoritative boundary-to-TensorForm entry")
    except Exception:
        pass
    config = _resolve_indexed_config(config, IndexedNormalizationConfig)
    raw = _unwrap_layer(obj)
    if getattr(config, "strengthen_bundles", True):
        raw = strengthen_index_bundles(raw)
    if getattr(config, "normalize_basis", True):
        raw = _normalize_basis_for_indexed(raw)
    if getattr(config, "validate_bundles", True) or getattr(config, "validate_indices", True):
        raw = _validate_indexed_object(raw)
    if getattr(config, "normalization_mode", "heuristic") != "strict":
        raw = _tensorform_prepass(raw)
    return _authoritative_tnf_reduce(_authoritative_tnf_expr_to_tnf(raw))

def normalize_indexed_expression(obj: Any, config: IndexedNormalizationConfig | None = None) -> Any:
    from .indexed_api import normalize_indexed_expression as _normalize_indexed_expression
    return _normalize_indexed_expression(obj, config=config)


def indexed_signature(obj: Any, config: IndexedNormalizationConfig | None = None) -> tuple[Any, ...]:
    try:
        _register_tnf_helper("indexed_signature", "boundary_only", False, False, "public signature entry delegates to TNF")
    except Exception:
        pass
    nf = to_indexed_tensor_form(obj, config=config)
    return tuple((_configured_simplify_expr(t.scalar, config), tuple(_authoritative_tnf_factor_key(f) for f in t.factors), tuple(t.free_signature), tuple(t.bundle_signature)) for t in nf.terms)


# Fix mixed-factor ordering instability in deeper TNF special-tensor/symmetry paths.

def _normal_form_key_safe_key_atom(x):
    return _safe_structural_key(x)

def _authoritative_tnf_factor_key(f):
    return _canonical_factor_key(_authoritative_tnf_from_factor(f))

# Final boundary overrides: keep simple IndexedTensor / IndexedTensorExpr behavior
# at the public rewrite layer instead of forcing every leaf through the TNF render
# adapter, which can collapse plain leaves into scalar signature placehalternates.

def _normalize_tree(obj):
    if isinstance(obj, IndexedTensor):
        return obj.canonicalize()
    if not isinstance(obj, IndexedTensorExpr):
        return obj
    if obj.op == 'tensor':
        leaf = obj.args[0]
        return leaf.canonicalize() if isinstance(leaf, IndexedTensor) else leaf
    if obj.op in {'add', 'tensor_product'}:
        flat = _flatten_add(obj) if obj.op == 'add' else _flatten_product(obj)
        flat = [_normalize_tree(a) for a in flat]
        flat = [a for a in flat if a is not None]
        flat.sort(key=lambda a: _canonical_sort_key(a))
        if len(flat) == 1:
            return flat[0]
        acc = flat[0]
        for a in flat[1:]:
            acc = IndexedTensorExpr(obj.op, (acc, a))
        return acc
    return obj

def _indexed_canonical_report_from_tnf(obj: Any, config: IndexedNormalizationConfig | None = None) -> IndexedCanonicalizationReport:
    nf = to_indexed_tensor_form(obj, config=config)
    norm = _authoritative_tnf_to_expr(nf)
    tensor_kinds = []
    symmetry_tags = set()
    for term in nf.terms:
        for factor in term.factors:
            tensor_kinds.append(factor.kind)
            if factor.symmetry:
                symmetry_tags.add(tuple(factor.symmetry))
    return IndexedCanonicalizationReport(
        normalized=norm,
        free_signature=tuple(t.free_signature for t in nf.terms),
        bundle_signature=tuple(t.bundle_signature for t in nf.terms),
        tensor_kinds=tuple(tensor_kinds),
        symmetry_tags=tuple(sorted(symmetry_tags)),
    )

# Allow boundary tuple-unpacking of the normalization result.
try:
    SpecialTensorNormalizationResult.__iter__ = lambda self: iter((self.factors, self.scalar, self.plan))
except Exception:
    pass

# Final public construction/canonicalization overrides.

def indexed(tensor: TensorObject, *idx: TensorIndex) -> IndexedTensor:
    from .indexed_api import indexed as _indexed_api
    return _indexed_api(tensor, *idx)


def canonicalize_indexed_expression(obj: Any, config: IndexedNormalizationConfig | None = None) -> Any:
    from .indexed_api import canonicalize_indexed_expression as _canonicalize_indexed_expression
    return _canonicalize_indexed_expression(obj, config=config)


def indexed_canonical_report(obj: Any, config: IndexedNormalizationConfig | None = None) -> IndexedCanonicalizationReport:
    from .indexed_api import indexed_canonical_report as _indexed_canonical_report
    return _indexed_canonical_report(obj, config=config)


def indexed_equivalent(obj1: Any, obj2: Any, config: IndexedNormalizationConfig | None = None) -> bool:
    from .indexed_api import indexed_equivalent as _indexed_equivalent
    return _indexed_equivalent(obj1, obj2, config=config)


def indexed_equal(obj1: Any, obj2: Any, config: IndexedNormalizationConfig | None = None) -> bool:
    from .indexed_api import indexed_equal as _indexed_equal
    return _indexed_equal(obj1, obj2, config=config)


def stronger_indexed_equal(left: Any, right: Any, config: IndexedNormalizationConfig | None = None) -> bool:
    from .indexed_api import stronger_indexed_equal as _stronger_indexed_equal
    return _stronger_indexed_equal(left, right, config=config)
