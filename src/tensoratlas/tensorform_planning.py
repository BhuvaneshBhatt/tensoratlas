from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import sympy as sp

from .symbolic_decision import is_zero, light_simplify, canonical_simplify, is_equal
from .bundle_identity import bundle_key, bundle_name, bundle_dim
from .canonical_keys import structural_key
from .simplification_policy import normal_simplify
from .cache_utils import BoundedCache
from .tensorform_types import ContractionPlan, IndexedTensorForm, TensorFormTerm
from .fields import ScalarField


@dataclass(frozen=True)
class SpecialTensorNormalizationResult:
    factors: tuple
    scalar: object
    plan: object


@dataclass(frozen=True)
class TensorFormRenderOptions:
    multiline: bool = True
    show_bundles: bool = True
    show_coefficients: bool = True


_SPECIAL_TENSOR_CACHE: BoundedCache[Any, SpecialTensorNormalizationResult] = BoundedCache(maxsize=2048)
_NORMAL_FORM_CACHE: BoundedCache[Any, IndexedTensorForm] = BoundedCache(maxsize=4096)


def _ti():
    from . import tensor_indices as ti
    return ti


def _estimate_factor_cost(leaf):
    ti = _ti()
    kind = ti._leaf_tensor_kind(leaf)
    base = {"delta": 1, "identity": 1, "metric_ll": 2, "metric_uu": 2, "epsilon": 3}.get(kind, 5)
    return base + len(getattr(leaf, "indices", ()))


def build_contraction_graph(indexed_factors: Sequence[Any] | IndexedTensorForm | Any, config: Any | None = None) -> dict[int, dict[int, int]]:
    ti = _ti()
    indexed_factors = ti._coerce_indexed_factors(indexed_factors, config=config)
    graph: dict[int, dict[int, int]] = {i: {} for i in range(len(indexed_factors))}
    incidence: dict[tuple[str, str | None], dict[str, dict[int, int]]] = {}
    for factor_idx, factor in enumerate(indexed_factors):
        for idx in factor.indices:
            key = (idx.name, bundle_key(idx.bundle))
            bucket = incidence.setdefault(key, {"u": {}, "l": {}})
            side = bucket[idx.variance]
            side[factor_idx] = side.get(factor_idx, 0) + 1
    for bucket in incidence.values():
        uppers = bucket["u"]
        lowers = bucket["l"]
        for ui, ucount in uppers.items():
            for li, lcount in lowers.items():
                if ui == li:
                    continue
                weight = ucount * lcount
                a, b = sorted((ui, li))
                graph[a][b] = graph[a].get(b, 0) + weight
                graph[b][a] = graph[b].get(a, 0) + weight
    return graph


def build_contraction_plan(indexed_factors: Sequence[Any] | IndexedTensorForm | Any, config: Any | None = None) -> ContractionPlan:
    ti = _ti()
    indexed_factors = ti._coerce_indexed_factors(indexed_factors, config=config)
    graph = build_contraction_graph(indexed_factors, config=config)
    priorities = []
    ordered = []
    remaining = set(range(len(indexed_factors)))
    while remaining:
        scored = []
        for i in remaining:
            degree = sum(graph.get(i, {}).values())
            scored.append((-degree, ti._product_priority_key(indexed_factors[i]), i))
        _, prio, chosen = min(scored)
        ordered.append(indexed_factors[chosen])
        priorities.append((prio, dict(graph.get(chosen, {}))))
        remaining.remove(chosen)
    edge_cost = sum(weight for i, nbrs in graph.items() for j, weight in nbrs.items() if i < j)
    cost = sum(_estimate_factor_cost(f) for f in ordered) + edge_cost
    return ContractionPlan(tuple(ordered), tuple(priorities), cost)


def _typed_factor_signature(leaf):
    ti = _ti()
    return (
        ti._leaf_tensor_kind(leaf),
        leaf.tensor.name or "<anon>",
        leaf.tensor.variance_spec,
        tuple(getattr(b, "name", str(b)) for b in leaf.tensor.slot_bases),
        tuple((bundle_name(i.bundle), bundle_dim(i.bundle), i.variance, i.name) for i in leaf.indices),
        tuple(sorted((k, tuple(tuple(g) for g in v)) for k, v in (getattr(leaf.tensor, "symmetry_metadata", {}) or {}).items())),
    )


def _tnf_cache_key(obj):
    ti = _ti()
    try:
        return ("nf", structural_key(obj))
    except Exception:
        return ("tnf_raw", type(obj).__name__, getattr(obj, "name", None), getattr(obj, "op", None), getattr(obj, "variance_spec", None))


def _special_tensor_cache_key(indexed_factors, scalar_expr):
    return (tuple(_typed_factor_signature(f) for f in indexed_factors), normal_simplify(scalar_expr))


def special_tensor_normalize(indexed_factors: Sequence[Any] | IndexedTensorForm | Any, scalar_expr: sp.Expr | None = None, config: Any | None = None) -> SpecialTensorNormalizationResult:
    ti = _ti()
    if scalar_expr is None and isinstance(indexed_factors, IndexedTensorForm):
        if len(indexed_factors.terms) != 1:
            raise ValueError("Special-tensor normalization expects a single TensorForm term when given an IndexedTensorForm.")
        term = indexed_factors.terms[0]
        scalar_expr = term.scalar
        indexed_factors = tuple(ti._authoritative_tnf_to_expr(IndexedTensorForm((TensorFormTerm(sp.Integer(1), (f,), (), ()),))) for f in term.factors)
    elif scalar_expr is None and isinstance(indexed_factors, (ti.IndexedTensor, ti.IndexedTensorExpr)):
        nf = ti.to_indexed_tensor_form(indexed_factors, config=config)
        if len(nf.terms) != 1:
            raise ValueError("Special-tensor normalization expects a single TensorForm term when given an indexed expression.")
        term = nf.terms[0]
        scalar_expr = term.scalar
        indexed_factors = tuple(ti._authoritative_tnf_to_expr(IndexedTensorForm((TensorFormTerm(sp.Integer(1), (f,), (), ()),))) for f in term.factors)
    elif scalar_expr is None:
        scalar_expr = sp.Integer(1)
    indexed_factors = tuple(indexed_factors)
    key = _special_tensor_cache_key(indexed_factors, scalar_expr)
    cached = _SPECIAL_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached
    plan = build_contraction_plan(indexed_factors, config=config)
    factors = list(plan.ordered_factors)
    scalar = canonical_simplify(scalar_expr)
    changed = True
    while changed:
        changed = False
        done, newf = ti._delta_chain_simplify(factors)
        if done:
            factors = list(newf)
            changed = True
            continue
        done, newf = ti._metric_chain_simplify(factors)
        if done:
            factors = list(newf)
            changed = True
            continue
        done, newf, news = ti._epsilon_metric_chain_simplify(factors, scalar)
        if done:
            factors, scalar = list(newf), canonical_simplify(news)
            changed = True
            continue
        done, newf, news = ti._epsilon_epsilon_extended(factors, scalar)
        if done:
            factors, scalar = list(newf), canonical_simplify(news)
            changed = True
            continue
    res = SpecialTensorNormalizationResult(tuple(factors), scalar, plan)
    _SPECIAL_TENSOR_CACHE[key] = res
    return res


def _special_tensor_engine(indexed_factors, scalar_expr):
    res = special_tensor_normalize(indexed_factors, scalar_expr)
    return list(res.factors), res.scalar, res.plan


def _collect_normal_form(obj):
    ti = _ti()
    key = _tnf_cache_key(obj)
    cached = _NORMAL_FORM_CACHE.get(key)
    if cached is not None:
        return cached
    if isinstance(obj, ScalarField):
        nf = IndexedTensorForm((TensorFormTerm(canonical_simplify(obj.expr), tuple(), tuple(), tuple()),))
        _NORMAL_FORM_CACHE[key] = nf
        return nf
    if isinstance(obj, ti.IndexedTensor):
        nf = IndexedTensorForm((ti._leaf_fast_monomial(obj),))
        _NORMAL_FORM_CACHE[key] = nf
        return nf
    if isinstance(obj, ti.IndexedTensorExpr) and obj.op == 'add':
        terms = []
        for a in ti._flatten_add(obj):
            terms.extend(_collect_normal_form(a).terms)
        nf = ti._combine_like_terms_nf(IndexedTensorForm(tuple(terms)))
        _NORMAL_FORM_CACHE[key] = nf
        return nf
    if isinstance(obj, ti.IndexedTensorExpr) and obj.op == 'tensor_product':
        scalar = sp.Integer(1)
        factors = []
        free_sig = []
        bundle_sig = []
        for a in ti._flatten_product(obj):
            nf = _collect_normal_form(a)
            if len(nf.terms) != 1:
                nf = ti._combine_like_terms_nf(nf)
            if len(nf.terms) != 1:
                raw = IndexedTensorForm((TensorFormTerm(sp.Integer(1), (("raw", str(obj)),), tuple(), tuple()),))
                _NORMAL_FORM_CACHE[key] = raw
                return raw
            t = nf.terms[0]
            scalar = light_simplify(scalar * t.scalar)
            factors.extend(t.factors)
            free_sig.extend(t.free_signature)
            bundle_sig.extend(t.bundle_signature)
        mono = TensorFormTerm(
            scalar=canonical_simplify(scalar),
            factors=tuple(sorted(factors)),
            free_signature=tuple(sorted(free_sig, key=str)),
            bundle_signature=tuple(sorted(bundle_sig, key=str)),
        )
        nf = IndexedTensorForm((mono,))
        _NORMAL_FORM_CACHE[key] = nf
        return nf
    nf = IndexedTensorForm((TensorFormTerm(sp.Integer(1), (("raw", str(obj)),), tuple(), tuple()),))
    _NORMAL_FORM_CACHE[key] = nf
    return nf


def render_indexed_tensor_form(nf: IndexedTensorForm, options: TensorFormRenderOptions | None = None) -> str:
    options = options or TensorFormRenderOptions()
    if not nf.terms:
        return "0"
    pieces = []
    for term in nf.terms:
        coeff = "" if (not options.show_coefficients or is_equal(light_simplify(term.scalar), sp.Integer(1))) else f"{term.scalar} * "
        facs = []
        for f in term.factors:
            if isinstance(f, tuple) and len(f) >= 5:
                kind, name, varspec, _bases, idxs, _symm = f[:6]
                idxtxt = []
                for bundle_name, bundle_dim, variance, idxname in idxs:
                    base = idxname
                    if options.show_bundles and bundle_name:
                        base += f":{bundle_name}"
                    idxtxt.append(("^" if variance == "u" else "_") + base)
                facs.append(f"{name}[{kind}]({', '.join(idxtxt)})")
            else:
                facs.append(str(f))
        body = " ⊗ ".join(facs) if facs else "1"
        pieces.append(coeff + body)
    sep = "\n+ " if options.multiline else " + "
    return sep.join(pieces)
