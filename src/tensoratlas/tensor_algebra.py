from __future__ import annotations

from itertools import permutations, product
from math import factorial
from dataclasses import dataclass, field
import time
from typing import Any, Iterable, Mapping, Sequence

import sympy as sp

from .symbolic_decision import is_equal, is_zero
from .simplification_core import light_simplify, canonical_simplify
from .simplification_policy import normal_simplify
from .canonical_keys import structural_key as _tensor_structural_key

from .fields import ScalarField, TensorField, VectorField
from .mappings import CoordinateMap
from .normal_forms import TNFTensorArray, as_tnf_array, tnf_array_to_sympy, tnf_build_array, tnf_scalar_array


def _rank0_tensor_array(expr):
    return tnf_scalar_array(expr, cleaner=normal_simplify)


def _as_tensor(obj):
    if isinstance(obj, TensorObject):
        return obj.to_tensor_field()
    if isinstance(obj, TensorField):
        return obj
    if isinstance(obj, VectorField):
        return obj.as_tensor()
    if isinstance(obj, ScalarField):
        return TensorField(obj.chart, _rank0_tensor_array(obj.expr), "")
    raise TypeError(f"Unsupported tensor-like object: {type(obj)!r}")



def _tensor_product_entry(idx, tensors, ranks):
    total = sp.Integer(1)
    pos = 0
    for tensor, rank in zip(tensors, ranks):
        tensor_idx = idx[pos:pos + rank]
        total *= tensor.components[tensor_idx] if rank else tensor.components[()]
        pos += rank
    return total


def _indexed_factor_signature(factor):
    try:
        name = getattr(factor.tensor, 'name', None) or 'T'
        symmetry = tuple(sorted((k, tuple(tuple(g) for g in v)) for k, v in getattr(factor.tensor, 'symmetry_metadata', {}).items()))
        variances = tuple(i.variance for i in getattr(factor, 'indices', ()))
        bundles = tuple(getattr(i, 'bundle', None) for i in getattr(factor, 'indices', ()))
        return (name, symmetry, variances, bundles, len(getattr(factor, 'indices', ())))
    except Exception:
        return (str(factor),)


def _build_indexed_hypergraph(obj: Any):
    try:
        from .tensor_indices import IndexedTensor, IndexedTensorExpr
    except Exception:
        IndexedTensor = IndexedTensorExpr = tuple()
    expr = getattr(obj, 'expr', obj)
    factors = [f for f in _flatten_tensor_product_expr(expr) if isinstance(f, IndexedTensor)]
    counts = {}
    incidences = {}
    for fi, factor in enumerate(factors):
        for si, idx in enumerate(factor.indices):
            data = counts.setdefault(idx.name, {'u': 0, 'l': 0, 'bundle': idx.bundle})
            data[idx.variance] += 1
            incidences.setdefault(idx.name, []).append((fi, si, idx.variance))
    factor_nodes = []
    for fi, factor in enumerate(factors):
        factor_nodes.append({
            'factor_id': fi,
            'signature': _indexed_factor_signature(factor),
            'indices': tuple(idx.name for idx in factor.indices),
            'variances': tuple(idx.variance for idx in factor.indices),
        })
    index_nodes = {}
    for name, data in counts.items():
        part = 'dummy' if data['u'] and data['l'] else 'free'
        index_nodes[name] = {
            'name': name,
            'partition': part,
            'bundle': data['bundle'],
            'up': data['u'],
            'down': data['l'],
            'degree': len(incidences.get(name, ())),
            'incidences': tuple((fi, si) for fi, si, _ in incidences.get(name, ())),
        }
    return factors, factor_nodes, index_nodes


def _wl_refine_indexed_hypergraph(factor_nodes, index_nodes, *, max_rounds: int = 8):
    factor_labels = {node['factor_id']: node['signature'] for node in factor_nodes}
    index_labels = {
        name: (data['partition'], data['bundle'], data['up'], data['down'], data['degree'])
        for name, data in index_nodes.items()
    }
    for _ in range(max_rounds):
        new_factor = {}
        for node in factor_nodes:
            neighborhood = tuple((index_labels[name], var) for name, var in zip(node['indices'], node['variances']))
            new_factor[node['factor_id']] = (factor_labels[node['factor_id']], neighborhood)
        new_index = {}
        for name, data in index_nodes.items():
            incident = tuple(sorted((factor_labels[fi], factor_nodes[fi]['variances'][si]) for fi, si in data['incidences']))
            new_index[name] = (index_labels[name], incident)
        if new_factor == factor_labels and new_index == index_labels:
            break
        factor_labels, index_labels = new_factor, new_index
    return factor_labels, index_labels


def _rename_indexed_factor_names(factor, mapping):
    if not mapping:
        return factor
    return factor.rename_indices(mapping)
    total = sp.Integer(1)
    pos = 0
    for tensor, rank in zip(tensors, ranks):
        tensor_idx = idx[pos:pos + rank]
        total *= tensor.components[tensor_idx] if rank else tensor.components[()]
        pos += rank
    return total

def identity_tensor(chart: Any, variance_spec: str = "ul") -> TensorField:
    if variance_spec != "ul":
        raise NotImplementedError("identity_tensor currently supports variance_spec='ul' only.")
    dim = chart.dimension
    arr = tnf_build_array((dim, dim), lambda idx: sp.Integer(1) if idx[0] == idx[1] else sp.Integer(0))
    return TensorField(chart, arr, variance_spec)


def tensor_product(*objs: Any) -> Any:
    try:
        from .tensor_indices import IndexedTensor, IndexedTensorExpr
    except Exception:
        IndexedTensor = IndexedTensorExpr = tuple()
    if any(isinstance(obj, (IndexedTensor, IndexedTensorExpr)) for obj in objs):
        from .tensor_indices import IndexedTensorExpr
        if not objs:
            raise ValueError("tensor_product requires at least one input.")
        acc = objs[0]
        for obj in objs[1:]:
            acc = IndexedTensorExpr('tensor_product', (acc, obj))
        return acc
    tensors = [_as_tensor(obj) for obj in objs]
    if not tensors:
        raise ValueError("tensor_product requires at least one input.")
    chart = tensors[0].chart
    if any(t.chart != chart for t in tensors[1:]):
        raise ValueError("All tensor factors must live on the same chart.")
    variance = "".join(t.variance_spec for t in tensors)
    dim = chart.dimension
    rank = len(variance)
    if rank == 0:
        total = sp.Integer(1)
        for t in tensors:
            total *= t.components[()]
        return TensorField(chart, _rank0_tensor_array(total), "")
    ranks = [len(t.variance_spec) for t in tensors]
    out = tnf_build_array((dim,) * rank, lambda idx: _tensor_product_entry(idx, tensors, ranks))
    return TensorField(chart, out, variance)


def tensor_transpose(tensor: TensorField, perm: Sequence[int]) -> TensorField:
    perm = tuple(perm)
    rank = len(tensor.variance_spec)
    if sorted(perm) != list(range(rank)):
        raise ValueError("perm must be a permutation of range(rank).")
    obj = TensorObject.from_tensor_field(tensor)
    if perm == tuple(range(rank)):
        return tensor
    if rank == 2 and perm == (1, 0):
        if symmetric_tensor_q(obj):
            return tensor
        if antisymmetric_tensor_q(obj):
            return tensor_from_components(obj.chart, obj.components.applyfunc(lambda e: -e), obj.variance_spec, obj.slot_bases, symmetry_metadata=obj.symmetry_metadata, domain_metadata=obj.domain_metadata).to_tensor_field()
    out = tensor.components.permutedims(perm)
    variance = "".join(tensor.variance_spec[i] for i in perm)
    return TensorField(tensor.chart, out, variance)


def tensor_permute(obj: Any, perm: Sequence[int]) -> Any:
    """Public slot-permutation wrapper for tensor-like objects."""
    if isinstance(obj, TensorObject):
        return obj.permute_slots(tuple(perm))
    if isinstance(obj, TensorField):
        return tensor_transpose(obj, perm)
    if isinstance(obj, TensorExpr) and obj.op == 'tensor_product':
        return tensor_sort(obj)
    try:
        from .tensor_indices import IndexedTensor, IndexedTensorExpr
    except Exception:
        IndexedTensor = IndexedTensorExpr = tuple()
    if isinstance(obj, (IndexedTensor, IndexedTensorExpr)):
        # Keep indexed objects on the symbolic path.
        return obj
    t = _as_tensor(obj)
    return tensor_transpose(t, perm)


def tensor_symmetrize(obj: Any, slots: Sequence[int] | None = None) -> Any:
    """Symmetrize a tensor-like object over the specified slots."""
    if isinstance(obj, TensorObject):
        slots = tuple(range(len(obj.variance_spec)) if slots is None else slots)
        return obj.symmetrize_slots(slots)
    t = _as_tensor(obj)
    slots = tuple(range(len(t.variance_spec)) if slots is None else slots)
    return symmetrize_slots(t, slots=slots, antisymmetric=False)


def tensor_antisymmetrize(obj: Any, slots: Sequence[int] | None = None) -> Any:
    """Antisymmetrize a tensor-like object over the specified slots."""
    if isinstance(obj, TensorObject):
        slots = tuple(range(len(obj.variance_spec)) if slots is None else slots)
        return obj.antisymmetrize_slots(slots)
    t = _as_tensor(obj)
    slots = tuple(range(len(t.variance_spec)) if slots is None else slots)
    return symmetrize_slots(t, slots=slots, antisymmetric=True)


def tensor_has_symmetry(obj: Any, spec: Mapping[str, Sequence[Sequence[int]]] | str) -> bool:
    """Return True when a tensor advertises or satisfies a symmetry specification."""
    rep = tensor_symmetry(obj)
    if isinstance(spec, str):
        return bool(rep.get(spec))
    for key, groups in spec.items():
        have = {tuple(g) for g in rep.get(key, tuple())}
        need = {tuple(g) for g in groups}
        if not need.issubset(have):
            return False
    return True


def tensor_project_symmetry(obj: Any, spec: Mapping[str, Sequence[Sequence[int]]] | str) -> Any:
    """Project a tensor onto the requested symmetry class using slot symmetrization."""
    out = obj
    if isinstance(spec, str):
        if spec == 'symmetric':
            return tensor_symmetrize(out)
        if spec == 'antisymmetric':
            return tensor_antisymmetrize(out)
        raise ValueError(f'Unsupported symmetry spec string: {spec!r}')
    for key, groups in spec.items():
        for group in groups:
            if key == 'symmetric':
                out = tensor_symmetrize(out, tuple(group))
            elif key == 'antisymmetric':
                out = tensor_antisymmetrize(out, tuple(group))
    return out


def tensor_symmetry_class(obj: Any) -> str:
    """Summarize the most prominent symmetry class of a tensor-like object."""
    rep = tensor_symmetry(obj)
    if rep.get('antisymmetric') and rep.get('symmetric'):
        return 'mixed'
    if rep.get('antisymmetric'):
        return 'antisymmetric'
    if rep.get('symmetric'):
        return 'symmetric'
    if hermitian_tensor_q(obj):
        return 'hermitian'
    if antihermitian_tensor_q(obj):
        return 'antihermitian'
    return 'generic'


def symmetrize_slots(tensor: TensorField, slots: Iterable[int] | None = None, antisymmetric: bool = False) -> TensorField:
    rank = len(tensor.variance_spec)
    slots = tuple(range(rank) if slots is None else slots)
    if any(s < 0 or s >= rank for s in slots):
        raise ValueError("Invalid slot index.")
    if len(set(slots)) != len(slots):
        raise ValueError("Duplicate slots are not allowed.")
    perms = list(permutations(range(len(slots))))
    norm = sp.Integer(factorial(len(slots)))
    out = tnf_build_array(tensor.components.shape, lambda idx: _symmetrized_entry(tensor, idx, slots, perms, antisymmetric, norm))
    return TensorField(tensor.chart, out, tensor.variance_spec)


def _perm_parity(perm):
    inversions = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _symmetrized_entry(tensor, idx, slots, perms, antisymmetric, norm):
    total = sp.Integer(0)
    for perm in perms:
        source = list(idx)
        for local_src, local_dst in enumerate(perm):
            source[slots[local_src]] = idx[slots[local_dst]]
        sign = sp.Integer(_perm_parity(perm)) if antisymmetric else sp.Integer(1)
        total += sign * tensor.components[tuple(source)]
    return normal_simplify(total / norm)


def tensor_expand(obj: Any) -> TensorField:
    t = _as_tensor(obj)
    out = t.components.applyfunc(sp.expand)
    return TensorField(t.chart, out, t.variance_spec)



from .tensor_core import TensorObject, TensorExpr, StructuredTensorArray, tensor_from_components, tensor_simplify, tensor_expand as expr_tensor_expand, tensor_reduce as expr_tensor_reduce



def _is_scalar_like_tensor_object(obj: Any) -> bool:
    return isinstance(obj, TensorObject) and len(obj.variance_spec) == 0



def _tensor_structural_key(obj: Any) -> tuple:
    try:
        from . import tensor_indices as ti
        return _tensor_structural_key(obj)
    except Exception:
        if isinstance(obj, ScalarField):
            return ('scalar', sp.srepr(light_simplify(obj.expr)))
        if isinstance(obj, TensorExpr):
            return ('expr', obj.op, tuple(_tensor_structural_key(a) for a in obj.args))
        return (type(obj).__name__, str(obj))

def _tensor_name_key(obj: Any) -> tuple:
    try:
        from .tensor_indices import IndexedTensor
    except Exception:
        IndexedTensor = tuple()
    if isinstance(obj, IndexedTensor):
        base = obj.tensor.name or ''
        inds = tuple((i.variance, i.name, i.bundle) for i in obj.indices)
        return ('indexed', base, inds)
    if isinstance(obj, TensorObject):
        return ('tensor_object', obj.name or '', obj.variance_spec, tuple((b.name, b.kind) for b in obj.slot_bases))
    if isinstance(obj, TensorField):
        return ('tensor_field', getattr(obj, 'variance_spec', ''), getattr(obj, 'chart', None).__class__.__name__)
    if isinstance(obj, ScalarField):
        return ('scalar', sp.srepr(light_simplify(obj.expr)))
    if isinstance(obj, TensorExpr):
        return ('expr', obj.op, tuple(_tensor_name_key(a) for a in obj.args))
    return _tensor_structural_key(obj)


def _flatten_tensor_product_expr(obj: Any) -> list[Any]:
    if isinstance(obj, TensorExpr) and obj.op == 'tensor_product':
        return _flatten_tensor_product_expr(obj.args[0]) + _flatten_tensor_product_expr(obj.args[1])
    try:
        from .tensor_indices import IndexedTensorExpr as _IndexedTensorExpr
    except Exception:
        _IndexedTensorExpr = tuple()
    if isinstance(obj, _IndexedTensorExpr) and obj.op == 'tensor_product':
        return _flatten_tensor_product_expr(obj.args[0]) + _flatten_tensor_product_expr(obj.args[1])
    return [obj]


def _rebuild_tensor_product_expr(factors: Sequence[Any], *, symbolic: bool = False) -> Any:
    if not factors:
        raise ValueError('Need at least one factor to rebuild tensor product.')
    acc = factors[0]
    for factor in factors[1:]:
        if symbolic:
            try:
                from .tensor_indices import IndexedTensor, IndexedTensorExpr
            except Exception:
                IndexedTensor = IndexedTensorExpr = tuple()
            if isinstance(acc, (IndexedTensor, IndexedTensorExpr)) or isinstance(factor, (IndexedTensor, IndexedTensorExpr)):
                from .tensor_indices import IndexedTensorExpr
                acc = IndexedTensorExpr('tensor_product', (acc, factor))
            else:
                acc = TensorExpr('tensor_product', (acc, factor))
        else:
            acc = tensor_product(acc, factor)
    return acc


def tensor_sort(obj: Any) -> Any:
    """Canonicalize tensor-product factor order for public tensor expressions."""
    if isinstance(obj, TensorExpr) and obj.op == 'tensor_product':
        factors = [_eval_expr(f) if isinstance(f, TensorExpr) else f for f in _flatten_tensor_product_expr(obj)]
        factors = sorted(factors, key=_tensor_name_key)
        return _rebuild_tensor_product_expr(factors, symbolic=True)
    try:
        from .tensor_indices import IndexedTensor, IndexedTensorExpr
    except Exception:
        IndexedTensor = IndexedTensorExpr = tuple()
    if isinstance(obj, IndexedTensorExpr) and obj.op == 'tensor_product':
        factors = [f for f in _flatten_tensor_product_expr(obj)]
        factors = sorted(factors, key=_tensor_name_key)
        return _rebuild_tensor_product_expr(factors, symbolic=True)
    return obj


def symmetric_tensor_q(obj: Any, slots: Sequence[int] | None = None, tolerance: float | None = None) -> bool:
    rep = tensor_symmetry(obj, tolerance=tolerance)
    groups = rep.get('symmetric', tuple())
    if slots is None:
        return bool(groups)
    return tuple(slots) in groups


def antisymmetric_tensor_q(obj: Any, slots: Sequence[int] | None = None, tolerance: float | None = None) -> bool:
    rep = tensor_symmetry(obj, tolerance=tolerance)
    groups = rep.get('antisymmetric', tuple())
    if slots is None:
        return bool(groups)
    return tuple(slots) in groups


def hermitian_tensor_q(obj: Any, tolerance: float | None = None) -> bool:
    arr = None
    if isinstance(obj, TensorObject):
        arr = obj.components
    elif isinstance(obj, TensorField) and len(obj.variance_spec) == 2:
        arr = obj.components
    if arr is None or len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        return False
    dim = arr.shape[0]
    for i in range(dim):
        for j in range(dim):
            if not _approx_zero(arr[(i, j)] - sp.conjugate(arr[(j, i)]), tolerance):
                return False
    return True


def antihermitian_tensor_q(obj: Any, tolerance: float | None = None) -> bool:
    arr = None
    if isinstance(obj, TensorObject):
        arr = obj.components
    elif isinstance(obj, TensorField) and len(obj.variance_spec) == 2:
        arr = obj.components
    if arr is None or len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        return False
    dim = arr.shape[0]
    for i in range(dim):
        for j in range(dim):
            if not _approx_zero(arr[(i, j)] + sp.conjugate(arr[(j, i)]), tolerance):
                return False
    return True


def tensor_symmetry_report(obj: Any, tolerance: float | None = None) -> dict[str, Any]:
    rep = dict(tensor_symmetry(obj, tolerance=tolerance))
    rep['rank'] = tensor_rank(obj)
    rep['dimensions'] = tensor_dimensions(obj)
    rep['is_symmetric'] = symmetric_tensor_q(obj, tolerance=tolerance)
    rep['is_antisymmetric'] = antisymmetric_tensor_q(obj, tolerance=tolerance)
    rep['is_hermitian'] = hermitian_tensor_q(obj, tolerance=tolerance)
    rep['is_antihermitian'] = antihermitian_tensor_q(obj, tolerance=tolerance)
    rep['symmetry_class'] = tensor_symmetry_class(obj)
    rep['supported_queries'] = ('symmetric', 'antisymmetric', 'hermitian', 'antihermitian')
    return rep


def tensor_conjugate_transpose(obj: Any, perm: Sequence[int] | None = None) -> Any:
    """Conjugate-transpose a rank-2 tensor-like object with basic symmetry simplifications."""
    perm = tuple((1, 0) if perm is None else perm)
    if isinstance(obj, TensorObject):
        transposed = obj.permute_slots(perm).simplify(canonicalize_symmetry=False)
        arr = transposed.components.applyfunc(sp.conjugate)
        out = tensor_from_components(obj.chart, arr, transposed.variance_spec, transposed.slot_bases, name=obj.name, symmetry_metadata=transposed.symmetry_metadata, domain_metadata=transposed.domain_metadata)
        if perm == (1, 0):
            if hermitian_tensor_q(obj):
                return obj
            if antihermitian_tensor_q(obj):
                return tensor_from_components(obj.chart, obj.components.applyfunc(lambda e: -e), obj.variance_spec, obj.slot_bases, name=obj.name, symmetry_metadata=obj.symmetry_metadata, domain_metadata=obj.domain_metadata)
        return out
    if isinstance(obj, TensorField):
        return tensor_conjugate_transpose(TensorObject.from_tensor_field(obj))
    if isinstance(obj, sp.MatrixBase):
        return obj.conjugate().transpose()
    raise TypeError(f'Unsupported tensor-like object: {type(obj)!r}')


def _split_indexed_connected_components(obj: Any) -> tuple[Any, tuple[tuple[int, ...], ...], tuple[str, ...]]:
    from .tensor_indices import IndexedTensor, IndexedTensorExpr, build_contraction_graph, canonical_indexed_form
    if not isinstance(obj, (IndexedTensor, IndexedTensorExpr)):
        return obj, tuple(), tuple()
    factors = _flatten_tensor_product_expr(obj)
    if len(factors) <= 1:
        return obj, tuple((0,),) if factors else tuple(), tuple()
    graph = build_contraction_graph(factors)
    seen = set()
    comps = []
    for node in range(len(factors)):
        if node in seen:
            continue
        stack = [node]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(n for n, w in graph.get(cur, {}).items() if w and n not in seen)
        comps.append(tuple(sorted(comp)))
    if len(comps) == 1:
        return obj, tuple(comps), tuple()
    rebuilt = []
    notes = ['separated disconnected contraction components']
    for comp in comps:
        rebuilt.append(_rebuild_tensor_product_expr([factors[i] for i in comp], symbolic=True) if len(comp) > 1 else factors[comp[0]])
    rebuilt = [canonical_indexed_form(part) for part in rebuilt]
    rebuilt = sorted(rebuilt, key=_tensor_name_key)
    return _rebuild_tensor_product_expr(rebuilt, symbolic=True), tuple(comps), tuple(notes)


def _separate_scalar_factors(obj: Any) -> tuple[Any, sp.Expr, tuple[str, ...]]:
    notes = []
    if isinstance(obj, TensorExpr) and obj.op == 'tensor_product':
        factors = [_eval_expr(f) if isinstance(f, TensorExpr) else f for f in _flatten_tensor_product_expr(obj)]
        scalar = sp.Integer(1)
        nonscalar = []
        for factor in factors:
            if isinstance(factor, ScalarField):
                scalar *= factor.expr
            elif _is_scalar_like_tensor_object(factor):
                scalar *= factor.components[()]
            else:
                nonscalar.append(factor)
        if scalar != 1:
            notes.append('separated scalar tensor-product factors')
        if not nonscalar:
            chart = _as_tensor(factors[0]).chart if factors else None
            final_scalar = canonical_simplify(scalar, final=True)
            return ScalarField(chart, final_scalar) if chart is not None else final_scalar, sp.Integer(1), tuple(notes)
        rebuilt = _rebuild_tensor_product_expr(sorted(nonscalar, key=_tensor_name_key))
        return rebuilt, canonical_simplify(scalar, final=True), tuple(notes)
    return obj, sp.Integer(1), tuple()


def push_forward(mapping: CoordinateMap, obj: Any) -> Any:
    if isinstance(obj, ScalarField):
        raise TypeError("Use pull_back for scalar fields; push-forward is for contravariant objects.")
    if isinstance(obj, VectorField):
        return obj.transform(mapping)
    t = _as_tensor(obj)
    return t.transform(mapping)


def pull_back(mapping: CoordinateMap, obj: Any) -> Any:
    if isinstance(obj, ScalarField):
        return obj.transform(mapping)
    if isinstance(obj, VectorField):
        return obj.transform(mapping)
    t = _as_tensor(obj)
    return t.transform(mapping)


def _approx_zero(expr: sp.Expr, tolerance: float | None = None) -> bool:
    expr = sp.sympify(expr)
    if tolerance is None:
        return is_zero(expr)
    if expr == 0:
        return True
    try:
        return abs(complex(sp.N(expr))) <= tolerance
    except Exception:
        return is_zero(expr)


def tensor_q(obj: Any) -> bool:
    return isinstance(obj, (TensorField, VectorField, ScalarField, TensorObject, StructuredTensorArray))


def tensor_dimensions(obj: Any) -> tuple[int, ...]:
    if isinstance(obj, StructuredTensorArray):
        return tuple(obj.shape)
    if isinstance(obj, TensorObject):
        return tuple(obj.components.shape)
    if isinstance(obj, TensorField):
        return tuple(obj.components.shape)
    if isinstance(obj, VectorField):
        return tuple(obj.components.shape)
    if isinstance(obj, ScalarField):
        return tuple()
    raise TypeError(f"Unsupported tensor-like object: {type(obj)!r}")


def tensor_rank(obj: Any) -> int:
    return len(tensor_dimensions(obj))


def tensor_symmetry(obj: Any, tolerance: float | None = None) -> dict[str, tuple[tuple[int, ...], ...]]:
    if isinstance(obj, StructuredTensorArray):
        return dict(obj.symmetry_metadata)
    if isinstance(obj, TensorObject):
        if obj.symmetry_metadata:
            return dict(obj.symmetry_metadata)
        obj = obj.to_tensor_field()
    if isinstance(obj, VectorField) or isinstance(obj, ScalarField):
        return {}
    if isinstance(obj, TensorField):
        rank = len(obj.variance_spec)
        if rank != 2:
            return {}
        dim = obj.chart.dimension
        symmetric = True
        antisymmetric = True
        for i in range(dim):
            for j in range(dim):
                a = obj.components[(i, j)] - obj.components[(j, i)]
                b = obj.components[(i, j)] + obj.components[(j, i)]
                if not _approx_zero(a, tolerance):
                    symmetric = False
                if not _approx_zero(b, tolerance):
                    antisymmetric = False
                if not symmetric and not antisymmetric:
                    return {}
        out = {}
        if symmetric:
            out['symmetric'] = ((0, 1),)
        if antisymmetric:
            out['antisymmetric'] = ((0, 1),)
        return out
    raise TypeError(f"Unsupported tensor-like object: {type(obj)!r}")


def _tensorobject_result_for_like(original: Any, result: Any) -> Any:
    if isinstance(original, TensorObject):
        return result
    if isinstance(result, TensorObject):
        return result.to_tensor_field()
    return result


@dataclass(frozen=True)
class TensorReductionReport:
    input_kind: str
    used_contraction_graph: bool = False
    contraction_edges: tuple[tuple[int, int], ...] = tuple()
    disconnected_components: tuple[tuple[int, ...], ...] = tuple()
    scalar_factor: sp.Expr = sp.Integer(1)
    notes: tuple[str, ...] = tuple()
    stages: tuple[str, ...] = tuple()
    stage_counts: Mapping[str, int] = field(default_factory=dict)
    stage_durations_ms: Mapping[str, float] = field(default_factory=dict)
    contraction_plan_cost: int | None = None
    contraction_plan_order: tuple[str, ...] = tuple()


_last_tensor_reduction_report: TensorReductionReport | None = None


def last_tensor_reduction_report() -> TensorReductionReport | None:
    return _last_tensor_reduction_report


def sparse_tensor(shape: Sequence[int], entries: Mapping[tuple[int, ...], Any], *, symmetry_metadata: Mapping[str, tuple[tuple[int, ...], ...]] | None = None, domain_metadata: Mapping[str, Any] | None = None) -> StructuredTensorArray:
    return StructuredTensorArray(tuple(shape), {tuple(k): sp.sympify(v) for k, v in entries.items()}, dict(symmetry_metadata or {}), dict(domain_metadata or {}))


def tensor_to_structured(obj: Any, *, symmetry_metadata: Mapping[str, tuple[tuple[int, ...], ...]] | None = None, domain_metadata: Mapping[str, Any] | None = None) -> StructuredTensorArray:
    if isinstance(obj, StructuredTensorArray):
        return obj
    if isinstance(obj, TensorObject):
        md = dict(obj.symmetry_metadata)
        md.update(dict(symmetry_metadata or {}))
        dd = dict(obj.domain_metadata)
        dd.update(dict(domain_metadata or {}))
        return StructuredTensorArray.from_dense(obj.components, symmetry_metadata=md, domain_metadata=dd)
    t = _as_tensor(obj)
    return StructuredTensorArray.from_dense(t.components, symmetry_metadata=dict(symmetry_metadata or {}), domain_metadata=dict(domain_metadata or {}))


def tensor_from_structured(chart: Any, array: StructuredTensorArray, variance_spec: str, slot_bases: Sequence[Any] | None = None, *, name: str | None = None) -> TensorObject:
    return tensor_from_components(chart, array.to_dense(), variance_spec, slot_bases, name=name, symmetry_metadata=array.symmetry_metadata, domain_metadata=array.domain_metadata)




def tensor_metadata(obj: Any) -> dict[str, Any]:
    """Return a normalized metadata summary for tensor-like objects.

    This is intended as a stable interop layer across TensorObject,
    StructuredTensorArray, TensorField, VectorField, and ScalarField.
    """
    if isinstance(obj, StructuredTensorArray):
        return {
            'kind': 'StructuredTensorArray',
            'shape': tuple(obj.shape),
            'rank': len(obj.shape),
            'symmetry_metadata': dict(obj.symmetry_metadata),
            'domain_metadata': dict(obj.domain_metadata),
        }
    if isinstance(obj, TensorObject):
        return {
            'kind': 'TensorObject',
            'chart': obj.chart,
            'shape': tuple(obj.components.shape),
            'rank': len(obj.variance_spec),
            'variance_spec': obj.variance_spec,
            'slot_bases': tuple(obj.slot_bases),
            'name': obj.name,
            'symmetry_metadata': dict(obj.symmetry_metadata),
            'domain_metadata': dict(obj.domain_metadata),
        }
    if isinstance(obj, TensorField):
        return {
            'kind': 'TensorField',
            'chart': obj.chart,
            'shape': tuple(obj.components.shape),
            'rank': len(obj.variance_spec),
            'variance_spec': obj.variance_spec,
        }
    if isinstance(obj, VectorField):
        return {
            'kind': 'VectorField',
            'chart': obj.chart,
            'shape': tuple(obj.components.shape),
            'rank': 1,
            'variance_spec': 'u',
        }
    if isinstance(obj, ScalarField):
        return {
            'kind': 'ScalarField',
            'chart': obj.chart,
            'shape': tuple(),
            'rank': 0,
            'variance_spec': '',
        }
    arr = tensor_array(obj)
    shape = getattr(arr, 'shape', tuple())
    return {'kind': type(obj).__name__, 'shape': tuple(shape), 'rank': len(shape)}


def tensor_rebuild_like(obj: Any, components: Any, *, name: str | None = None,
                        symmetry_metadata: Mapping[str, tuple[tuple[int, ...], ...]] | None = None,
                        domain_metadata: Mapping[str, Any] | None = None) -> Any:
    """Rebuild an object of the same public kind with new components.

    Metadata is preserved by default and can be overridden explicitly.
    """
    if isinstance(obj, StructuredTensorArray):
        md = dict(obj.symmetry_metadata)
        md.update(dict(symmetry_metadata or {}))
        dd = dict(obj.domain_metadata)
        dd.update(dict(domain_metadata or {}))
        dense = as_tnf_array(components)
        return StructuredTensorArray.from_dense(dense, symmetry_metadata=md, domain_metadata=dd)
    if isinstance(obj, TensorObject):
        md = dict(obj.symmetry_metadata)
        md.update(dict(symmetry_metadata or {}))
        dd = dict(obj.domain_metadata)
        dd.update(dict(domain_metadata or {}))
        return tensor_from_components(obj.chart, components, obj.variance_spec, obj.slot_bases, name=name or obj.name, symmetry_metadata=md, domain_metadata=dd)
    if isinstance(obj, TensorField):
        return TensorField(obj.chart, as_tnf_array(components), obj.variance_spec)
    if isinstance(obj, VectorField):
        arr = as_tnf_array(components)
        if len(arr.shape) == 2 and arr.shape[1] == 1:
            return VectorField(obj.chart, tnf_array_to_sympy(arr))
        if len(arr.shape) == 1:
            return VectorField(obj.chart, sp.Matrix(list(arr.entries)))
        raise ValueError('VectorField rebuild requires rank-1 components.')
    if isinstance(obj, ScalarField):
        arr = as_tnf_array(components)
        if arr.shape not in (tuple(), ()):
            raise ValueError('ScalarField rebuild requires rank-0 components.')
        return ScalarField(obj.chart, arr[()])
    raise TypeError(f'Unsupported tensor-like object: {type(obj)!r}')


def tensor_roundtrip_structured(obj: Any, *, strict: bool = False) -> Any:
    """Round-trip a tensor-like object through StructuredTensorArray storage."""
    if isinstance(obj, StructuredTensorArray):
        return obj
    structured = tensor_to_structured(obj)
    if isinstance(obj, TensorObject):
        rebuilt = tensor_from_structured(obj.chart, structured, obj.variance_spec, obj.slot_bases, name=obj.name)
    else:
        rebuilt = tensor_rebuild_like(obj, structured.to_dense())
    if strict and not tensor_interop_report(obj).get('lossless_roundtrip', False):
        raise ValueError('Structured round-trip did not preserve all interoperability invariants.')
    return rebuilt


def _normalized_metadata_view(obj: Any) -> dict[str, Any]:
    meta = tensor_metadata(obj)
    return {
        'kind': meta.get('kind'),
        'rank': meta.get('rank'),
        'shape': tuple(meta.get('shape', tuple())),
        'variance_spec': meta.get('variance_spec', ''),
        'symmetry_metadata': dict(meta.get('symmetry_metadata', {})),
        'domain_metadata': dict(meta.get('domain_metadata', {})),
    }


def tensor_interop_report(obj: Any) -> dict[str, Any]:
    """Report interoperability invariants for a tensor-like object."""
    base = tensor_metadata(obj)
    structured = tensor_to_structured(obj)
    rebuilt = tensor_roundtrip_structured(obj)
    dense_equal = tensor_array(rebuilt) == tensor_array(obj)
    original_view = _normalized_metadata_view(obj)
    rebuilt_view = _normalized_metadata_view(rebuilt)
    invariants = {
        'kind': original_view.get('kind') == rebuilt_view.get('kind'),
        'rank': original_view.get('rank') == rebuilt_view.get('rank'),
        'shape': tuple(original_view.get('shape', tuple())) == tuple(rebuilt_view.get('shape', tuple())),
        'variance_spec': original_view.get('variance_spec', '') == rebuilt_view.get('variance_spec', ''),
    }
    out = dict(base)
    out.update({
        'structured_shape': tuple(structured.shape),
        'structured_nonzero_entries': len(structured.entries),
        'roundtrip_kind': type(rebuilt).__name__,
        'roundtrip_dense_equal': bool(dense_equal),
        'roundtrip_symmetry_metadata_equal': dict(structured.symmetry_metadata) == dict(getattr(rebuilt, 'symmetry_metadata', structured.symmetry_metadata)),
        'roundtrip_domain_metadata_equal': dict(structured.domain_metadata) == dict(getattr(rebuilt, 'domain_metadata', structured.domain_metadata)),
        'metadata_view_equal': original_view == rebuilt_view,
        'lossless_roundtrip': bool(dense_equal) and original_view == rebuilt_view,
        'original_metadata_view': original_view,
        'rebuilt_metadata_view': rebuilt_view,
        'invariants': invariants,
        'all_invariants_hold': all(invariants.values()) and bool(dense_equal),
        'preserved_metadata_keys': tuple(sorted(k for k in original_view if original_view.get(k) == rebuilt_view.get(k))),
    })
    return out

def tensor_array(obj: Any) -> Any:
    """Return the component-array representation of a tensor-like object.

    For TensorObject and TensorField inputs this returns the underlying TNFTensorArray.
    For StructuredTensorArray inputs it returns the object unchanged.
    Scalars are returned as rank-0 TNFTensorArray containers.
    """
    if isinstance(obj, StructuredTensorArray):
        return obj
    if isinstance(obj, TensorObject):
        return obj.components
    if isinstance(obj, TensorField):
        return obj.components
    if isinstance(obj, VectorField):
        return obj.as_tensor().components
    if isinstance(obj, ScalarField):
        return _rank0_tensor_array(obj.expr)
    if isinstance(obj, TNFTensorArray):
        return obj
    if hasattr(obj, 'shape') and hasattr(obj, '__getitem__'):
        try:
            return as_tnf_array(obj)
        except Exception:
            pass
    raise TypeError(f"Unsupported tensor-like object: {type(obj)!r}")


def tensor_element(obj: Any, indices: Sequence[int] | tuple[int, ...] = ()) -> Any:
    """Return a single component of a tensor-like object."""
    idx = tuple(indices)
    if isinstance(obj, StructuredTensorArray):
        return obj.entries.get(idx, sp.Integer(0))
    if isinstance(obj, TensorObject):
        return obj.components[idx]
    if isinstance(obj, TensorField):
        return obj.components[idx]
    if isinstance(obj, VectorField):
        return obj.as_tensor().components[idx]
    if isinstance(obj, ScalarField):
        if idx not in ((), tuple()):
            raise IndexError('ScalarField components use the empty index tuple ().')
        return obj.expr
    arr = tensor_array(obj)
    return arr[idx]


def _tensor_graph_payload(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], detail_nodes: list[dict[str, Any]] | None = None, detail_edges: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload = {
        'nodes': tuple(nodes),
        'edges': tuple(edges),
    }
    if detail_nodes is not None:
        payload['detail_nodes'] = tuple(detail_nodes)
    if detail_edges is not None:
        payload['detail_edges'] = tuple(detail_edges)
    return payload


def tensor_graph(obj: Any, *, as_networkx: bool = False) -> Any:
    """Build a lightweight planning/inspection graph for tensor products and contractions.

    The payload distinguishes factor nodes, slot nodes, and contraction edges. For indexed
    products it also records any available contraction-plan metadata.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    plan_payload: dict[str, Any] = {}
    try:
        from .tensor_indices import IndexedTensor, IndexedTensorExpr, build_contraction_graph, build_contraction_plan
    except Exception:
        IndexedTensor = IndexedTensorExpr = tuple()
        build_contraction_graph = build_contraction_plan = None

    if isinstance(obj, (IndexedTensor, IndexedTensorExpr)):
        factors = _flatten_tensor_product_expr(obj)
        factor_nodes: list[dict[str, Any]] = []
        contraction_edges: list[dict[str, Any]] = []
        detail_nodes: list[dict[str, Any]] = []
        detail_edges: list[dict[str, Any]] = []
        for i, factor in enumerate(factors):
            fnode = {'id': f'f{i}', 'kind': 'factor', 'label': getattr(getattr(factor, 'tensor', None), 'name', getattr(factor, 'name', f'factor_{i}')) or f'factor_{i}', 'index': i}
            factor_nodes.append(fnode)
            detail_nodes.append(dict(fnode))
            if hasattr(factor, 'indices'):
                for k, idx in enumerate(factor.indices):
                    slot_id = f'f{i}:s{k}'
                    detail_nodes.append({'id': slot_id, 'kind': 'slot', 'factor': i, 'slot': k, 'variance': getattr(idx, 'variance', '?'), 'name': getattr(idx, 'name', None), 'bundle': getattr(idx, 'bundle', None)})
                    detail_edges.append({'source': f'f{i}', 'target': slot_id, 'kind': 'incidence'})
        if build_contraction_graph is not None:
            graph = build_contraction_graph(factors)
            seen = set()
            for i, nbrs in graph.items():
                for j, w in nbrs.items():
                    if w and (j, i) not in seen:
                        seen.add((i, j))
                        edge = {'source': i, 'target': j, 'kind': 'contraction', 'weight': w, 'label': str(w)}
                        contraction_edges.append(edge)
                        detail_edges.append({'source': f'f{i}', 'target': f'f{j}', 'kind': 'contraction', 'weight': w, 'label': str(w)})
        if build_contraction_plan is not None:
            try:
                plan = build_contraction_plan(factors)
                plan_payload = {
                    'ordered_factor_count': len(getattr(plan, 'ordered_factors', tuple())),
                    'estimated_cost': getattr(plan, 'estimated_cost', None),
                    'priorities': tuple(getattr(plan, 'priorities', tuple())),
                    'order_labels': tuple(getattr(getattr(f, 'tensor', None), 'name', getattr(f, 'name', f'factor_{k}')) or f'factor_{k}' for k, f in enumerate(tuple(getattr(plan, 'ordered_factors', tuple())))),
                    'steps': tuple((k, k + 1) for k in range(max(len(tuple(getattr(plan, 'ordered_factors', tuple()))) - 1, 0))),
                }
            except Exception:
                plan_payload = {}
        payload = _tensor_graph_payload(factor_nodes, contraction_edges, detail_nodes, detail_edges)
        if plan_payload:
            payload['plan'] = plan_payload
        detail = payload.get('detail_nodes', payload['nodes'])
        payload['summary'] = {
            'factor_nodes': sum(1 for n in detail if n.get('kind') == 'factor'),
            'slot_nodes': sum(1 for n in detail if n.get('kind') == 'slot'),
            'contraction_edges': sum(1 for e in payload.get('detail_edges', payload['edges']) if e.get('kind') == 'contraction'),
            'estimated_cost': plan_payload.get('estimated_cost') if plan_payload else None,
        }
    elif isinstance(obj, TensorExpr) and obj.op == 'tensor_product':
        factors = _flatten_tensor_product_expr(obj)
        if any(isinstance(f, (IndexedTensor, IndexedTensorExpr)) for f in factors):
            symbolic_obj = _rebuild_tensor_product_expr(factors, symbolic=True)
            return tensor_graph(symbolic_obj, as_networkx=as_networkx)
        for i, factor in enumerate(factors):
            nodes.append({'id': f'f{i}', 'kind': 'factor', 'label': getattr(factor, 'name', f'factor_{i}'), 'index': i})
        payload = _tensor_graph_payload(nodes, edges)
        payload['summary'] = {'factor_nodes': len(nodes), 'slot_nodes': 0, 'contraction_edges': 0}
    elif isinstance(obj, TensorObject):
        rank = len(obj.variance_spec)
        slot_nodes = []
        detail_nodes = [{'id': 'tensor', 'kind': 'tensor', 'label': obj.name or 'tensor', 'rank': rank}]
        detail_edges = []
        for slot, var in enumerate(obj.variance_spec):
            sid = f's{slot}'
            node = {'id': sid, 'kind': 'slot', 'slot': slot, 'variance': var}
            slot_nodes.append(node)
            detail_nodes.append(dict(node))
            detail_edges.append({'source': 'tensor', 'target': sid, 'kind': 'incidence'})
        payload = _tensor_graph_payload(slot_nodes, tuple(), detail_nodes, detail_edges)
        payload['summary'] = {'factor_nodes': 1, 'slot_nodes': rank, 'contraction_edges': 0}
    else:
        raise TypeError(f'Unsupported tensor graph object: {type(obj)!r}')
    if as_networkx:
        try:
            import networkx as nx
        except Exception as exc:
            raise ImportError('networkx is required for as_networkx=True') from exc
        g = nx.Graph()
        for node in payload['nodes']:
            nd = dict(node)
            node_id = nd.pop('id')
            g.add_node(node_id, **nd)
        for edge in payload['edges']:
            ed = dict(edge)
            s = ed.pop('source')
            t = ed.pop('target')
            g.add_edge(s, t, **ed)
        if 'plan' in payload:
            g.graph['plan'] = payload['plan']
        if 'summary' in payload:
            g.graph['summary'] = payload['summary']
        return g
    return payload

def array_transform(obj: Any, matrices: Sequence[Any], slots: Sequence[int] | None = None) -> Any:
    if isinstance(obj, StructuredTensorArray):
        base = obj.to_dense()
        structured = True
    elif isinstance(obj, TensorObject):
        base = obj.components
        structured = False
    else:
        t = _as_tensor(obj)
        base = t.components
        structured = False
    base = as_tnf_array(base)
    rank = len(base.shape)
    if slots is None:
        slots = tuple(range(rank))
    if len(slots) != len(matrices):
        raise ValueError('slots and matrices must have the same length.')
    mats = [sp.Matrix(m) for m in matrices]
    def entry(new_idx):
        total = sp.Integer(0)
        for old_idx in product(*[range(s) for s in base.shape]):
            coeff = sp.Integer(1)
            for slot, mat in zip(slots, mats):
                coeff *= mat[new_idx[slot], old_idx[slot]]
            total += coeff * base[old_idx]
        return normal_simplify(total)
    out = tnf_build_array(base.shape, entry)
    if structured:
        return StructuredTensorArray.from_dense(out)
    if isinstance(obj, TensorObject):
        return tensor_from_components(obj.chart, out, obj.variance_spec, obj.slot_bases, name=obj.name, symmetry_metadata=obj.symmetry_metadata, domain_metadata=obj.domain_metadata)
    return TensorField(_as_tensor(obj).chart, out, _as_tensor(obj).variance_spec)


def array_hodge_star(obj: Any) -> Any:
    if isinstance(obj, TensorObject):
        return obj.hodge_star()
    return _as_tensor(obj).hodge_star()


def array_exterior_derivative(obj: Any) -> Any:
    if isinstance(obj, TensorObject):
        return obj.exterior_derivative()
    if isinstance(obj, ScalarField):
        return obj.exterior_derivative()
    return _as_tensor(obj).exterior_derivative()




def tensor_diagonal(obj: Any, slots: tuple[int, int] | None = None) -> Any:
    """Extract the diagonal over one slot pair without contracting it."""
    if isinstance(obj, TensorObject):
        base = obj
    else:
        base = TensorObject.from_tensor_field(_as_tensor(obj))
    rank = len(base.variance_spec)
    if rank < 2:
        raise ValueError('tensor_diagonal requires rank at least 2.')
    if slots is None:
        slots = (0, 1)
    i, j = slots
    if i == j or not (0 <= i < rank and 0 <= j < rank):
        raise ValueError('Invalid diagonal slot pair.')
    dim = base.chart.dimension
    keep = tuple(k for k in range(rank) if k not in slots)
    out_shape = (dim,) * (rank - 1)
    def entry(idx):
        idx = tuple(idx)
        total_idx = []
        src = iter(idx)
        diag_value = None
        for pos in range(rank):
            if pos == i or pos == j:
                if diag_value is None:
                    diag_value = next(src)
                total_idx.append(diag_value)
            else:
                total_idx.append(next(src))
        return base.components[tuple(total_idx)]
    out = tnf_build_array(out_shape, entry)
    variance = ''.join(base.variance_spec[k] for k in range(rank) if k != j)
    slot_bases = tuple(b for k, b in enumerate(base.slot_bases) if k != j)
    result = tensor_from_components(base.chart, out, variance, slot_bases, name=base.name, symmetry_metadata=base.symmetry_metadata, domain_metadata=base.domain_metadata)
    return _tensorobject_result_for_like(obj, result)


def tensor_flatten(obj: Any) -> Any:
    """Flatten tensor components to a simple column-array representation."""
    arr = tensor_array(obj)
    dense = as_tnf_array(arr)
    flat = as_tnf_array(sp.Matrix(list(dense.entries)))
    if isinstance(obj, StructuredTensorArray):
        return StructuredTensorArray.from_dense(flat, symmetry_metadata=obj.symmetry_metadata, domain_metadata=obj.domain_metadata)
    return flat


def tensor_reshape(obj: Any, shape: Sequence[int], *, variance_spec: str | None = None, slot_bases: Sequence[Any] | None = None) -> Any:
    """Reshape tensor-like component storage and rebuild a matching public object when possible."""
    arr = as_tnf_array(tensor_array(obj))
    shape = tuple(shape)
    if sp.prod(shape) != sp.prod(arr.shape if arr.shape else (1,)):
        raise ValueError('New shape must preserve the number of components.')
    dense = tnf_build_array(shape, lambda idx: arr.entries[sum(idx[k] * int(sp.prod(shape[k+1:]) or 1) for k in range(len(shape)))] )
    if isinstance(obj, StructuredTensorArray):
        return StructuredTensorArray.from_dense(dense, symmetry_metadata=obj.symmetry_metadata, domain_metadata=obj.domain_metadata)
    if isinstance(obj, TensorObject):
        vs = variance_spec if variance_spec is not None else ('u' * len(shape))
        sb = tuple(slot_bases) if slot_bases is not None else tuple(obj.slot_bases[:len(shape)])
        if shape == (obj.chart.dimension,) * len(shape) and len(vs) == len(shape) and len(sb) == len(shape):
            return tensor_from_components(obj.chart, dense, vs, sb, name=obj.name, symmetry_metadata=obj.symmetry_metadata, domain_metadata=obj.domain_metadata)
    return dense


def tensor_map(obj: Any, func: Any) -> Any:
    """Apply a scalar-valued function elementwise while preserving tensor metadata when possible."""
    if isinstance(obj, StructuredTensorArray):
        dense = obj.to_dense().applyfunc(lambda e: sp.sympify(func(e)))
        return StructuredTensorArray.from_dense(dense, symmetry_metadata=obj.symmetry_metadata, domain_metadata=obj.domain_metadata)
    if isinstance(obj, TensorObject):
        dense = obj.components.applyfunc(lambda e: sp.sympify(func(e)))
        return tensor_from_components(obj.chart, dense, obj.variance_spec, obj.slot_bases, name=obj.name, symmetry_metadata=obj.symmetry_metadata, domain_metadata=obj.domain_metadata)
    if isinstance(obj, TensorField):
        return TensorField(obj.chart, obj.components.applyfunc(lambda e: sp.sympify(func(e))), obj.variance_spec)
    if isinstance(obj, VectorField):
        return VectorField(obj.chart, sp.Matrix([sp.sympify(func(e)) for e in list(obj.components)]), variance=obj.variance)
    if isinstance(obj, ScalarField):
        return ScalarField(obj.chart, sp.sympify(func(obj.expr)))
    return as_tnf_array(tensor_array(obj)).applyfunc(lambda e: sp.sympify(func(e)))


def zero_tensor_like(obj: Any) -> Any:
    """Create a zero object with the same public kind and metadata as the input."""
    if isinstance(obj, ScalarField):
        return ScalarField(obj.chart, sp.Integer(0))
    arr = tensor_array(obj)
    zero = tnf_build_array(getattr(arr, 'shape', tuple()), lambda idx: sp.Integer(0))
    return tensor_rebuild_like(obj, zero)


def _is_zero_tensorlike(obj: Any) -> bool:
    if isinstance(obj, ScalarField):
        return is_zero(obj.expr)
    try:
        arr = as_tnf_array(tensor_array(obj))
    except Exception:
        return False
    return all(is_zero(entry) for entry in arr.entries)


def _propagate_product_symmetry_metadata(factors: Sequence[Any]) -> dict[str, tuple[tuple[int, ...], ...]]:
    out: dict[str, list[tuple[int, ...]]] = {}
    offset = 0
    for factor in factors:
        if tensor_q(factor):
            rep = tensor_symmetry(factor)
            for key, groups in rep.items():
                for group in groups:
                    out.setdefault(key, []).append(tuple(offset + int(g) for g in group))
            offset += tensor_rank(factor)
    return {k: tuple(v) for k, v in out.items() if v}


def tensor_covariant_derivative(obj: Any) -> Any:
    from .calculus import covariant_derivative
    return covariant_derivative(obj)


def tensor_hessian(obj: Any) -> Any:
    from .calculus import hessian
    return hessian(obj)


def tensor_lie_derivative(obj: Any, vector: VectorField) -> Any:
    from .calculus import lie_derivative
    return lie_derivative(obj, vector)


def tensor_exterior_derivative(obj: Any) -> Any:
    from .calculus import exterior_derivative
    return exterior_derivative(obj)

def _authoritative_reduce_indexed(obj: Any, *, preprocess_contractions: bool = True, split_components: bool = True, sort_products: bool = True, notes: list[str] | None = None) -> tuple[Any, tuple[tuple[int, int], ...], tuple[tuple[int, ...], ...], bool, int | None]:
    try:
        from .tensor_indices import IndexedTensor, IndexedTensorExpr, build_contraction_graph, build_contraction_plan, normalize_indexed_expression
    except Exception:
        return obj, tuple(), tuple(), False, None
    if not isinstance(obj, (IndexedTensor, IndexedTensorExpr)):
        return obj, tuple(), tuple(), False, None
    work = obj
    disconnected_components = tuple()
    used_graph = False
    edges = tuple()
    contraction_plan_cost = None
    contraction_plan_order: tuple[str, ...] = tuple()
    if preprocess_contractions and split_components:
        work, disconnected_components, extra_notes = _split_indexed_connected_components(work)
        if notes is not None:
            notes.extend(extra_notes)
    if sort_products:
        work = tensor_sort(work)
    try:
        graph = build_contraction_graph(_flatten_tensor_product_expr(work))
        seen = set()
        edge_list = []
        for i, nbrs in graph.items():
            for j, w in nbrs.items():
                if w and (j, i) not in seen:
                    seen.add((i, j))
                    edge_list.append((i, j))
        edges = tuple(sorted(edge_list))
        used_graph = bool(edges)
    except Exception:
        pass
    try:
        plan = build_contraction_plan(_flatten_tensor_product_expr(work))
        contraction_plan_cost = getattr(plan, 'estimated_cost', None)
    except Exception:
        pass
    try:
        work = normalize_indexed_expression(work)
    except Exception:
        pass
    return work, edges, disconnected_components, used_graph, contraction_plan_cost


def _symmetry_zero_reduce(obj: Any) -> tuple[Any, tuple[str, ...]]:
    notes: list[str] = []
    if isinstance(obj, (TensorObject, TensorField)):
        rep = tensor_symmetry(obj)
        sym = {tuple(g) for g in rep.get('symmetric', tuple())}
        anti = {tuple(g) for g in rep.get('antisymmetric', tuple())}
        conflict = tuple(sorted(sym & anti))
        if conflict:
            notes.append('reduced conflicting symmetric/antisymmetric tensor to zero')
            return zero_tensor_like(obj), tuple(notes)
        if anti:
            arr = as_tnf_array(tensor_array(obj))
            changed = False
            def entry(idx):
                nonlocal changed
                for group in anti:
                    if len({idx[g] for g in group}) < len(group):
                        changed = True
                        return sp.Integer(0)
                return arr[idx]
            reduced = tnf_build_array(arr.shape, entry)
            if changed:
                notes.append('zeroed repeated antisymmetric diagonal components')
                return tensor_rebuild_like(obj, reduced), tuple(notes)
    return obj, tuple(notes)

def tensor_reduce(obj: Any, assumptions: sp.Expr | None = None, symmetry: bool = True, basis: bool = True,
                  preprocess_contractions: bool = True, tolerance: float | None = None,
                  stages: Sequence[str] | None = None, sort_products: bool = True,
                  split_components: bool = True, reduce_transposes: bool = True,
                  use_symmetry: bool = True, full_tensorform: bool = True) -> Any:
    """Reduce a tensor-like object through a configurable staged pipeline."""
    global _last_tensor_reduction_report
    if stages is None:
        stages = (
            'separate_scalars',
            'sort_products',
            'split_components',
            'reduce_transposes',
            'symmetry',
            'tensorform',
            'simplify',
        )
    stages = tuple(stages)
    stage_counts: dict[str, int] = {}
    stage_durations_ms: dict[str, float] = {}
    used_graph = False
    edges: tuple[tuple[int, int], ...] = tuple()
    disconnected_components: tuple[tuple[int, ...], ...] = tuple()
    notes: list[str] = []
    scalar_factor = sp.Integer(1)
    contraction_plan_cost = None
    try:
        from .tensor_indices import (
            build_contraction_graph,
            build_contraction_plan,
            canonical_indexed_form,
            indexed_canonical_report,
            IndexedTensor,
            IndexedTensorExpr,
        )
    except Exception:
        build_contraction_graph = build_contraction_plan = None
        canonical_indexed_form = indexed_canonical_report = None
        IndexedTensor = IndexedTensorExpr = tuple()

    def _record(name: str, started: float, *, count: int = 1):
        stage_counts[name] = stage_counts.get(name, 0) + count
        stage_durations_ms[name] = stage_durations_ms.get(name, 0.0) + (time.perf_counter() - started) * 1000.0

    if isinstance(obj, (IndexedTensor, IndexedTensorExpr)):
        work = obj
        t0 = time.perf_counter()
        work, edges, disconnected_components, used_graph, contraction_plan_cost = _authoritative_reduce_indexed(
            work,
            preprocess_contractions=preprocess_contractions,
            split_components=split_components and 'split_components' in stages,
            sort_products=sort_products and 'sort_products' in stages,
            notes=notes,
        )
        _record('authoritative_indexed_reduce', t0)
        if sort_products and 'sort_products' in stages:
            stage_counts['sort_products'] = stage_counts.get('sort_products', 0) + 1
            stage_durations_ms.setdefault('sort_products', 0.0)
        if disconnected_components:
            stage_counts['split_components'] = stage_counts.get('split_components', 0) + len(disconnected_components)
        if edges:
            stage_counts['contraction_graph'] = stage_counts.get('contraction_graph', 0) + 1
        if contraction_plan_cost is not None:
            stage_counts['contraction_plan'] = stage_counts.get('contraction_plan', 0) + 1
        if full_tensorform and canonical_indexed_form is not None and 'tensorform' in stages:
            t0 = time.perf_counter()
            work = canonical_indexed_form(work)
            _record('tensorform', t0)
        if build_contraction_plan is not None:
            try:
                _plan = build_contraction_plan(_flatten_tensor_product_expr(work))
                contraction_plan_order = tuple(getattr(getattr(f, 'tensor', None), 'name', getattr(f, 'name', '<anon>')) or '<anon>' for f in getattr(_plan, 'ordered_factors', tuple()))
            except Exception:
                contraction_plan_order = tuple()
        if indexed_canonical_report is not None and 'tensorform' in stages:
            t0 = time.perf_counter()
            try:
                report = indexed_canonical_report(work)
                notes.append(f'tensorform terms={len(getattr(report, "term_signatures", ())) or len(getattr(report, "normal_form_terms", ())) }')
            except Exception:
                pass
            _record('tensorform_report', t0)
        _last_tensor_reduction_report = TensorReductionReport(type(obj).__name__, used_graph, edges, disconnected_components, scalar_factor, tuple(notes), stages, dict(stage_counts), dict(stage_durations_ms), contraction_plan_cost)
        return work

    work = obj
    if 'separate_scalars' in stages:
        t0 = time.perf_counter()
        work, scalar_factor, scalar_notes = _separate_scalar_factors(work)
        notes.extend(scalar_notes)
        _record('separate_scalars', t0, count=1 if scalar_factor != 1 else 0)
    if sort_products and 'sort_products' in stages:
        t0 = time.perf_counter()
        work = tensor_sort(work)
        _record('sort_products', t0)

    if isinstance(work, TensorObject):
        out = work
        if use_symmetry and symmetry and 'symmetry' in stages:
            out, zero_notes = _symmetry_zero_reduce(out)
            notes.extend(zero_notes)
        if reduce_transposes and 'reduce_transposes' in stages:
            t0 = time.perf_counter()
            out = tensor_simplify(out)
            _record('reduce_transposes', t0)
        if use_symmetry and symmetry and 'symmetry' in stages:
            t0 = time.perf_counter()
            out = tensor_project_symmetry(out, out.symmetry_metadata) if out.symmetry_metadata else out.canonicalize_symmetry()
            _record('symmetry', t0, count=len(out.symmetry_metadata))
        if 'simplify' in stages:
            t0 = time.perf_counter()
            out = expr_tensor_reduce(out)
            if assumptions is not None:
                out = out.simplify(assumptions=assumptions, canonicalize_symmetry=symmetry)
            elif symmetry:
                out = out.canonicalize_symmetry()
            _record('simplify', t0)
        if scalar_factor != 1:
            out = tensor_from_components(out.chart, out.components.applyfunc(lambda e: normal_simplify(scalar_factor * e)), out.variance_spec, out.slot_bases, name=out.name, symmetry_metadata=out.symmetry_metadata, domain_metadata=out.domain_metadata)
        _last_tensor_reduction_report = TensorReductionReport('TensorObject', used_graph, edges, disconnected_components, scalar_factor, tuple(notes), stages, dict(stage_counts), dict(stage_durations_ms), contraction_plan_cost)
        return out.to_tensor_field()

    if isinstance(work, TensorExpr):
        t0 = time.perf_counter()
        evaluated = tensor_simplify(work)
        _record('simplify', t0)
        if scalar_factor != 1:
            evaluated = tensor_from_components(evaluated.chart, evaluated.components.applyfunc(lambda e: normal_simplify(scalar_factor * e)), evaluated.variance_spec, evaluated.slot_bases, name=evaluated.name, symmetry_metadata=evaluated.symmetry_metadata, domain_metadata=evaluated.domain_metadata)
        _last_tensor_reduction_report = TensorReductionReport(type(obj).__name__, used_graph, edges, disconnected_components, scalar_factor, tuple(notes), stages, dict(stage_counts), dict(stage_durations_ms), contraction_plan_cost)
        return evaluated.to_tensor_field()

    t0 = time.perf_counter()
    t = _as_tensor(work)
    out = t.components.applyfunc(lambda e: sp.refine(normal_simplify(e), assumptions) if assumptions is not None else normal_simplify(e))
    if scalar_factor != 1:
        out = out.applyfunc(lambda e: normal_simplify(scalar_factor * e))
    tf = TensorField(t.chart, out, t.variance_spec)
    _record('simplify', t0)
    _last_tensor_reduction_report = TensorReductionReport(type(obj).__name__, used_graph, edges, disconnected_components, scalar_factor, tuple(notes), stages, dict(stage_counts), dict(stage_durations_ms), contraction_plan_cost)
    return tf

def tensor_reduce_native(obj: Any, **kwargs) -> Any:
    return tensor_reduce(obj, **kwargs)


def tensor_contract(obj: Any, pairs: Sequence[tuple[int, int]]) -> Any:
    """Contract a tensor-like object over the given slot pairs."""
    if not pairs:
        return obj
    if isinstance(obj, TensorObject):
        return obj.multi_contract(tuple(tuple(p) for p in pairs))
    tensor = _as_tensor(obj)
    wrapped = TensorObject.from_tensor_field(tensor)
    result = wrapped.multi_contract(tuple(tuple(p) for p in pairs))
    return _tensorobject_result_for_like(obj, result)


def tensor_trace(obj: Any, slots: tuple[int, int] | None = None) -> Any:
    """Trace a tensor-like object over one slot pair."""
    tensor = obj if isinstance(obj, TensorObject) else _as_tensor(obj)
    variance = tensor.variance_spec if isinstance(tensor, TensorObject) else tensor.variance_spec
    if slots is None:
        found = None
        for i in range(len(variance)):
            for j in range(i + 1, len(variance)):
                if variance[i] != variance[j]:
                    found = (i, j)
                    break
            if found is not None:
                break
        if found is None:
            raise ValueError('No contractible slot pair found for tensor_trace.')
        slots = found
    return tensor_contract(obj, [slots])


def kronecker_delta_tensor(chart: Any) -> TensorField:
    """Return the (1,1) identity/Kronecker delta tensor δ^i_j."""
    return identity_tensor(chart, "ul")


def metric_tensor(chart: Any, variance_spec: str = "ll") -> TensorField:
    """Return the metric tensor g_ij or inverse metric g^ij as a TensorField."""
    coords = chart.symbols()
    if variance_spec == "ll":
        mat = chart.metric(coords)
    elif variance_spec == "uu":
        mat = chart.inverse_metric(coords)
    else:
        raise NotImplementedError("metric_tensor supports variance_spec='ll' or 'uu' only.")
    if mat is None:
        raise ValueError("Chart does not define a metric.")
    arr = tnf_build_array((chart.dimension, chart.dimension), lambda idx: mat[idx])
    return TensorField(chart, arr, variance_spec)



def _perm_sign(seq):
    inv = 0
    for i in range(len(seq)):
        for j in range(i+1, len(seq)):
            if seq[i] == seq[j]:
                return sp.Integer(0)
            if seq[i] > seq[j]:
                inv += 1
    return sp.Integer(-1 if inv % 2 else 1)


def levi_civita_symbol(chart: Any, variance_spec: str = "lll") -> TensorField:
    """Return the Levi-Civita permutation tensor as a plain symbol tensor.

    This is the alternating symbol ε with components in {−1,0,1}. For a metric-aware
    volume form, use ``volume_form``.
    """
    n = chart.dimension
    if len(variance_spec) != n:
        raise ValueError("Levi-Civita symbol rank must equal chart dimension.")
    arr = tnf_build_array((n,) * n, _perm_sign)
    return TensorField(chart, arr, variance_spec)


def permutation_tensor(chart: Any, variance_spec: str = "lll") -> TensorField:
    return levi_civita_symbol(chart, variance_spec)


def volume_form(chart: Any, variance_spec: str = "lll") -> TensorField:
    """Return the metric volume tensor/form in the chosen variance pattern.

    For all-lower indices this is sqrt(det(g)) * ε_{i1...in}. Other variance patterns are
    obtained by raising indices with the metric.
    """
    n = chart.dimension
    if len(variance_spec) != n:
        raise ValueError("volume_form rank must equal chart dimension.")
    coords = chart.symbols()
    base = levi_civita_symbol(chart, 'l' * n)
    sqrtg = chart.sqrt_metric_det(coords)
    arr = tnf_build_array((n,) * n, lambda idx: canonical_simplify(sqrtg * base.components[idx], final=False))
    t = TensorField(chart, arr, 'l' * n)
    for slot, kind in enumerate(variance_spec):
        if kind == 'u':
            t = t.raise_index(slot)
    return t



def change_tensor_basis(obj: Any, new_bases: Sequence[Any] | Any, *, slots: Sequence[int] | None = None, coords: Sequence[Any] | None = None) -> Any:
    """Change one or more tensor slot bases.

    When new_bases is a single basis object, it is applied to every selected slot.
    When it is a sequence, it must either have full rank length or match the selected slots.
    """
    base = obj if isinstance(obj, TensorObject) else TensorObject.from_tensor_field(_as_tensor(obj))
    rank = len(base.variance_spec)
    if slots is None:
        target_slots = tuple(range(rank))
    else:
        target_slots = tuple(int(s) for s in slots)
    if any(s < 0 or s >= rank for s in target_slots):
        raise ValueError('Invalid slot index in change_tensor_basis.')
    updated = list(base.slot_bases)
    if isinstance(new_bases, (list, tuple)):
        seq = tuple(new_bases)
        if len(seq) == rank and slots is None:
            updated = list(seq)
        elif len(seq) == len(target_slots):
            for slot, new_basis in zip(target_slots, seq):
                updated[slot] = new_basis
        else:
            raise ValueError('new_bases must match tensor rank or selected slot count.')
    else:
        for slot in target_slots:
            updated[slot] = new_bases
    result = base.change_basis(tuple(updated), coords=None if coords is None else tuple(coords))
    return _tensorobject_result_for_like(obj, result)



def raise_tensor_slots(obj: Any, slots: Sequence[int]) -> Any:
    base = obj if isinstance(obj, TensorObject) else TensorObject.from_tensor_field(_as_tensor(obj))
    result = base.raise_slots(tuple(slots))
    return _tensorobject_result_for_like(obj, result)



def lower_tensor_slots(obj: Any, slots: Sequence[int]) -> Any:
    base = obj if isinstance(obj, TensorObject) else TensorObject.from_tensor_field(_as_tensor(obj))
    result = base.lower_slots(tuple(slots))
    return _tensorobject_result_for_like(obj, result)



def change_tensor_slots(obj: Any, *, perm: Sequence[int] | None = None, raise_slots_spec: Sequence[int] | None = None, lower_slots_spec: Sequence[int] | None = None, contract_pairs: Sequence[tuple[int, int]] | None = None) -> Any:
    """Apply a bundle of slot changes in a deterministic order.

    The order is permutation, then raising, then lowering, then contraction.
    """
    current = obj
    if perm is not None:
        current = tensor_permute(current, perm)
    if raise_slots_spec:
        current = raise_tensor_slots(current, tuple(raise_slots_spec))
    if lower_slots_spec:
        current = lower_tensor_slots(current, tuple(lower_slots_spec))
    if contract_pairs:
        current = tensor_contract(current, tuple(tuple(p) for p in contract_pairs))
    return current



def normalize_contraction_graph(obj: Any) -> dict[str, Any]:
    """Return a deterministically ordered tensor-contraction graph payload."""
    payload = tensor_graph(obj)
    nodes = sorted((dict(n) for n in payload.get('nodes', ())), key=lambda n: (str(n.get('kind')), str(n.get('id'))))
    edges = sorted((dict(e) for e in payload.get('edges', ())), key=lambda e: (str(e.get('kind')), str(e.get('source')), str(e.get('target')), str(e.get('weight', ''))))
    out = {'nodes': tuple(nodes), 'edges': tuple(edges)}
    if 'detail_nodes' in payload:
        out['detail_nodes'] = tuple(sorted((dict(n) for n in payload['detail_nodes']), key=lambda n: (str(n.get('kind')), str(n.get('id')))))
    if 'detail_edges' in payload:
        out['detail_edges'] = tuple(sorted((dict(e) for e in payload['detail_edges']), key=lambda e: (str(e.get('kind')), str(e.get('source')), str(e.get('target')), str(e.get('weight', '')))))
    factor_ids = [n['id'] for n in out.get('detail_nodes', out['nodes']) if n.get('kind') in {'factor', 'tensor'}]
    contraction_edges = [e for e in out.get('detail_edges', out['edges']) if e.get('kind') == 'contraction']
    out['summary'] = {
        'factor_nodes': len(factor_ids),
        'contraction_edges': len(contraction_edges),
        'key': tuple((e.get('source'), e.get('target'), e.get('weight', 0)) for e in contraction_edges),
    }
    return out



def contraction_graph_key(obj: Any) -> tuple:
    payload = normalize_contraction_graph(obj)
    return (
        tuple((n.get('id'), n.get('kind')) for n in payload.get('nodes', ())),
        tuple((e.get('source'), e.get('target'), e.get('kind'), e.get('weight', 0)) for e in payload.get('edges', ())),
    )



def tensor_product_dispatch(*objs: Any, pipeline: Sequence[str] | None = None, mode: str | None = None, **kwargs) -> Any:
    cleaned = [obj for obj in objs if obj is not None]
    if not cleaned:
        raise ValueError('tensor_product_dispatch requires at least one input.')
    result = cleaned[0] if len(cleaned) == 1 else tensor_product(*cleaned)
    return unified_tensor_rewrite_pipeline(result, operator="tensor_product", families=pipeline, mode=mode, **kwargs) if (pipeline is not None or kwargs.get("apply_pipeline")) else result


def tensor_wedge_dispatch(*objs: Any, pipeline: Sequence[str] | None = None, mode: str | None = None, **kwargs) -> Any:
    cleaned = [obj for obj in objs if obj is not None]
    if not cleaned:
        raise ValueError('tensor_wedge_dispatch requires at least one input.')
    if len(cleaned) == 1:
        result = cleaned[0]
    else:
        acc = cleaned[0] if isinstance(cleaned[0], TensorObject) else TensorObject.from_tensor_field(_as_tensor(cleaned[0]))
        for obj in cleaned[1:]:
            rhs = obj if isinstance(obj, TensorObject) else TensorObject.from_tensor_field(_as_tensor(obj))
            acc = acc.wedge(rhs)
        result = acc
    return unified_tensor_rewrite_pipeline(result, operator="tensor_wedge", families=pipeline, mode=mode, **kwargs) if (pipeline is not None or kwargs.get("apply_pipeline")) else result


def tensor_transpose_dispatch(obj: Any, perm: Sequence[int] | None = None, pipeline: Sequence[str] | None = None, mode: str | None = None, **kwargs) -> Any:
    if perm is None:
        base = obj if isinstance(obj, TensorObject) else TensorObject.from_tensor_field(_as_tensor(obj))
        if len(base.variance_spec) != 2:
            raise ValueError('tensor_transpose_dispatch needs perm for non-rank-2 inputs.')
        perm = (1, 0)
    result = tensor_permute(obj, perm)
    return unified_tensor_rewrite_pipeline(result, operator="tensor_transpose", families=pipeline, mode=mode, **kwargs) if (pipeline is not None or kwargs.get("apply_pipeline")) else result


def tensor_contract_dispatch(obj: Any, *pairs: tuple[int, int], pipeline: Sequence[str] | None = None, mode: str | None = None, **kwargs) -> Any:
    result = obj if not pairs else tensor_contract(obj, tuple(tuple(p) for p in pairs))
    return unified_tensor_rewrite_pipeline(result, operator="tensor_contract", families=pipeline, mode=mode, **kwargs) if (pipeline is not None or kwargs.get("apply_pipeline")) else result


@dataclass(frozen=True)
class IndexedReductionStep:
    name: str
    before: Any
    after: Any
    changed: bool


@dataclass(frozen=True)
class IndexedReductionReport:
    original: Any
    final: Any
    requested_stages: tuple[str, ...]
    executed_steps: tuple[IndexedReductionStep, ...]


def canonical_reduce_contraction_graph(obj: Any) -> Any:
    try:
        from .tensor_indices import IndexedTensor, IndexedTensorExpr, alpha_rename_dummies
    except Exception:
        IndexedTensor = IndexedTensorExpr = tuple()
        alpha_rename_dummies = None
    if not isinstance(obj, (IndexedTensor, IndexedTensorExpr)):
        return tensor_reduce(obj, stages=("sort_products", "reduce_transposes", "symmetry", "simplify"))
    from .indexed_api import canonicalize_indexed_expression
    work = canonicalize_indexed_expression(obj)
    expr = getattr(work, 'expr', work)
    factors, factor_nodes, index_nodes = _build_indexed_hypergraph(expr)
    if len(factors) <= 1:
        return canonicalize_indexed_expression(work)
    factor_labels, index_labels = _wl_refine_indexed_hypergraph(factor_nodes, index_nodes)
    rename = {name: f'd{k}' for k, name in enumerate(sorted((n for n, d in index_nodes.items() if d['partition'] == 'dummy'), key=lambda n: (index_labels[n], n)))}
    ordered = []
    for fi, factor in enumerate(factors):
        renamed = _rename_indexed_factor_names(factor, rename)
        ordered.append((factor_labels[fi], str(renamed), fi, renamed))
    rebuilt = IndexedTensorExpr('tensor_product', tuple(item[-1] for item in sorted(ordered, key=lambda item: (item[0], item[1], item[2]))))
    if alpha_rename_dummies is not None:
        rebuilt = alpha_rename_dummies(rebuilt)
    return canonicalize_indexed_expression(rebuilt)


def tensor_reduce_indexed_staged(obj: Any, *, stages: Sequence[str] | None = None, config: Any | None = None, with_report: bool = False) -> Any:
    from .indexed_api import normalize_indexed_expression, canonicalize_indexed_expression
    if stages is None:
        stages = ("structural", "graph", "tensorform", "simplify")
    ordered = []
    for name in ("structural", "graph", "tensorform", "simplify"):
        if name in tuple(stages):
            ordered.append(name)
    current = obj
    steps = []
    for name in ordered:
        before = current
        if name == "structural":
            current = normalize_indexed_expression(current, config=config)
        elif name == "graph":
            current = canonical_reduce_contraction_graph(current)
        elif name == "tensorform":
            current = canonicalize_indexed_expression(current, config=config)
        elif name == "simplify":
            current = canonicalize_indexed_expression(current, config=config)
        steps.append(IndexedReductionStep(name, before, current, current != before))
    if with_report:
        return current, IndexedReductionReport(obj, current, tuple(ordered), tuple(steps))
    return current


def unified_tensor_rewrite_pipeline(expr: Any, *, operator: str | None = None, layer: str | None = None, families: Sequence[str] | None = None, mode: str | None = None, **kwargs) -> Any:
    from .rewrite_families import apply_rewrite_families
    current = expr
    if layer is None:
        try:
            from .tensor_indices import IndexedTensor, IndexedTensorExpr
        except Exception:
            IndexedTensor = IndexedTensorExpr = tuple()
        layer = "indexed" if isinstance(current, (IndexedTensor, IndexedTensorExpr)) else "abstract" if hasattr(current, 'expr') else "indexed"
    if families is None:
        families = ("all",) if layer == "abstract" else ("canonicalize",)
    return apply_rewrite_families(current, families=families, layer=layer, mode=mode, **kwargs)


def tensor_reduce_component_staged(obj: Any, *, stages: Sequence[str] | None = None, assumptions: sp.Expr | None = None, with_report: bool = False) -> Any:
    if stages is None:
        stages = ("structural", "graph", "tensorform", "simplify")
    ordered = []
    for name in ("structural", "graph", "tensorform", "simplify"):
        if name in tuple(stages):
            ordered.append(name)
    current = obj
    steps = []
    for name in ordered:
        before = current
        if name == "structural":
            current = tensor_reduce(current, assumptions=assumptions, stages=("separate_scalars", "sort_products", "split_components"))
        elif name == "graph":
            current = canonical_reduce_contraction_graph(current)
        elif name == "tensorform":
            current = tensor_reduce(current, assumptions=assumptions, stages=("reduce_transposes", "symmetry", "tensorform"))
        elif name == "simplify":
            current = tensor_reduce(current, assumptions=assumptions, stages=("simplify",))
        steps.append(IndexedReductionStep(name, before, current, current != before))
    if with_report:
        return current, IndexedReductionReport(obj, current, tuple(ordered), tuple(steps))
    return current

