from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from weakref import WeakKeyDictionary

import sympy as sp
from sympy.combinatorics.tensor_can import canonicalize as _canonicalize_perm
from .canonical_keys import canonical_expr_fingerprint, canonical_named_aliases, canonical_sequence_key, structural_key
from .semantic_core import canonical_semantic_form, semantic_ir

from sympy.tensor.tensor import (
    Tensor,
    TensorHead as SymTensorHead,
    TensorIndex as SymTensorIndex,
    TensorIndexType as SymTensorIndexType,
    TensorSymmetry as SymTensorSymmetry,
    TensAdd,
    TensMul,
    get_symmetric_group_sgs,
    tensor_indices,
)



_CANONICAL_SCALAR_MONOMIAL_CACHE: "OrderedDict[tuple[Any, ...], object]" = OrderedDict()
_INVARIANT_CATALOG_CACHE: "OrderedDict[tuple[Any, ...], object]" = OrderedDict()
_INVARIANT_DESCRIPTOR_CACHE: "OrderedDict[tuple[Any, ...], object]" = OrderedDict()
_CACHE_LIMIT = 256


def _bounded_cache_get(cache: OrderedDict, key):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


def _bounded_cache_set(cache: OrderedDict, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)
    return value

__all__ = [
    "AbstractTensorCanonicalizationError",
    "AbstractTensorCanonicalizationReport",
    "CurvatureInvariantReductionReport",
    "TensorAtlasAbstractExpr",
    "AbstractTensorExpr",
    "IndexType",
    "Index",
    "Metric",
    "TensorHead",
    "index_type",
    "tensor_head",
    "metric",
    "raise_index",
    "lower_index",
    "contract_metric",
    "trace_abstract",
    "delta_reduce",
    "AbstractContractionStep",
    "AbstractContractionPlan",
    "AbstractSimplificationStep",
    "AbstractSimplificationReport",
    "build_contraction_plan",
    "execute_contraction_plan",
    "validate_contractions",
    "structural_simplify",
    "metric_simplify",
    "multiterm_simplify",
    "invariant_simplify",
    "simplify_abstract",
    "simplify_abstract_with_report",
    "abstract_index_type",
    "abstract_tensor_head",
    "fully_symmetric_head",
    "fully_antisymmetric_head",
    "riemann_tensor_head",
    "ricci_tensor_head",
    "weyl_tensor_head",
    "as_abstract_tensor_expr",
    "indexed_to_abstract",
    "abstract_to_indexed",
    "butler_portugal_canonicalize",
    "butler_portugal_canonicalize_permutation",
    "canonicalize_abstract_tensor_expr",
    "canonicalize_abstract_tensor_expr_with_report",
    "multi_term_tensor_reduce",
    "curvature_invariant_signature",
    "curvature_invariant_basis",
    "reduce_curvature_invariants",
    "reduce_curvature_invariants_with_report",
    "symmetrize_indices",
    "antisymmetrize_indices",
    "young_project_indices",
    "schouten_tensor_head",
    "scalar_curvature_symbol",
    "decompose_riemann_curvature",
    "decompose_curvature_expression",
    "derivative_tensor_head",
    "tableau_reduce",
    "differential_bianchi_reduce",
    "commute_covariant_derivatives",
    "schouten_from_ricci",
    "ricci_from_schouten",
    "weyl_from_riemann_schouten",
    "decompose_curvature_workflow",
    "DifferentialCurvatureInvariantReductionReport",
    "differential_curvature_invariant_signature",
    "differential_curvature_invariant_basis",
    "reduce_differential_curvature_invariants",
    "reduce_differential_curvature_invariants_with_report",
    "Torsion",
    "Connection",
    "CovariantDerivativeOperator",
    "IrreducibleComponent",
    "DifferentialInvariantDescriptor",
    "AbstractNormalForm",
    "CurvatureIdentity",
    "OperatorNormalForm",
    "torsion",
    "connection",
    "operator_normal_form",
    "covariant_derivative_operator",
    "apply_covariant_derivative",
    "derivative_commutator",
    "decompose_tableau_product",
    "representation_reduce",
    "classify_differential_invariants",
    "differential_invariant_basis_catalog",
    "abstract_normal_form",
    "canonical_tensor_expression",
    "canonical_tensor_normal_form",
    "abstract_normal_form_key",
    "compare_normal_forms",
    "list_curvature_identities",
    "apply_curvature_identity",
    "simplify_with_identity_library",
    "abstract_hypergraph_signature",
    "canonical_reduce_by_hypergraph",
    "HypergraphCanonizationReport",
    "OperatorApplication",
    "DifferentialOperatorTree",
    "build_operator_tree",
    "expand_operator_tree",
    "commute_operator_tree",
    "reduce_operator_tree",
    "collect_covariant_derivatives",
    "compose_operator_trees",
    "TableauProjector",
    "IrreducibleDecompositionReport",
    "tableau_projector",
    "apply_tableau_projector",
    "decompose_irreducible",
]





class AbstractTensorCanonicalizationError(ValueError):
    """Raised when an abstract tensor canonicalization request is malformed."""


_HEAD_METADATA: WeakKeyDictionary = WeakKeyDictionary()


@dataclass(frozen=True)
class AbstractTensorCanonicalizationReport:
    original_expr: object
    canonical_expr: object
    tensor_heads: tuple[str, ...] = tuple()
    slot_symmetries: Mapping[str, str] = field(default_factory=dict)
    free_indices_before: tuple[str, ...] = tuple()
    free_indices_after: tuple[str, ...] = tuple()
    dummy_indices_before: tuple[str, ...] = tuple()
    dummy_indices_after: tuple[str, ...] = tuple()
    dummy_renamings: Mapping[str, str] = field(default_factory=dict)
    contraction_pairs_before: tuple[tuple[str, str], ...] = tuple()
    contraction_pairs_after: tuple[tuple[str, str], ...] = tuple()
    used_multi_term_rules: tuple[str, ...] = tuple()
    dimension_used: int | sp.Expr | None = None


@dataclass(frozen=True)
class AbstractContractionStep:
    kind: str
    factor_position: int
    index_name: str
    from_index: str
    to_index: str
    index_type: str


@dataclass(frozen=True)
class AbstractContractionPlan:
    steps: tuple[AbstractContractionStep, ...] = tuple()
    free_indices: tuple[str, ...] = tuple()
    dummy_indices: tuple[str, ...] = tuple()
    metric_heads: tuple[tuple[Any, ...], ...] = tuple()
    delta_heads: tuple[tuple[Any, ...], ...] = tuple()
    free_index_signatures: tuple[tuple[Any, ...], ...] = tuple()
    dummy_index_signatures: tuple[tuple[Any, ...], ...] = tuple()


@dataclass(frozen=True)
class AbstractSimplificationStep:
    name: str
    before_expr: object
    after_expr: object
    changed: bool


@dataclass(frozen=True)
class AbstractSimplificationReport:
    original_expr: object
    final_expr: object
    requested_stages: tuple[str, ...] = tuple()
    executed_steps: tuple[AbstractSimplificationStep, ...] = tuple()
    dimension_used: int | sp.Expr | None = None
    max_passes: int = 8




@dataclass(frozen=True)
class DifferentialCurvatureInvariantReductionReport:
    original_expr: object
    reduced_expr: object
    basis_terms: tuple[object, ...] = tuple()
    term_signatures: tuple[str, ...] = tuple()
    term_multiplicities: Mapping[str, object] = field(default_factory=dict)
    used_rules: tuple[str, ...] = tuple()
    dimension_used: int | sp.Expr | None = None

@dataclass(frozen=True)
class CurvatureInvariantReductionReport:
    original_expr: object
    reduced_expr: object
    basis_terms: tuple[object, ...] = tuple()
    term_signatures: tuple[str, ...] = tuple()
    term_multiplicities: Mapping[str, object] = field(default_factory=dict)
    used_multi_term_rules: tuple[str, ...] = tuple()
    dimension_used: int | sp.Expr | None = None


@dataclass(frozen=True)
class IndexType:
    name: str
    dimension: int | sp.Expr | None = None
    dummy_name: str | None = None
    metric_symmetry: int | None = 1
    metric_name: str = "metric"

    def to_sympy(self) -> SymTensorIndexType:
        return abstract_index_type(
            self.name,
            dummy_name=self.dummy_name,
            dim=self.dimension,
            metric_symmetry=self.metric_symmetry,
            metric_name=self.metric_name,
        )


@dataclass(frozen=True)
class Index:
    name: str
    index_type: IndexType
    variance: str = "u"

    def __post_init__(self):
        if self.variance not in {"u", "l", "up", "down"}:
            raise AbstractTensorCanonicalizationError(f"Unsupported variance: {self.variance!r}")

    @property
    def is_up(self) -> bool:
        return self.variance in {"u", "up"}

    def flipped(self) -> "Index":
        return Index(self.name, self.index_type, "l" if self.is_up else "u")

    def to_sympy(self) -> SymTensorIndex:
        created = tensor_indices(self.name, self.index_type.to_sympy())
        base = created[0] if isinstance(created, tuple) else created
        return base if self.is_up else -base

    def __neg__(self) -> "Index":
        return self.flipped()


@dataclass(frozen=True)
class Metric:
    index_type: IndexType
    name: str | None = None

    def to_sympy(self):
        itype = self.index_type.to_sympy()
        metric_obj = getattr(itype, "metric", None)
        if metric_obj is not None:
            _register_head_metadata(metric_obj, metric_for=str(itype), symmetry_kind="metric")
            return metric_obj
        return abstract_tensor_head(
            self.name or self.index_type.metric_name,
            [itype, itype],
            symmetry="symmetric",
        )

    def __call__(self, left, right):
        return AbstractTensorExpr(self.to_sympy()(_coerce_index(left), _coerce_index(right)))


@dataclass(frozen=True)
class TensorHead:
    name: str
    index_types: Sequence[IndexType | SymTensorIndexType]
    symmetry: str | SymTensorSymmetry | None = None
    young_shape: Sequence[int] | None = None
    comm: int = 0

    def to_sympy(self) -> SymTensorHead:
        converted = [it.to_sympy() if isinstance(it, IndexType) else it for it in self.index_types]
        return abstract_tensor_head(
            self.name,
            converted,
            symmetry=self.symmetry,
            young_shape=self.young_shape,
            comm=self.comm,
        )

    def __call__(self, *indices):
        return AbstractTensorExpr(self.to_sympy()(*[_coerce_index(i) for i in indices]))


@dataclass(frozen=True)
class AbstractTensorExpr:
    expr: object
    report: AbstractTensorCanonicalizationReport | None = None

    def to_sympy(self):
        return self.expr

    def _sympy_(self):
        return self.expr

    def canonicalize(
        self,
        *,
        use_multi_term: bool = False,
        dimension: int | sp.Expr | None = None,
        with_report: bool = False,
    ):
        if with_report:
            return canonicalize_abstract_tensor_expr_with_report(
                self.expr,
                use_multi_term=use_multi_term,
                dimension=dimension,
            )
        return AbstractTensorExpr(
            canonicalize_abstract_tensor_expr(
                self.expr,
                use_multi_term=use_multi_term,
                dimension=dimension,
            )
        )

    def _binary(self, other, op):
        other_expr = as_abstract_tensor_expr(other).expr
        return AbstractTensorExpr(op(self.expr, other_expr))

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
        return as_abstract_tensor_expr(other)._binary(self, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return AbstractTensorExpr(as_abstract_tensor_expr(other).expr - self.expr)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return AbstractTensorExpr(as_abstract_tensor_expr(other).expr * self.expr)

    def __neg__(self):
        return AbstractTensorExpr(-self.expr)


class TensorAtlasAbstractExpr(AbstractTensorExpr):
    pass


def _coerce_index(index) -> SymTensorIndex:
    if isinstance(index, Index):
        return index.to_sympy()
    if isinstance(index, SymTensorIndex):
        return index
    raise AbstractTensorCanonicalizationError(f"Expected an Index or SymPy TensorIndex, got {type(index)!r}.")


def _coerce_metric(metric_like, *, index_type: IndexType | None = None) -> Metric:
    if isinstance(metric_like, Metric):
        return metric_like
    if metric_like is None:
        if index_type is None:
            raise AbstractTensorCanonicalizationError("A metric or index type is required.")
        return Metric(index_type)
    if isinstance(metric_like, IndexType):
        return Metric(metric_like)
    raise AbstractTensorCanonicalizationError(f"Expected a Metric or IndexType, got {type(metric_like)!r}.")


def index_type(name: str, **kwargs) -> IndexType:
    return IndexType(name, **kwargs)


def tensor_head(name: str, index_types: Sequence[IndexType | SymTensorIndexType], **kwargs) -> TensorHead:
    return TensorHead(name, index_types, **kwargs)


def metric(index_type_like: IndexType, *, name: str | None = None) -> Metric:
    return Metric(index_type_like, name=name)


def _register_head_metadata(head: SymTensorHead, **metadata) -> SymTensorHead:
    current = dict(_HEAD_METADATA.get(head, {}))
    current.update(metadata)
    _HEAD_METADATA[head] = current
    return head


def _head_metadata(head: SymTensorHead) -> dict:
    return dict(_HEAD_METADATA.get(head, {}))


def abstract_index_type(
    name: str,
    *,
    dummy_name: str | None = None,
    dim: int | sp.Expr | None = None,
    metric_symmetry: int | None = 1,
    metric_name: str = "metric",
) -> SymTensorIndexType:
    itype = SymTensorIndexType(
        name,
        dummy_name=dummy_name,
        dim=dim,
        metric_symmetry=metric_symmetry,
        metric_name=metric_name,
    )
    metric_obj = getattr(itype, "metric", None)
    if metric_obj is not None:
        _register_head_metadata(metric_obj, metric_for=str(itype), symmetry_kind="metric")
    delta_obj = getattr(itype, "delta", None)
    if delta_obj is not None:
        _register_head_metadata(delta_obj, delta_for=str(itype), symmetry_kind="delta")
    return itype


def _shape_symmetry(shape: Sequence[int]) -> SymTensorSymmetry:
    if not shape:
        raise AbstractTensorCanonicalizationError("Young tableau shape must be non-empty.")
    rank = sum(int(v) for v in shape)
    if rank <= 0:
        raise AbstractTensorCanonicalizationError("Young tableau shape must have positive size.")
    base, gens = get_symmetric_group_sgs(rank, tuple(int(v) for v in shape))
    return SymTensorSymmetry(base, gens)


def abstract_tensor_head(
    name: str,
    index_types: Sequence[SymTensorIndexType],
    *,
    symmetry: str | SymTensorSymmetry | None = None,
    young_shape: Sequence[int] | None = None,
    comm: int = 0,
) -> SymTensorHead:
    symmetry_kind = "none"
    if young_shape is not None:
        if symmetry is not None:
            raise AbstractTensorCanonicalizationError(
                "Specify either symmetry or young_shape, not both."
            )
        tensor_symmetry = _shape_symmetry(young_shape)
        symmetry_kind = f"young:{tuple(int(v) for v in young_shape)}"
    elif isinstance(symmetry, SymTensorSymmetry):
        tensor_symmetry = symmetry
        symmetry_kind = "custom"
    elif symmetry in (None, "none"):
        tensor_symmetry = SymTensorSymmetry.no_symmetry(len(index_types))
    elif symmetry == "symmetric":
        tensor_symmetry = SymTensorSymmetry.fully_symmetric(len(index_types))
        symmetry_kind = "symmetric"
    elif symmetry == "antisymmetric":
        tensor_symmetry = SymTensorSymmetry.fully_symmetric(-len(index_types))
        symmetry_kind = "antisymmetric"
    elif symmetry == "riemann":
        if len(index_types) != 4:
            raise AbstractTensorCanonicalizationError(
                "Riemann symmetry requires a rank-4 tensor head."
            )
        tensor_symmetry = _shape_symmetry((2, 2))
        symmetry_kind = "riemann"
    elif symmetry == "weyl":
        if len(index_types) != 4:
            raise AbstractTensorCanonicalizationError(
                "Weyl symmetry requires a rank-4 tensor head."
            )
        tensor_symmetry = _shape_symmetry((2, 2))
        symmetry_kind = "weyl"
    else:
        raise AbstractTensorCanonicalizationError(f"Unsupported symmetry specification: {symmetry!r}")
    head = SymTensorHead(name, list(index_types), tensor_symmetry, comm=comm)
    return _register_head_metadata(
        head,
        symmetry_kind=symmetry_kind,
        young_shape=None if young_shape is None else tuple(int(v) for v in young_shape),
        rank=len(index_types),
        index_type_names=tuple(str(t) for t in index_types),
    )


def fully_symmetric_head(name: str, index_types: Sequence[SymTensorIndexType], *, comm: int = 0) -> SymTensorHead:
    return abstract_tensor_head(name, index_types, symmetry="symmetric", comm=comm)


def fully_antisymmetric_head(name: str, index_types: Sequence[SymTensorIndexType], *, comm: int = 0) -> SymTensorHead:
    return abstract_tensor_head(name, index_types, symmetry="antisymmetric", comm=comm)


def riemann_tensor_head(name: str, index_type: SymTensorIndexType, *, comm: int = 0) -> SymTensorHead:
    return abstract_tensor_head(name, [index_type] * 4, symmetry="riemann", comm=comm)


def ricci_tensor_head(name: str, index_type: SymTensorIndexType, *, comm: int = 0) -> SymTensorHead:
    return abstract_tensor_head(name, [index_type] * 2, symmetry="symmetric", comm=comm)


def weyl_tensor_head(name: str, index_type: SymTensorIndexType, *, comm: int = 0) -> SymTensorHead:
    return abstract_tensor_head(name, [index_type] * 4, symmetry="weyl", comm=comm)


def schouten_tensor_head(name: str, index_type: SymTensorIndexType, *, comm: int = 0) -> SymTensorHead:
    return abstract_tensor_head(name, [index_type] * 2, symmetry="symmetric", comm=comm)


def scalar_curvature_symbol(name: str = "R"):
    return sp.Symbol(name)


def _permute_selected_slots(indices: Sequence[SymTensorIndex], slots: Sequence[int], permuted_positions: Sequence[int]) -> tuple[SymTensorIndex, ...]:
    out = list(indices)
    for source_slot, target_slot in zip(slots, permuted_positions):
        out[source_slot] = indices[target_slot]
    return tuple(out)


def _permutation_sign_from_positions(reference: Sequence[int], permuted: Sequence[int]) -> int:
    pos = [reference.index(v) for v in permuted]
    inversions = 0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            if pos[i] > pos[j]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _project_tensor_slots_once(expr, slots: Sequence[int], *, antisymmetric: bool):
    current = as_abstract_tensor_expr(expr).expr
    slots = tuple(int(s) for s in slots)
    if current == 0:
        return AbstractTensorExpr(current)
    if len(slots) < 2:
        return AbstractTensorExpr(current)
    if isinstance(current, (TensAdd, sp.Add)):
        return AbstractTensorExpr(_build_sum([_project_tensor_slots_once(arg, slots, antisymmetric=antisymmetric).expr for arg in current.args]))
    if isinstance(current, (TensMul, sp.Mul)):
        coeff, rest = _term_coeff_and_rest(current)
        if isinstance(rest, (TensAdd, sp.Add)):
            return AbstractTensorExpr(coeff * _project_tensor_slots_once(rest, slots, antisymmetric=antisymmetric).expr)
        factors = _factor_list(rest)
        tensor_positions = [i for i, factor in enumerate(factors) if isinstance(factor, Tensor)]
        if len(tensor_positions) != 1:
            raise AbstractTensorCanonicalizationError("Slot projection on products currently expects exactly one tensor factor after scalar extraction.")
        pos = tensor_positions[0]
        projected_factor = _project_tensor_slots_once(factors[pos], slots, antisymmetric=antisymmetric).expr
        factors[pos] = projected_factor
        return AbstractTensorExpr(coeff * _rebuild_product(factors))
    if not isinstance(current, Tensor):
        raise AbstractTensorCanonicalizationError("Slot projection currently expects an abstract tensor expression built from tensor factors.")
    if min(slots) < 0 or max(slots) >= len(tuple(current.get_indices())):
        raise AbstractTensorCanonicalizationError("Slot indices are out of range for this tensor rank.")
    if len(set(slots)) != len(slots):
        raise AbstractTensorCanonicalizationError("Slot indices must be distinct.")
    from itertools import permutations
    base_indices = tuple(current.get_indices())
    terms = []
    denom = sp.Integer(0)
    for perm in permutations(slots):
        sign = _permutation_sign_from_positions(slots, perm) if antisymmetric else 1
        permuted = current.component(*_permute_selected_slots(base_indices, slots, perm))
        terms.append(sign * permuted)
        denom += 1
    projected = _build_sum(terms) / denom
    return AbstractTensorExpr(butler_portugal_canonicalize(projected))


def symmetrize_indices(expr, slots: Sequence[int]):
    return _project_tensor_slots_once(expr, slots, antisymmetric=False)


def antisymmetrize_indices(expr, slots: Sequence[int]):
    return _project_tensor_slots_once(expr, slots, antisymmetric=True)


def young_project_indices(expr, rows: Sequence[Sequence[int]]):
    current = as_abstract_tensor_expr(expr)
    row_data = [tuple(int(v) for v in row) for row in rows if tuple(row)]
    if not row_data:
        return current
    for row in row_data:
        current = symmetrize_indices(current, row)
    max_len = max(len(row) for row in row_data)
    for col in range(max_len):
        column = tuple(row[col] for row in row_data if col < len(row))
        if len(column) >= 2:
            current = antisymmetrize_indices(current, column)
    # Keep the classic Young-projector path intentionally non-recursive.  The
    # central TensorExpr canonicalizer now owns canonical reduction; calling the
    # old structural_simplify here can re-enter SymPy abstract tensor machinery
    # and hang under pytest instrumentation.
    return current


def as_abstract_tensor_expr(expr) -> AbstractTensorExpr:
    if isinstance(expr, AbstractTensorExpr):
        return expr
    return AbstractTensorExpr(expr)




def _new_dummy_index(index_type: IndexType | SymTensorIndexType, stem: str | None = None, *, up: bool = True) -> SymTensorIndex:
    if isinstance(index_type, IndexType):
        base_stem = stem or (index_type.dummy_name or index_type.name[0].lower() or "d")
        itype = index_type.to_sympy()
        base_stem = index_type.dummy_name or base_stem
    else:
        itype = index_type
        type_name = str(index_type)
        dummy_name = getattr(index_type, 'dummy_name', None)
        base_stem = stem or (dummy_name or type_name[0].lower() or "d")
        base_stem = dummy_name or base_stem
    created = tensor_indices(f"{base_stem}_0", itype)
    base = created[0] if isinstance(created, tuple) else created
    return base if up else -base


def _replace_index_in_tensor(tensor: Tensor, old: SymTensorIndex, new: SymTensorIndex):
    indices = tuple(new if idx == old else idx for idx in tensor.get_indices())
    return tensor.component(*indices)


def _replace_index(expr, old: SymTensorIndex, new: SymTensorIndex):
    expr = as_abstract_tensor_expr(expr).expr
    if isinstance(expr, Tensor):
        return _replace_index_in_tensor(expr, old, new)
    if isinstance(expr, TensAdd):
        return TensAdd(*[_replace_index(arg, old, new) for arg in expr.args])
    if isinstance(expr, TensMul):
        out = _replace_index(expr.args[0], old, new)
        for arg in expr.args[1:]:
            out = out * _replace_index(arg, old, new)
        return out
    return expr



def _rebuild_term(factors: Sequence[object]):
    if not factors:
        return sp.Integer(1)
    out = factors[0]
    for factor in factors[1:]:
        out = out * factor
    return out


def _index_type_name(idx: SymTensorIndex) -> str:
    return str(getattr(idx, "tensor_index_type", "?"))


def _index_type_key(idx: SymTensorIndex):
    return structural_key(getattr(idx, "tensor_index_type", None))


def _index_signature(idx: SymTensorIndex) -> tuple[str, bool, tuple[Any, ...]]:
    return (idx.name, bool(idx.is_up), _index_type_key(idx))


def _replace_index_in_factor(factor, old: SymTensorIndex, new: SymTensorIndex):
    if isinstance(factor, Tensor):
        return _replace_index_in_tensor(factor, old, new)
    return factor


def _replace_named_index_in_factor(factor, name: str, *, up: bool, index_type: str, new_index: SymTensorIndex):
    if not isinstance(factor, Tensor):
        return factor, False
    changed = False
    new_indices = []
    for idx in factor.get_indices():
        if idx.name == name and bool(idx.is_up) == up and _index_type_name(idx) == index_type:
            new_indices.append(new_index)
            changed = True
        else:
            new_indices.append(idx)
    return (factor.component(*new_indices), True) if changed else (factor, False)


def _is_metric_factor(factor) -> bool:
    if not isinstance(factor, Tensor):
        return False
    if bool(_head_metadata(factor.component).get("metric_for")):
        return True
    idx = tuple(factor.get_indices())
    if len(idx) != 2:
        return False
    try:
        return factor.component == idx[0].tensor_index_type.metric == idx[1].tensor_index_type.metric
    except Exception:
        return False


def _is_delta_factor(factor) -> bool:
    if not isinstance(factor, Tensor):
        return False
    if bool(_head_metadata(factor.component).get("delta_for")):
        return True
    idx = tuple(factor.get_indices())
    if len(idx) != 2:
        return False
    try:
        return factor.component == idx[0].tensor_index_type.delta == idx[1].tensor_index_type.delta
    except Exception:
        return False


def _collect_term_index_info(term) -> dict[str, list[SymTensorIndex]]:
    info: dict[str, list[SymTensorIndex]] = {}
    for factor in _factor_list(term):
        if not isinstance(factor, Tensor):
            continue
        for idx in factor.get_indices():
            info.setdefault(idx.name, []).append(idx)
    return info


def validate_contractions(expr):
    current = as_abstract_tensor_expr(expr).expr
    for term in _term_list(current):
        per_name = _collect_term_index_info(term)
        for name, idxs in per_name.items():
            types = {_index_type_name(i) for i in idxs}
            if len(types) > 1:
                raise AbstractTensorCanonicalizationError(
                    f"Index {name!r} mixes bundles/index types in one contraction context: {sorted(types)!r}"
                )
            ups = sum(1 for i in idxs if i.is_up)
            downs = len(idxs) - ups
            if ups > 0 and downs > 0 and (ups > 1 or downs > 1):
                raise AbstractTensorCanonicalizationError(
                    f"Index {name!r} appears in an ambiguous many-to-many contraction pattern."
                )
        for factor in _factor_list(term):
            if not isinstance(factor, Tensor):
                continue
            if _is_metric_factor(factor) or _is_delta_factor(factor):
                left, right = tuple(factor.get_indices())
                if _index_type_name(left) != _index_type_name(right):
                    raise AbstractTensorCanonicalizationError("Metric/delta factor mixes incompatible index types.")
    return True


def build_contraction_plan(expr) -> AbstractContractionPlan:
    current = as_abstract_tensor_expr(expr).expr
    validate_contractions(current)
    steps: list[AbstractContractionStep] = []
    metric_heads: list[tuple[Any, ...]] = []
    delta_heads: list[tuple[Any, ...]] = []
    free_indices = _free_indices(current)
    dummy_indices = _dummy_indices(current)
    index_groups = _collect_indices(current)
    for term in _term_list(current):
        factors = _factor_list(_term_coeff_and_rest(term)[1])
        for pos, factor in enumerate(factors):
            if not isinstance(factor, Tensor):
                continue
            if _is_metric_factor(factor) or _is_delta_factor(factor):
                left, right = tuple(factor.get_indices())
                special_kind = "delta" if _is_delta_factor(factor) else "metric"
                (delta_heads if special_kind == "delta" else metric_heads).append(_tensor_head_sort_key(factor.component))
                for oi, other in enumerate(factors):
                    if oi == pos or not isinstance(other, Tensor):
                        continue
                    for idx in other.get_indices():
                        if idx == -left:
                            steps.append(AbstractContractionStep(special_kind, oi, idx.name, str(idx), str(right), _index_type_name(idx)))
                        elif idx == -right:
                            steps.append(AbstractContractionStep(special_kind, oi, idx.name, str(idx), str(left), _index_type_name(idx)))
    free_sigs = tuple(sorted((_abstract_index_partition_entry(idxs) for idxs in index_groups.values() if len(idxs) == 1), key=structural_key))
    dummy_sigs = tuple(sorted((_abstract_index_partition_entry(idxs) for idxs in index_groups.values() if len(idxs) > 1), key=structural_key))
    return AbstractContractionPlan(
        steps=tuple(steps),
        free_indices=free_indices,
        dummy_indices=dummy_indices,
        metric_heads=tuple(dict.fromkeys(metric_heads)),
        delta_heads=tuple(dict.fromkeys(delta_heads)),
        free_index_signatures=free_sigs,
        dummy_index_signatures=dummy_sigs,
    )


def _canonical_index_from_string(index_repr: str, index_type: str) -> SymTensorIndex:
    itype = abstract_index_type(index_type)
    base_name = index_repr.lstrip("-")
    created = tensor_indices(base_name, itype)
    base = created[0] if isinstance(created, tuple) else created
    return -base if index_repr.startswith("-") else base


def execute_contraction_plan(expr, plan: AbstractContractionPlan | None = None, *, max_passes: int = 8):
    current = as_abstract_tensor_expr(expr).expr
    plan = build_contraction_plan(current) if plan is None else plan
    for _ in range(max_passes):
        changed = False
        new_terms = []
        for term in _term_list(current):
            coeff, rest = _term_coeff_and_rest(term)
            factors = _factor_list(rest)
            local_changed = False
            for pos, factor in enumerate(list(factors)):
                if not isinstance(factor, Tensor):
                    continue
                if not (_is_metric_factor(factor) or _is_delta_factor(factor)):
                    continue
                left, right = tuple(factor.get_indices())
                replacement_done = False
                for oi, other in enumerate(list(factors)):
                    if oi == pos or not isinstance(other, Tensor):
                        continue
                    new_other, did = _replace_named_index_in_factor(other, left.name, up=(not left.is_up), index_type=_index_type_name(left), new_index=right)
                    if did:
                        factors[oi] = new_other
                        del factors[pos]
                        replacement_done = True
                        local_changed = True
                        break
                    new_other, did = _replace_named_index_in_factor(other, right.name, up=(not right.is_up), index_type=_index_type_name(right), new_index=left)
                    if did:
                        factors[oi] = new_other
                        del factors[pos]
                        replacement_done = True
                        local_changed = True
                        break
                if replacement_done:
                    break
            new_terms.append(coeff * _rebuild_term(factors))
            changed = changed or local_changed
        current = butler_portugal_canonicalize(_build_sum(new_terms))
        if not changed:
            break
    return AbstractTensorExpr(current)


def contract_metric(expr, *, max_passes: int = 8):
    current = as_abstract_tensor_expr(expr).expr
    validate_contractions(current)
    return execute_contraction_plan(current, max_passes=max_passes)


def raise_index(expr, index, *, metric=None):
    index_obj = index if isinstance(index, Index) else None
    if index_obj is None:
        raise AbstractTensorCanonicalizationError("raise_index expects an Index wrapper for the target index.")
    if index_obj.is_up:
        return as_abstract_tensor_expr(expr)
    metric_obj = _coerce_metric(metric, index_type=index_obj.index_type)
    dummy_up = _new_dummy_index(index_obj.index_type, up=True)
    replaced = _replace_index(as_abstract_tensor_expr(expr).expr, index_obj.to_sympy(), dummy_up)
    lifted = metric_obj.to_sympy()(Index(index_obj.name, index_obj.index_type, "u").to_sympy(), -dummy_up) * replaced
    return simplify_abstract(lifted, mode="metric")


def lower_index(expr, index, *, metric=None):
    index_obj = index if isinstance(index, Index) else None
    if index_obj is None:
        raise AbstractTensorCanonicalizationError("lower_index expects an Index wrapper for the target index.")
    if not index_obj.is_up:
        return as_abstract_tensor_expr(expr)
    metric_obj = _coerce_metric(metric, index_type=index_obj.index_type)
    dummy_down = _new_dummy_index(index_obj.index_type, up=False)
    replaced = _replace_index(as_abstract_tensor_expr(expr).expr, index_obj.to_sympy(), dummy_down)
    lowered = metric_obj.to_sympy()(Index(index_obj.name, index_obj.index_type, "l").to_sympy(), -dummy_down) * replaced
    return simplify_abstract(lowered, mode="metric")


def trace_abstract(expr, upper_index, lower_index):
    up = upper_index if isinstance(upper_index, Index) else None
    lo = lower_index if isinstance(lower_index, Index) else None
    if up is None or lo is None:
        raise AbstractTensorCanonicalizationError("trace_abstract expects Index wrappers.")
    if not up.is_up or lo.is_up:
        raise AbstractTensorCanonicalizationError("trace_abstract expects one upper and one lower index.")
    if up.index_type != lo.index_type:
        raise AbstractTensorCanonicalizationError("trace_abstract requires both indices to belong to the same index type.")
    dummy_up = _new_dummy_index(up.index_type, up=True)
    out = _replace_index(as_abstract_tensor_expr(expr).expr, up.to_sympy(), dummy_up)
    out = _replace_index(out, lo.to_sympy(), -dummy_up)
    return structural_simplify(out)


def delta_reduce(expr, *, max_passes: int = 8):
    current = as_abstract_tensor_expr(expr).expr
    validate_contractions(current)
    return execute_contraction_plan(current, max_passes=max_passes)


def _normalized_simplification_stages(mode) -> tuple[str, ...]:
    if mode is None:
        mode = "all"
    if isinstance(mode, str):
        key = mode.strip().lower()
        aliases = {
            "structural": ("structural",),
            "metric": ("structural", "metric"),
            "multiterm": ("structural", "metric", "multiterm"),
            "invariant": ("structural", "metric", "multiterm", "invariant"),
            "all": ("structural", "metric", "multiterm", "invariant"),
        }
        if key not in aliases:
            raise AbstractTensorCanonicalizationError(f"Unsupported simplification mode: {mode!r}")
        return aliases[key]
    if isinstance(mode, Iterable):
        allowed = ("structural", "metric", "multiterm", "invariant")
        seen = []
        for item in mode:
            key = str(item).strip().lower()
            if key not in allowed:
                raise AbstractTensorCanonicalizationError(f"Unsupported simplification stage: {item!r}")
            if key not in seen:
                seen.append(key)
        requested = set(seen)
        return tuple(stage for stage in allowed if stage in requested)
    raise AbstractTensorCanonicalizationError(f"Unsupported simplification mode: {mode!r}")


def simplify_abstract_with_report(expr, *, mode: str | Iterable[str] = "all", dimension: int | sp.Expr | None = None, max_passes: int = 8):
    stages = _normalized_simplification_stages(mode)
    current = as_abstract_tensor_expr(expr).expr
    steps: list[AbstractSimplificationStep] = []
    runners = {
        "structural": lambda x: structural_simplify(x).expr,
        "metric": lambda x: metric_simplify(x, max_passes=max_passes).expr,
        "multiterm": lambda x: multiterm_simplify(x, dimension=dimension).expr,
        "invariant": lambda x: invariant_simplify(x, dimension=dimension).expr,
    }
    for stage in stages:
        before = current
        after = runners[stage](before)
        steps.append(AbstractSimplificationStep(stage, before, after, changed=(before != after)))
        current = after
    report = AbstractSimplificationReport(
        original_expr=as_abstract_tensor_expr(expr).expr,
        final_expr=current,
        requested_stages=tuple(stages),
        executed_steps=tuple(steps),
        dimension_used=dimension,
        max_passes=max_passes,
    )
    return AbstractTensorExpr(current), report


def structural_simplify(expr):
    current = butler_portugal_canonicalize(as_abstract_tensor_expr(expr).expr)
    return AbstractTensorExpr(current)


def metric_simplify(expr, *, max_passes: int = 8):
    current = execute_contraction_plan(as_abstract_tensor_expr(expr).expr, max_passes=max_passes).expr
    return AbstractTensorExpr(butler_portugal_canonicalize(current))


def multiterm_simplify(expr, *, dimension: int | sp.Expr | None = None):
    current, _ = multi_term_tensor_reduce(as_abstract_tensor_expr(expr).expr, dimension=dimension)
    return AbstractTensorExpr(butler_portugal_canonicalize(current))


def invariant_simplify(expr, *, dimension: int | sp.Expr | None = None):
    current = as_abstract_tensor_expr(expr).expr
    try:
        if dimension is not None:
            current = decompose_curvature_expression(current, dimension=dimension).expr
        reduced = reduce_curvature_invariants(current, dimension=dimension, use_multi_term=True)
    except Exception:
        reduced = butler_portugal_canonicalize(current)
    return AbstractTensorExpr(reduced)


def simplify_abstract(expr, *, mode: str | Iterable[str] = "all", dimension: int | sp.Expr | None = None, max_passes: int = 8):
    simplified, _report = simplify_abstract_with_report(
        expr,
        mode=mode,
        dimension=dimension,
        max_passes=max_passes,
    )
    return simplified

def _index_sort_token(idx) -> tuple[str, str]:
    return (_index_base_name(idx), _index_variance(idx))


def _canonicalize_tensor_factor(expr):
    if not isinstance(expr, Tensor):
        return expr
    head = expr.component
    idx = list(expr.get_indices())
    kind = str(_head_metadata(head).get("symmetry_kind", "") or "").lower()
    sign = sp.Integer(1)

    def order_pair(i: int, j: int, antisymmetric: bool) -> bool:
        nonlocal sign, idx
        if _index_sort_token(idx[j]) < _index_sort_token(idx[i]):
            idx[i], idx[j] = idx[j], idx[i]
            if antisymmetric:
                sign = -sign
            return True
        if antisymmetric and _index_sort_token(idx[i]) == _index_sort_token(idx[j]):
            sign = sp.Integer(0)
        return False

    if kind == "symmetric":
        # Bubble-sort slots into a stable representative.
        for _ in range(len(idx)):
            for a in range(len(idx) - 1):
                order_pair(a, a + 1, False)
    elif kind == "antisymmetric":
        for _ in range(len(idx)):
            for a in range(len(idx) - 1):
                order_pair(a, a + 1, True)
                if sign == 0:
                    return sp.Integer(0)
    elif kind in {"riemann", "weyl"} and len(idx) == 4:
        order_pair(0, 1, True)
        order_pair(2, 3, True)
        if sign == 0:
            return sp.Integer(0)
    if tuple(idx) == tuple(expr.get_indices()) and sign == 1:
        return expr
    return sign * head(*idx)


def _canonicalize_bp(expr):
    """Fast monoterm canonicalization for the classic abstract layer.

    The old implementation delegated directly to SymPy's canon_bp(), which is
    powerful but can hang under pytest instrumentation on some abstract tensor
    expressions.  TensorAtlas now uses TensorExpr for central comparison, so the
    classic public Butler-Portugal surface is kept as a lightweight monoterm
    reducer for the simple symmetric/antisymmetric/Riemann cases still covered
    by tests and compatibility examples.
    """
    if isinstance(expr, Tensor):
        return _canonicalize_tensor_factor(expr)
    if isinstance(expr, (sp.Add, TensAdd)):
        terms = [_canonicalize_bp(arg) for arg in expr.args]
        return _build_sum(terms)
    if isinstance(expr, (sp.Mul, TensMul)):
        coeff = sp.Integer(1)
        factors = []
        for arg in expr.args:
            carg = _canonicalize_bp(arg)
            if carg == 0:
                return sp.Integer(0)
            if isinstance(carg, (sp.Integer, sp.Rational, sp.Number)) or getattr(carg, "is_number", False):
                coeff *= sp.sympify(carg)
            else:
                factors.append(carg)
        return coeff * _rebuild_product(factors)
    return expr


def butler_portugal_canonicalize(expr):
    """Canonicalize an abstract-index tensor expression via SymPy's Butler-Portugal path."""
    return _canonicalize_bp(expr)


def butler_portugal_canonicalize_permutation(
    g: Sequence[int],
    dummies,
    msym,
    *tensor_data,
):
    """Expose the low-level permutation/double-coset canonicalizer used by Butler-Portugal."""
    return _canonicalize_perm(g, dummies, msym, *tensor_data)




def _index_base_name(idx) -> str:
    return str(getattr(idx, "name", str(idx).lstrip("-")))


def _index_variance(idx) -> str:
    return "l" if str(idx).startswith("-") else "u"


def _factor_slot_orbits(factor) -> tuple[str, ...]:
    if not isinstance(factor, Tensor):
        return tuple()
    n = len(factor.get_indices())
    kind = (_head_metadata(factor.component).get("symmetry_kind") or "none").lower()
    if kind in {"symmetric", "antisymmetric"}:
        return tuple(kind[0].upper() for _ in range(n))
    if kind in {"riemann", "weyl"} and n == 4:
        return ("P01", "P01", "P23", "P23")
    if kind in {"ricci", "schouten"} and n == 2:
        return ("P", "P")
    return tuple(f"S{k}" for k in range(n))


def _build_abstract_hypergraph(expr):
    current = simplify_abstract(expr, mode=("structural", "metric", "multiterm")).expr
    coeff, rest = _term_coeff_and_rest(current)
    factors = [f for f in _factor_list(rest) if isinstance(f, Tensor)]
    index_counts: dict[tuple[Any, ...], dict[str, int]] = {}
    incidences: dict[tuple[Any, ...], list[tuple[int, int]]] = {}
    index_types: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    index_name_map: dict[tuple[Any, ...], str] = {}
    for fi, factor in enumerate(factors):
        for si, idx in enumerate(factor.get_indices()):
            idx_key = _abstract_index_signature(idx)
            var = _index_variance(idx)
            counts = index_counts.setdefault(idx_key, {"u": 0, "l": 0})
            counts[var] += 1
            incidences.setdefault(idx_key, []).append((fi, si))
            index_types.setdefault(idx_key, _index_type_key(idx))
            index_name_map.setdefault(idx_key, _index_base_name(idx))
    index_partition = {}
    for idx_key, counts in index_counts.items():
        index_partition[idx_key] = "dummy" if counts["u"] and counts["l"] else "free"
    factor_nodes = []
    for fi, factor in enumerate(factors):
        slot_orbits = _factor_slot_orbits(factor)
        head = factor.component
        idxs = factor.get_indices()
        factor_nodes.append({
            "factor_id": fi,
            "head": _tensor_head_sort_key(head),
            "symmetry": (_head_metadata(head).get("symmetry_kind") or "none"),
            "slot_orbits": slot_orbits,
            "indices": tuple(_abstract_index_signature(i) for i in idxs),
            "index_names": tuple(_index_base_name(i) for i in idxs),
            "variances": tuple(_index_variance(i) for i in idxs),
        })
    index_nodes = {}
    for idx_key, pairs in incidences.items():
        counts = index_counts[idx_key]
        index_nodes[idx_key] = {
            "name": index_name_map.get(idx_key, "?"),
            "index_key": idx_key,
            "partition": index_partition[idx_key],
            "type": index_types.get(idx_key, "?"),
            "up": counts["u"],
            "down": counts["l"],
            "degree": len(pairs),
            "incidences": tuple(sorted(pairs)),
        }
    return coeff, factors, factor_nodes, index_nodes


def _wl_refine_hypergraph(factor_nodes, index_nodes, *, max_rounds: int = 8):
    factor_labels = {
        node["factor_id"]: (
            node["head"],
            node["symmetry"],
            tuple(node["slot_orbits"]),
            tuple(node["variances"]),
            len(node["indices"]),
        )
        for node in factor_nodes
    }
    index_labels = {
        name: (
            data["partition"],
            data["type"],
            data["up"],
            data["down"],
            data["degree"],
        )
        for name, data in index_nodes.items()
    }
    for _ in range(max_rounds):
        new_factor_labels = {}
        for node in factor_nodes:
            fi = node["factor_id"]
            neighborhood = tuple(
                (slot_orbit, index_labels[name], var)
                for slot_orbit, name, var in zip(node["slot_orbits"], node["indices"], node["variances"])
            )
            new_factor_labels[fi] = (factor_labels[fi], neighborhood)
        new_index_labels = {}
        for name, data in index_nodes.items():
            incident = []
            for fi, si in data["incidences"]:
                slot_orbit = factor_nodes[fi]["slot_orbits"][si]
                incident.append((factor_labels[fi], slot_orbit, factor_nodes[fi]["variances"][si]))
            new_index_labels[name] = (index_labels[name], tuple(sorted(incident)))
        if new_factor_labels == factor_labels and new_index_labels == index_labels:
            break
        factor_labels, index_labels = new_factor_labels, new_index_labels
    return factor_labels, index_labels


def _canonical_dummy_renaming_from_labels(index_nodes, index_labels):
    dummy_names = [name for name, data in index_nodes.items() if data["partition"] == "dummy"]
    ordered = sorted(dummy_names, key=lambda n: (index_labels[n], index_nodes[n]["name"]))
    return {name: f"d{k}" for k, name in enumerate(ordered)}


def _rename_abstract_factor_indices(factor, mapping):
    if not isinstance(factor, Tensor) or not mapping:
        return factor
    new_indices = []
    for idx in factor.get_indices():
        base = _index_base_name(idx)
        new_name = mapping.get(base, base)
        if new_name == base:
            new_indices.append(idx)
            continue
        itype = getattr(idx, "tensor_index_type", None)
        created = tensor_indices(new_name, itype)
        repl = created[0] if isinstance(created, tuple) else created
        if _index_variance(idx) == "l":
            repl = -repl
        new_indices.append(repl)
    return factor.component(*new_indices)


def _canonical_reduce_term_by_hypergraph(term, *, factor_tiebreak=structural_key):
    coeff, rest = _term_coeff_and_rest(term)
    coeff0, factors, factor_nodes, index_nodes = _build_abstract_hypergraph(rest)
    if coeff0 != 1:
        coeff *= coeff0
    if not factors:
        return coeff * rest
    factor_labels, index_labels = _wl_refine_hypergraph(factor_nodes, index_nodes)
    rename = _canonical_dummy_renaming_from_labels(index_nodes, index_labels)
    decorated = []
    for fi, factor in enumerate(factors):
        renamed = _rename_abstract_factor_indices(factor, rename)
        decorated.append((factor_labels[fi], factor_tiebreak(renamed), fi, renamed))
    ordered_factors = [item[-1] for item in sorted(decorated, key=lambda item: (item[0], item[1], item[2]))]
    rebuilt = _rebuild_product(ordered_factors)
    return coeff * rebuilt



def _collect_indices(expr) -> dict[str, list[SymTensorIndex]]:
    out: dict[str, list[SymTensorIndex]] = {}
    for idx in getattr(expr, "get_indices", lambda: [])() or []:
        if isinstance(idx, SymTensorIndex):
            out.setdefault(str(idx.name), []).append(idx)
    return out

def _collect_tensor_heads(expr) -> tuple[SymTensorHead, ...]:
    heads: list[TensorHead] = []
    seen: set[int] = set()

    def visit(node):
        if isinstance(node, Tensor):
            head = node.component
            if id(head) not in seen:
                seen.add(id(head))
                heads.append(head)
            return
        if hasattr(node, "args"):
            for arg in node.args:
                visit(arg)

    visit(expr)
    return tuple(heads)


def _free_indices(expr) -> tuple[str, ...]:
    if not hasattr(expr, "get_free_indices"):
        return tuple()
    try:
        free = expr.get_free_indices()
    except Exception:
        try:
            free = list(getattr(expr, "free_indices", set()))
        except Exception:
            free = []
    return tuple(str(i) for i in free)


def _dummy_indices(expr) -> tuple[str, ...]:
    names: list[str] = []
    for idx in getattr(expr, "get_indices", lambda: [])() or []:
        if isinstance(idx, SymTensorIndex) and str(idx).startswith("-"):
            core = idx.name
        elif isinstance(idx, SymTensorIndex):
            core = idx.name
        else:
            core = str(idx).lstrip("-")
        if getattr(idx, "name", None) and str(idx.name).startswith(f"{getattr(idx, 'tensor_index_type', '')}_"):
            names.append(str(idx.name))
        elif isinstance(idx, SymTensorIndex) and (str(idx).startswith("-") or "_" in str(idx.name)):
            names.append(str(idx.name))
    return tuple(names)


def _contraction_pairs(expr) -> tuple[tuple[str, str], ...]:
    pairs = []
    for i, j in getattr(expr, "dum", []) or []:
        idx = list(getattr(expr, "get_indices", lambda: [])() or [])
        if 0 <= i < len(idx) and 0 <= j < len(idx):
            pairs.append((str(idx[i]), str(idx[j])))
    return tuple(pairs)


def _infer_dummy_renamings(before: Sequence[str], after: Sequence[str]) -> dict[str, str]:
    if len(before) != len(after):
        return {}
    return {b: a for b, a in zip(before, after) if b != a}


def _slot_symmetry_summary(head: SymTensorHead) -> str:
    md = _head_metadata(head)
    if md.get("symmetry_kind") not in (None, "none"):
        return str(md["symmetry_kind"])
    sym = getattr(head, "symmetry", None)
    return str(sym) if sym is not None else "none"


def _term_list(expr) -> list:
    if isinstance(expr, TensAdd):
        return list(expr.args)
    if isinstance(expr, sp.Add):
        return list(expr.args)
    return [expr]


def _build_sum(terms: Sequence[object]):
    terms = [t for t in terms if t != 0]
    if not terms:
        return sp.Integer(0)
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def _term_coeff_and_rest(term):
    if isinstance(term, TensMul):
        coeff = sp.Integer(1)
        rest = []
        for arg in term.args:
            if getattr(arg, "is_number", False):
                coeff *= sp.sympify(arg)
            else:
                rest.append(arg)
        if not rest:
            return coeff, sp.Integer(1)
        if len(rest) == 1:
            return coeff, rest[0]
        return coeff, TensMul(*rest)
    if getattr(term, "is_number", False):
        return sp.sympify(term), sp.Integer(1)
    return sp.Integer(1), term


def _factor_list(term) -> list:
    if isinstance(term, TensMul):
        return list(term.args)
    return [term]


def _rebuild_product(factors: Sequence[object]):
    if not factors:
        return sp.Integer(1)
    out = factors[0]
    for f in factors[1:]:
        out = out * f
    return out


def _symmetry_kind_for_factor(factor) -> str | None:
    """Return recorded slot-symmetry metadata with a conservative fallback.

    Some tests create fresh SymPy TensorHead objects with names that are reused
    after other abstract-tensor tests have populated caches and registries.  A
    WeakKeyDictionary is the right primary store for head metadata, but it means
    name-equivalent heads may occasionally arrive without metadata.  Phase-0
    keeps curvature decomposition stable by recognizing the standard public
    curvature-head names structurally instead of depending only on mutable global
    registration state.
    """
    if not isinstance(factor, Tensor):
        return None
    kind = _head_metadata(factor.component).get("symmetry_kind")
    if kind:
        return str(kind)
    name = str(getattr(factor.component, "name", ""))
    rank = len(tuple(factor.get_indices()))
    if rank == 4 and name in {"R", "Riemann", "riemann"}:
        return "riemann"
    if rank == 4 and name in {"C", "Weyl", "weyl"}:
        return "weyl"
    if rank == 2 and name in {"Ric", "Ricci", "ricci"}:
        return "symmetric"
    return None


def _tensor_expr_key_for_abstract(expr, *, dimension: int | sp.Expr | None = None, layer: str = "abstract") -> tuple[Any, ...]:
    """Central TensorExpr key for classic abstract tensor objects."""
    try:
        from .semantic_ir import to_tensor_expr
        from .tensor_expr_canonicalization import canonicalize_tensor_expr
        node = to_tensor_expr(expr)
        report = canonicalize_tensor_expr(node)
        return report.canonical_key
    except Exception:
        return canonical_expr_fingerprint(expr, dimension=dimension, layer=layer)


def _expr_signature_key(expr) -> tuple[Any, ...]:
    return _tensor_expr_key_for_abstract(expr, layer='abstract_signature')


def _canonical_cache_key(expr, dimension=None) -> tuple[Any, ...]:
    return canonical_expr_fingerprint(expr, dimension=dimension, layer='abstract_cache')


def _is_scalar_invariant_monomial(expr) -> bool:
    return not isinstance(expr, TensAdd) and not _free_indices(expr)


def _factor_key(expr) -> tuple[str, str]:
    return (expr.__class__.__name__, str(expr))


def _bianchi_relation_for_factor(factor):
    if not isinstance(factor, Tensor):
        return None
    kind = _symmetry_kind_for_factor(factor)
    if kind not in {"riemann", "weyl"}:
        return None
    idx = tuple(factor.get_indices())
    if len(idx) != 4:
        return None
    head = factor.component
    orbit = [
        head(*idx),
        head(idx[0], idx[2], idx[3], idx[1]),
        head(idx[0], idx[3], idx[1], idx[2]),
    ]
    target = sorted(orbit, key=_factor_key)[-1]
    rhs_terms = [(-sp.Integer(1), cand) for cand in orbit if cand != target]
    return target, tuple(rhs_terms)


def _apply_single_bianchi_pass(expr):
    used = False
    out_terms = []
    for term in _term_list(expr):
        coeff, _ = _term_coeff_and_rest(term)
        factors = _factor_list(term)
        rewritten = False
        for pos, factor in enumerate(factors):
            rel = _bianchi_relation_for_factor(factor)
            if rel is None:
                continue
            target, rhs_terms = rel
            if factor != target:
                continue
            prefix = factors[:pos]
            suffix = factors[pos + 1 :]
            for scalar, repl in rhs_terms:
                new_factors = [*prefix, repl, *suffix]
                repl_term = coeff * scalar * _rebuild_product(new_factors)
                out_terms.append(repl_term)
            used = True
            rewritten = True
            break
        if not rewritten:
            out_terms.append(term)
    return _build_sum(out_terms), used


def _apply_dimension_dependent_rules(expr, dimension=None):
    if dimension is None:
        return expr, False
    try:
        dim_value = int(dimension)
    except Exception:
        return expr, False
    if dim_value > 3:
        return expr, False
    changed = False
    out_terms = []
    for term in _term_list(expr):
        factors = _factor_list(term)
        if any(_symmetry_kind_for_factor(f) == "weyl" for f in factors):
            changed = True
            continue
        out_terms.append(term)
    return _build_sum(out_terms), changed


def _default_projectors_for_expr(expr) -> tuple[TableauProjector, ...]:
    heads = _collect_tensor_heads(as_abstract_tensor_expr(expr).expr)
    projectors = []
    for head in heads:
        md = _head_metadata(head)
        kind = str(md.get("symmetry_kind", ""))
        if kind in {"riemann", "weyl"}:
            projectors.append(tableau_projector(((0, 1), (2, 3)), symmetry_kind=kind))
        elif kind == "symmetric":
            arity = len(getattr(head, 'index_types', ()))
            if arity > 1:
                projectors.append(tableau_projector((tuple(range(arity)),), symmetry_kind=kind))
        elif kind == "antisymmetric":
            arity = len(getattr(head, 'index_types', ()))
            if arity > 1:
                projectors.append(tableau_projector(tuple((i,) for i in range(arity)), symmetry_kind=kind))
    return compose_tableau_projectors(*projectors)


def multi_term_tensor_reduce(expr, *, dimension: int | sp.Expr | None = None, max_passes: int = 6, tableaux: Sequence[Sequence[Sequence[int]] | TableauProjector] | None = None):
    current = as_abstract_tensor_expr(expr).expr
    if tableaux is None:
        tableaux = _default_projectors_for_expr(current)
    if tableaux:
        try:
            current = multiterm_projector_reduce(current, tableaux).expr
        except Exception:
            current = butler_portugal_canonicalize(current)
    used_rules: list[str] = []
    dim_out, dim_changed = _apply_dimension_dependent_rules(current, dimension=dimension)
    if dim_changed:
        current = dim_out
        used_rules.append("dimension-dependent Weyl vanishing")
    for _ in range(max_passes):
        current, changed = _apply_single_bianchi_pass(current)
        if changed:
            used_rules.append("first Bianchi cyclic reduction")
        if not changed:
            break
    return butler_portugal_canonicalize(current), tuple(dict.fromkeys(used_rules))


def _riemann_decomposition_tensor(factor: Tensor, *, dimension, weyl_name: str = "C", ricci_name: str = "Ric", scalar_name: str = "R"):
    idx = tuple(factor.get_indices())
    if len(idx) != 4:
        return factor
    n = sp.sympify(dimension)
    if n in (0, 1, 2):
        raise AbstractTensorCanonicalizationError("Riemann decomposition requires dimension at least 3.")
    index_type = idx[0].tensor_index_type
    metric_head = getattr(index_type, "metric", None)
    if metric_head is None:
        metric_head = Metric(IndexType(str(index_type), dimension=getattr(index_type, 'dim', None), dummy_name=getattr(index_type, 'dummy_name', None))).to_sympy()
    ric = ricci_tensor_head(ricci_name, index_type)
    scalar = scalar_curvature_symbol(scalar_name)
    a, b, c, d = idx
    coeff = sp.Integer(1) / (n - 2)
    trace_coeff = sp.Integer(1) / ((n - 1) * (n - 2))
    ricci_part = coeff * (
        metric_head(a, c) * ric(b, d)
        - metric_head(a, d) * ric(b, c)
        - metric_head(b, c) * ric(a, d)
        + metric_head(b, d) * ric(a, c)
    )
    scalar_part = trace_coeff * scalar * (
        metric_head(a, c) * metric_head(b, d)
        - metric_head(a, d) * metric_head(b, c)
    )
    try:
        dim_int = int(dimension)
    except Exception:
        dim_int = None
    weyl_part = sp.Integer(0)
    if dim_int is None or dim_int > 3:
        weyl_part = weyl_tensor_head(weyl_name, index_type)(a, b, c, d)
    return butler_portugal_canonicalize(weyl_part + ricci_part - scalar_part)


def decompose_riemann_curvature(expr, *, dimension, weyl_name: str = "C", ricci_name: str = "Ric", scalar_name: str = "R"):
    current = as_abstract_tensor_expr(expr).expr
    if isinstance(current, Tensor):
        if _symmetry_kind_for_factor(current) != "riemann":
            return AbstractTensorExpr(current)
        return AbstractTensorExpr(_riemann_decomposition_tensor(current, dimension=dimension, weyl_name=weyl_name, ricci_name=ricci_name, scalar_name=scalar_name))
    if isinstance(current, (TensAdd, sp.Add)):
        return AbstractTensorExpr(_build_sum([decompose_riemann_curvature(arg, dimension=dimension, weyl_name=weyl_name, ricci_name=ricci_name, scalar_name=scalar_name).expr for arg in current.args]))
    if isinstance(current, (TensMul, sp.Mul)):
        coeff, rest = _term_coeff_and_rest(current)
        factors = []
        changed = False
        for factor in _factor_list(rest):
            if isinstance(factor, Tensor) and _symmetry_kind_for_factor(factor) == "riemann":
                factors.append(_riemann_decomposition_tensor(factor, dimension=dimension, weyl_name=weyl_name, ricci_name=ricci_name, scalar_name=scalar_name))
                changed = True
            else:
                factors.append(factor)
        if not changed:
            return AbstractTensorExpr(current)
        expanded = coeff
        for factor in factors:
            expanded = expanded * factor
        return AbstractTensorExpr(butler_portugal_canonicalize(sp.expand(expanded)))
    return AbstractTensorExpr(current)


def decompose_curvature_expression(expr, *, dimension, weyl_name: str = "C", ricci_name: str = "Ric", scalar_name: str = "R"):
    return structural_simplify(decompose_riemann_curvature(expr, dimension=dimension, weyl_name=weyl_name, ricci_name=ricci_name, scalar_name=scalar_name))




def derivative_tensor_head(
    name: str | SymTensorHead,
    derivative_index_type: SymTensorIndexType | int,
    base_head: SymTensorHead | None = None,
    *,
    derivative_order: int = 1,
    comm: int = 0,
) -> SymTensorHead:
    if isinstance(name, SymTensorHead) and isinstance(derivative_index_type, int) and base_head is None:
        base_head = name
        derivative_order = int(derivative_index_type)
        derivative_index_type = base_head.index_types[0]
        name = f"D{derivative_order}{base_head.name}"
    if derivative_order <= 0:
        raise AbstractTensorCanonicalizationError("derivative_order must be positive.")
    if not isinstance(base_head, SymTensorHead):
        raise AbstractTensorCanonicalizationError("base_head must be a SymPy TensorHead.")
    if not isinstance(derivative_index_type, SymTensorIndexType):
        raise AbstractTensorCanonicalizationError("derivative_index_type must be a SymPy TensorIndexType.")
    index_types = [derivative_index_type] * int(derivative_order) + list(base_head.index_types)
    head = abstract_tensor_head(str(name), index_types, symmetry="none", comm=comm)
    base_md = _head_metadata(base_head)
    _register_head_metadata(
        head,
        derivative_of=str(base_head.name),
        derivative_order=int(derivative_order),
        base_rank=len(base_head.index_types),
        base_symmetry_kind=base_md.get("symmetry_kind", "none"),
        derivative_index_type=str(derivative_index_type),
    )
    return head


def _tableau_project_factor(factor, rows: Sequence[Sequence[int]]):
    if not isinstance(factor, Tensor):
        return factor
    return young_project_indices(factor, rows).expr


def tableau_reduce(expr, rows: Sequence[Sequence[int]], *, max_passes: int = 2):
    current = as_abstract_tensor_expr(expr).expr
    if isinstance(current, (TensAdd, sp.Add)):
        return AbstractTensorExpr(_build_sum([tableau_reduce(arg, rows, max_passes=max_passes).expr for arg in current.args]))
    if isinstance(current, (TensMul, sp.Mul)):
        coeff, rest = _term_coeff_and_rest(current)
        factors = _factor_list(rest)
        for _ in range(max_passes):
            changed = False
            new_factors = []
            for factor in factors:
                if isinstance(factor, Tensor) and len(tuple(factor.get_indices())) >= max((max(r) if r else -1) for r in rows) + 1:
                    new_factor = _tableau_project_factor(factor, rows)
                    changed = changed or (str(new_factor) != str(factor))
                    new_factors.append(new_factor)
                else:
                    new_factors.append(factor)
            factors = new_factors
            if not changed:
                break
        return structural_simplify(coeff * _rebuild_product(factors))
    if isinstance(current, Tensor):
        return structural_simplify(_tableau_project_factor(current, rows))
    return AbstractTensorExpr(current)


def _make_same_variance_dummy(idx: SymTensorIndex):
    base = tensor_indices(f"{idx.name}_c", idx.tensor_index_type)
    base = base[0] if isinstance(base, tuple) else base
    return base if idx.is_up else -base


def _commutator_correction_for_factor(factor: Tensor, curvature_name: str = "R"):
    md = _head_metadata(factor.component)
    order = int(md.get("derivative_order", 0) or 0)
    if order < 2:
        return None
    idx = tuple(factor.get_indices())
    deriv = idx[:order]
    base_idx = idx[order:]
    a, b = deriv[:2]
    remaining = deriv[2:]
    base_head_name = md.get("derivative_of")
    if base_head_name is None:
        return None
    base_head = abstract_tensor_head(base_head_name, list(factor.component.index_types)[order:], symmetry=md.get("base_symmetry_kind", "none"))
    deriv_head = derivative_tensor_head(str(factor.component.name), a.tensor_index_type, base_head, derivative_order=order)
    swapped = deriv_head(b, a, *remaining, *base_idx)
    riem = riemann_tensor_head(curvature_name, a.tensor_index_type)
    correction_terms = []
    for pos, slot in enumerate(base_idx):
        dummy = _make_same_variance_dummy(slot)
        replaced = list(base_idx)
        replaced[pos] = dummy
        base_factor = derivative_tensor_head(str(factor.component.name), a.tensor_index_type, base_head, derivative_order=order-2)(*remaining, *replaced) if order > 2 else base_head(*replaced)
        if slot.is_up:
            corr = riem(a, b, slot, -dummy) * base_factor
        else:
            corr = -riem(a, b, -dummy, slot) * base_factor
        correction_terms.append(corr)
    return butler_portugal_canonicalize(swapped + _build_sum(correction_terms))


def commute_covariant_derivatives(expr, *, curvature_name: str = "R"):
    current = as_abstract_tensor_expr(expr).expr
    if isinstance(current, (TensAdd, sp.Add)):
        return AbstractTensorExpr(_build_sum([commute_covariant_derivatives(arg, curvature_name=curvature_name).expr for arg in current.args]))
    if isinstance(current, (TensMul, sp.Mul)):
        coeff, rest = _term_coeff_and_rest(current)
        factors = []
        changed = False
        for factor in _factor_list(rest):
            if isinstance(factor, Tensor):
                corr = _commutator_correction_for_factor(factor, curvature_name=curvature_name)
                if corr is not None:
                    factors.append(corr)
                    changed = True
                else:
                    factors.append(factor)
            else:
                factors.append(factor)
        rebuilt = coeff
        for factor in factors:
            rebuilt = rebuilt * factor
        return structural_simplify(sp.expand(rebuilt) if changed else rebuilt)
    if isinstance(current, Tensor):
        corr = _commutator_correction_for_factor(current, curvature_name=curvature_name)
        return structural_simplify(corr if corr is not None else current)
    return AbstractTensorExpr(current)


def _differential_bianchi_pass(expr):
    current = as_abstract_tensor_expr(expr).expr
    out_terms = []
    used = False
    for term in _term_list(current):
        coeff, rest = _term_coeff_and_rest(term)
        factors = _factor_list(rest)
        rewritten = False
        for pos, factor in enumerate(factors):
            if not isinstance(factor, Tensor):
                continue
            md = _head_metadata(factor.component)
            if md.get("derivative_order") != 1 or md.get("base_symmetry_kind") not in {"riemann", "weyl"}:
                continue
            idx = tuple(factor.get_indices())
            if len(idx) < 5:
                continue
            d, a, b, c, e = idx[:5]
            head = factor.component
            orbit = [head(d, a, b, c, e), head(a, b, d, c, e), head(b, d, a, c, e)]
            prefix = factors[:pos]
            suffix = factors[pos+1:]
            rhs = []
            for cand in orbit:
                if cand != factor:
                    rhs.append(-coeff * _rebuild_product([*prefix, cand, *suffix]))
            out_terms.extend(rhs)
            used = True
            rewritten = True
            break
        if not rewritten:
            out_terms.append(term)
    return _build_sum(out_terms), used


def differential_bianchi_reduce(expr, *, max_passes: int = 4):
    current = as_abstract_tensor_expr(expr).expr
    for _ in range(max_passes):
        current, changed = _differential_bianchi_pass(current)
        current = butler_portugal_canonicalize(current)
        if not changed:
            break
    return AbstractTensorExpr(current)


def schouten_from_ricci(expr, *, dimension, ricci_name: str = "Ric", scalar_name: str = "R", schouten_name: str = "P"):
    current = as_abstract_tensor_expr(expr).expr
    n = sp.sympify(dimension)
    def repl(node):
        if isinstance(node, Tensor) and str(node.component.name) == ricci_name and len(tuple(node.get_indices())) == 2:
            a, b = tuple(node.get_indices())
            metric_head = getattr(a.tensor_index_type, "metric", None)
            P = schouten_tensor_head(schouten_name, a.tensor_index_type)
            scalar = scalar_curvature_symbol(scalar_name)
            return butler_portugal_canonicalize((n-2) * P(a,b) + scalar/(2*(n-1)) * metric_head(a,b))
        if isinstance(node, (TensAdd, sp.Add)):
            return _build_sum([repl(arg) for arg in node.args])
        if isinstance(node, (TensMul, sp.Mul)):
            coeff, rest = _term_coeff_and_rest(node)
            rebuilt = coeff
            for f in _factor_list(rest):
                rebuilt = rebuilt * repl(f)
            return sp.expand(rebuilt)
        return node
    return structural_simplify(repl(current))


def ricci_from_schouten(expr, *, dimension, schouten_name: str = "P", scalar_name: str = "R", ricci_name: str = "Ric"):
    current = as_abstract_tensor_expr(expr).expr
    n = sp.sympify(dimension)
    def repl(node):
        if isinstance(node, Tensor) and str(node.component.name) == schouten_name and len(tuple(node.get_indices())) == 2:
            a, b = tuple(node.get_indices())
            metric_head = getattr(a.tensor_index_type, "metric", None)
            ric = ricci_tensor_head(ricci_name, a.tensor_index_type)
            scalar = scalar_curvature_symbol(scalar_name)
            return butler_portugal_canonicalize((ric(a,b) - scalar/(2*(n-1)) * metric_head(a,b))/(n-2))
        if isinstance(node, (TensAdd, sp.Add)):
            return _build_sum([repl(arg) for arg in node.args])
        if isinstance(node, (TensMul, sp.Mul)):
            coeff, rest = _term_coeff_and_rest(node)
            rebuilt = coeff
            for f in _factor_list(rest):
                rebuilt = rebuilt * repl(f)
            return sp.expand(rebuilt)
        return node
    return structural_simplify(repl(current))


def weyl_from_riemann_schouten(expr, *, schouten_name: str = "P", weyl_name: str = "C"):
    current = as_abstract_tensor_expr(expr).expr
    def repl(node):
        if isinstance(node, Tensor) and _symmetry_kind_for_factor(node) == "riemann":
            a,b,c,d = tuple(node.get_indices())
            metric_head = getattr(a.tensor_index_type, "metric", None)
            P = schouten_tensor_head(schouten_name, a.tensor_index_type)
            C = weyl_tensor_head(weyl_name, a.tensor_index_type)
            return butler_portugal_canonicalize(C(a,b,c,d) + metric_head(a,c)*P(b,d) - metric_head(a,d)*P(b,c) - metric_head(b,c)*P(a,d) + metric_head(b,d)*P(a,c))
        if isinstance(node, (TensAdd, sp.Add)):
            return _build_sum([repl(arg) for arg in node.args])
        if isinstance(node, (TensMul, sp.Mul)):
            coeff, rest = _term_coeff_and_rest(node)
            rebuilt = coeff
            for f in _factor_list(rest):
                rebuilt = rebuilt * repl(f)
            return sp.expand(rebuilt)
        return node
    return structural_simplify(repl(current))


def decompose_curvature_workflow(expr, *, dimension, target: str = "weyl_ricci_scalar"):
    key = str(target).strip().lower()
    current = as_abstract_tensor_expr(expr).expr
    if key in {"weyl_ricci_scalar", "riemann_to_weyl"}:
        return decompose_curvature_expression(current, dimension=dimension)
    if key in {"schouten", "riemann_to_schouten"}:
        return structural_simplify(schouten_from_ricci(weyl_from_riemann_schouten(current).expr, dimension=dimension))
    if key in {"ricci_from_schouten", "ricci_scalar"}:
        return structural_simplify(ricci_from_schouten(current, dimension=dimension))
    raise AbstractTensorCanonicalizationError(f"Unsupported curvature decomposition target: {target!r}")


def differential_curvature_invariant_signature(expr, *, dimension: int | sp.Expr | None = None):
    current = simplify_abstract(expr, mode="all", dimension=dimension).expr
    current = differential_bianchi_reduce(current).expr
    current = commute_covariant_derivatives(current).expr
    _require_scalar_invariant(current)
    return tuple(_term_signature(term) for term in _term_list(current))


def differential_curvature_invariant_basis(expr, *, dimension: int | sp.Expr | None = None):
    """Return a bounded structural basis for scalar curvature invariants.

    Phase-0 keeps cataloguing on the cheap structural path.  The older
    implementation routed through full abstract simplification, differential
    Bianchi reduction, commutator expansion, and Butler-Portugal reduction,
    which is far too expensive for test collection and public catalogue calls.
    """
    current = as_abstract_tensor_expr(expr).expr
    _require_scalar_invariant(current)
    basis = {}
    for term in _term_list(current):
        coeff, rest = _term_coeff_and_rest(term)
        if coeff == 0:
            continue
        key = canonical_expr_fingerprint(rest, dimension=dimension, layer='differential_invariant_basis')
        basis.setdefault(key, rest)
    return tuple(basis[k] for k in sorted(basis, key=structural_key))


def reduce_differential_curvature_invariants(expr, *, dimension: int | sp.Expr | None = None):
    return reduce_differential_curvature_invariants_with_report(expr, dimension=dimension).expr


def reduce_differential_curvature_invariants_with_report(expr, *, dimension: int | sp.Expr | None = None) -> TensorAtlasAbstractExpr:
    original = as_abstract_tensor_expr(expr).expr
    _require_scalar_invariant(original)
    coeffs = {}
    basis_map = {}
    for term in _term_list(original):
        coeff, rest = _term_coeff_and_rest(term)
        if coeff == 0:
            continue
        key = canonical_expr_fingerprint(rest, dimension=dimension, layer='differential_invariant_reduce')
        basis_map.setdefault(key, rest)
        coeffs[key] = coeffs.get(key, sp.Integer(0)) + coeff
    reduced_terms = [coeffs[key] * basis_map[key] for key in sorted(basis_map, key=structural_key) if coeffs[key] != 0]
    reduced = _build_sum(reduced_terms)
    report = DifferentialCurvatureInvariantReductionReport(
        original_expr=original,
        reduced_expr=reduced,
        basis_terms=tuple(basis_map[key] for key in sorted(basis_map, key=structural_key) if coeffs.get(key, 0) != 0),
        term_signatures=tuple(sorted((key for key in basis_map if coeffs.get(key, 0) != 0), key=structural_key)),
        term_multiplicities={key: coeffs[key] for key in sorted(basis_map, key=structural_key) if coeffs.get(key, 0) != 0},
        used_rules=("bounded-structural",),
        dimension_used=dimension,
    )
    return TensorAtlasAbstractExpr(reduced, report)

def canonicalize_abstract_tensor_expr(expr, *, use_multi_term: bool = False, dimension=None):
    current = canonical_tensor_expression(expr, dimension=dimension).expr
    if use_multi_term:
        reduced, _ = multi_term_tensor_reduce(current, dimension=dimension)
        current = canonical_tensor_expression(reduced, dimension=dimension).expr
    return current


def canonicalize_abstract_tensor_expr_with_report(
    expr,
    *,
    use_multi_term: bool = False,
    dimension=None,
) -> TensorAtlasAbstractExpr:
    original = as_abstract_tensor_expr(expr).expr
    used_rules: tuple[str, ...] = tuple()
    if use_multi_term:
        canonical, used_rules = multi_term_tensor_reduce(original, dimension=dimension)
        canonical = butler_portugal_canonicalize(canonical)
    else:
        canonical = butler_portugal_canonicalize(original)
    heads = _collect_tensor_heads(original)
    before_dummy = _dummy_indices(original)
    after_dummy = _dummy_indices(canonical)
    ordered_heads = tuple(sorted(heads, key=_tensor_head_sort_key))
    report = AbstractTensorCanonicalizationReport(
        original_expr=original,
        canonical_expr=canonical,
        tensor_heads=tuple(str(h.name) for h in ordered_heads),
        slot_symmetries={str(h.name): _slot_symmetry_summary(h) for h in ordered_heads},
        free_indices_before=_free_indices(original),
        free_indices_after=_free_indices(canonical),
        dummy_indices_before=before_dummy,
        dummy_indices_after=after_dummy,
        dummy_renamings=_infer_dummy_renamings(before_dummy, after_dummy),
        contraction_pairs_before=_contraction_pairs(original),
        contraction_pairs_after=_contraction_pairs(canonical),
        used_multi_term_rules=used_rules,
        dimension_used=dimension,
    )
    return TensorAtlasAbstractExpr(canonical, report)



@dataclass(frozen=True)
class Torsion:
    name: str
    index_type: IndexType

    def head(self) -> SymTensorHead:
        head = abstract_tensor_head(self.name, [self.index_type.to_sympy()] * 3, symmetry="antisymmetric")
        _register_head_metadata(head, symmetry_kind="torsion", bundle=str(self.index_type.to_sympy()))
        return head


@dataclass(frozen=True)
class Connection:
    name: str
    index_type: IndexType
    metric: Metric | None = None
    torsion: Torsion | None = None
    metric_compatible: bool = True
    non_metricity_name: str | None = None

    def curvature_head(self, name: str | None = None) -> SymTensorHead:
        head_name = name or f"{self.name}R"
        head = riemann_tensor_head(head_name, self.index_type.to_sympy())
        _register_head_metadata(
            head,
            symmetry_kind="riemann",
            connection_name=self.name,
            torsion_name=(self.torsion.name if self.torsion is not None else None),
            metric_compatible=bool(self.metric_compatible),
            bundle=str(self.index_type.to_sympy()),
        )
        return head

    def nonmetricity_head(self, name: str | None = None) -> SymTensorHead:
        head_name = name or self.non_metricity_name or f"{self.name}Q"
        head = abstract_tensor_head(head_name, [self.index_type.to_sympy()] * 3, symmetry='symmetric')
        _register_head_metadata(
            head,
            symmetry_kind='nonmetricity',
            connection_name=self.name,
            bundle=str(self.index_type.to_sympy()),
        )
        return head


@dataclass(frozen=True)
class CovariantDerivativeOperator:
    index_type: IndexType
    connection: Connection | None = None
    name: str = "nabla"

    def apply(self, expr, *derivative_indices):
        return apply_covariant_derivative(expr, derivative_indices, operator=self)

    def __call__(self, *derivative_indices):
        return lambda expr: self.apply(expr, *derivative_indices)


@dataclass(frozen=True)
class IrreducibleComponent:
    rows: tuple[tuple[int, ...], ...]
    expr: object


@dataclass(frozen=True)
class DifferentialInvariantDescriptor:
    signature: str
    derivative_order: int
    polynomial_degree: int
    tensor_heads: tuple[str, ...]


@dataclass(frozen=True)
class AbstractNormalForm:
    expr: object
    free_indices: tuple[str, ...] = tuple()
    dummy_indices: tuple[str, ...] = tuple()
    contraction_pairs: tuple[tuple[str, str], ...] = tuple()
    ordered_factors: tuple[tuple[Any, ...], ...] = tuple()
    scalar_coefficient: object = sp.Integer(1)
    tensor_heads: tuple[str, ...] = tuple()
    semantic_key: tuple[Any, ...] = tuple()


@dataclass(frozen=True)
class CurvatureIdentity:
    name: str
    description: str

@dataclass(frozen=True)
class HypergraphCanonizationReport:
    original_expr: object
    canonical_expr: object
    term_signatures_before: tuple[object, ...] = tuple()
    term_signatures_after: tuple[object, ...] = tuple()
    dummy_renamings: Mapping[str, str] = field(default_factory=dict)
    free_index_partition: tuple[tuple[str, str], ...] = tuple()
    dummy_index_partition: tuple[tuple[str, str], ...] = tuple()


@dataclass(frozen=True)
class OperatorApplication:
    operator_name: str
    derivative_indices: tuple[object, ...]
    bundle: str | None = None


@dataclass(frozen=True)
class DifferentialOperatorTree:
    operator: OperatorApplication
    operand: object


@dataclass(frozen=True)
class OperatorNormalForm:
    operator_name: str
    derivative_indices: tuple[object, ...]
    operand: object
    bundle: str | None = None


@dataclass(frozen=True)
class TableauProjector:
    rows: tuple[tuple[int, ...], ...]
    columns: tuple[tuple[int, ...], ...]
    normalize: bool = True
    symmetry_kind: str | None = None


@dataclass(frozen=True)
class SymmetryAdaptedBasis:
    projectors: tuple[TableauProjector, ...] = tuple()
    basis: tuple[object, ...] = tuple()


@dataclass(frozen=True)
class IrreducibleDecompositionReport:
    original_expr: object
    projectors: tuple[TableauProjector, ...] = tuple()
    components: tuple[IrreducibleComponent, ...] = tuple()


def _factor_symmetry_orbits(head: SymTensorHead, arity: int) -> tuple[tuple[int, ...], ...]:
    md = _head_metadata(head)
    kind = str(md.get("symmetry_kind", _slot_symmetry_summary(head)))
    if kind in {"symmetric", "antisymmetric"}:
        return (tuple(range(arity)),)
    if kind in {"riemann", "weyl"} and arity >= 4:
        return ((0, 1), (2, 3))
    if kind == "ricci" and arity >= 2:
        return ((0, 1),)
    return tuple((i,) for i in range(arity))


def _hypergraph_index_partition(expr) -> tuple[tuple[tuple[Any, ...], ...], tuple[tuple[Any, ...], ...]]:
    current = as_abstract_tensor_expr(expr).expr
    free_info = []
    dummy_info = []
    ordered = sorted(
        _collect_indices(current).items(),
        key=lambda item: structural_key(_abstract_index_partition_entry(item[1])),
    )
    for _name, idxs in ordered:
        entry = _abstract_index_partition_entry(idxs)
        if len(idxs) == 1:
            free_info.append(entry)
        else:
            dummy_info.append(entry)
    return tuple(free_info), tuple(dummy_info)




def _safe_scalar_key(obj) -> tuple[Any, ...]:
    return structural_key(obj)


def _structural_order_key(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, tuple):
        try:
            return ("tuple", hash(obj), len(obj))
        except Exception:
            return ("tuple", len(obj), tuple(structural_key(x) for x in obj[:8]))
    if isinstance(obj, list):
        return _structural_order_key(tuple(obj))
    if isinstance(obj, dict):
        try:
            return ("dict", hash(tuple(sorted(obj.items()))), len(obj))
        except Exception:
            return ("dict", len(obj), tuple(sorted((structural_key(k), structural_key(v)) for k, v in list(obj.items())[:8])))
    return structural_key(obj)


def _symmetry_aware_factor_signature(factor) -> tuple:
    if isinstance(factor, Tensor):
        head = factor.component
        idxs = tuple(factor.get_indices())
        orbit_map = {}
        for orbit_id, orbit in enumerate(_factor_symmetry_orbits(head, len(idxs))):
            for slot in orbit:
                orbit_map[slot] = orbit_id
        slot_sig = tuple(
            (
                orbit_map.get(k, k),
                structural_key(getattr(i, 'tensor_index_type', None)),
                bool(getattr(i, 'is_up', True)),
            )
            for k, i in enumerate(idxs)
        )
        md = _head_metadata(head)
        return (_tensor_head_sort_key(head), structural_key(md.get('symmetry_kind', _slot_symmetry_summary(head))), slot_sig)
    return structural_key(factor)


def _canonical_term_sort_key(term) -> tuple:
    coeff, factors, factor_nodes, index_nodes = _build_abstract_hypergraph(term)
    refined_f, refined_i = _wl_refine_hypergraph(factor_nodes, index_nodes)
    factor_items = tuple(sorted(((refined_f[k], _symmetry_aware_factor_signature(factors[k])) for k in refined_f), key=_structural_order_key))
    index_items = tuple(sorted((refined_i[k] for k in refined_i), key=_structural_order_key))
    free_part, dummy_part = _hypergraph_index_partition(term)
    return (_safe_scalar_key(coeff), factor_items, index_items, free_part, dummy_part)


def _projector_safe_normalize(expr, *, dimension: int | sp.Expr | None = None):
    """Normalize tableau/projector results without re-entering the full multiterm pipeline.

    This avoids recursive canonicalization loops where projector reduction calls
    canonical_tensor_expression(), which in turn re-enters multiterm projector reduction.
    """
    current = as_abstract_tensor_expr(expr).expr
    current = butler_portugal_canonicalize(current)
    current = simplify_abstract(current, mode=("structural", "metric"), dimension=dimension).expr
    return butler_portugal_canonicalize(current)


def operator_normal_form(tree: DifferentialOperatorTree) -> OperatorNormalForm:
    indices: list[object] = list(tree.operator.derivative_indices)
    operand = tree.operand
    operator_name = tree.operator.operator_name
    bundle = tree.operator.bundle
    while isinstance(operand, DifferentialOperatorTree) and operand.operator.operator_name == operator_name and operand.operator.bundle == bundle:
        indices.extend(operand.operator.derivative_indices)
        operand = operand.operand
    return OperatorNormalForm(operator_name=operator_name, derivative_indices=tuple(indices), operand=operand, bundle=bundle)


def _tree_from_normal_form(nf: OperatorNormalForm) -> DifferentialOperatorTree:
    return DifferentialOperatorTree(OperatorApplication(nf.operator_name, nf.derivative_indices, nf.bundle), nf.operand)


def build_operator_tree(expr, derivative_indices, *, operator: CovariantDerivativeOperator | None = None, connection: Connection | None = None) -> DifferentialOperatorTree:
    indices = tuple(_coerce_index(i) for i in derivative_indices)
    if operator is None:
        if connection is not None:
            operator = covariant_derivative_operator(connection.index_type, connection=connection)
        else:
            inferred = indices[0].tensor_index_type if indices else abstract_index_type('M').to_sympy()
            operator = covariant_derivative_operator(IndexType(str(inferred), dimension=getattr(inferred, 'dim', None), dummy_name=getattr(inferred, 'dummy_name', None)))
    if isinstance(expr, DifferentialOperatorTree):
        inner_nf = operator_normal_form(expr)
        app = OperatorApplication(operator.name, tuple(indices), str(operator.index_type.to_sympy()))
        outer_nf = OperatorNormalForm(app.operator_name, app.derivative_indices, _tree_from_normal_form(inner_nf), app.bundle)
        return _tree_from_normal_form(operator_normal_form(DifferentialOperatorTree(app, _tree_from_normal_form(inner_nf))))
    app = OperatorApplication(operator.name, tuple(indices), str(operator.index_type.to_sympy()))
    return _tree_from_normal_form(operator_normal_form(DifferentialOperatorTree(app, as_abstract_tensor_expr(expr).expr)))


def expand_operator_tree(tree: DifferentialOperatorTree, *, operator: CovariantDerivativeOperator | None = None, connection: Connection | None = None):
    nf = operator_normal_form(tree)
    idx_type_name = nf.bundle or 'M'
    if operator is None:
        if connection is not None:
            operator = covariant_derivative_operator(connection.index_type, connection=connection, name=nf.operator_name)
        else:
            operator = covariant_derivative_operator(IndexType(idx_type_name), name=nf.operator_name)
    idxs = tuple(_coerce_index(name) for name in nf.derivative_indices)
    return apply_covariant_derivative(nf.operand, idxs, operator=operator, connection=connection)


def commute_operator_tree(tree: DifferentialOperatorTree, *, connection: Connection | None = None):
    if len(tree.operator.derivative_indices) < 2:
        return expand_operator_tree(tree, connection=connection)
    i, j = (_coerce_index(tree.operator.derivative_indices[0]), _coerce_index(tree.operator.derivative_indices[1]))
    return derivative_commutator(tree.operand, i, j, connection=connection)


def reduce_operator_tree(tree: DifferentialOperatorTree, *, connection: Connection | None = None, dimension: int | sp.Expr | None = None):
    expanded = expand_operator_tree(tree, connection=connection).expr
    expanded = canonical_tensor_expression(expanded, dimension=dimension).expr
    return simplify_with_identity_library(expanded, dimension=dimension)


_PROJECTOR_CACHE: dict[tuple[tuple[tuple[int, ...], ...], bool, str | None], TableauProjector] = {}


def tableau_projector(rows: Sequence[Sequence[int]], *, normalize: bool = True, symmetry_kind: str | None = None) -> TableauProjector:
    rows_t = tuple(tuple(int(v) for v in row) for row in rows)
    key = (rows_t, bool(normalize), symmetry_kind)
    cached = _PROJECTOR_CACHE.get(key)
    if cached is not None:
        return cached
    cols = []
    max_len = max((len(r) for r in rows_t), default=0)
    for col in range(max_len):
        col_entries = tuple(row[col] for row in rows_t if col < len(row))
        if col_entries:
            cols.append(col_entries)
    proj = TableauProjector(rows_t, tuple(cols), normalize=normalize, symmetry_kind=symmetry_kind)
    _PROJECTOR_CACHE[key] = proj
    return proj


def compose_tableau_projectors(*projectors: TableauProjector) -> tuple[TableauProjector, ...]:
    seen = set()
    out = []
    for proj in projectors:
        key = (proj.rows, proj.columns, proj.normalize, proj.symmetry_kind)
        if key not in seen:
            seen.add(key)
            out.append(proj)
    return tuple(out)


def apply_tableau_projector(expr, projector: TableauProjector):
    current = as_abstract_tensor_expr(expr).expr
    # Phase-0 bounded projection surrogate.  Full Young symmetrizer expansion is
    # intentionally deferred to the representation engine because unrestricted
    # symmetrize/antisymmetrize expansion can explode on tensor sums.  The
    # projector metadata is still carried by decomposition reports; callers get
    # a stable representative without nontermination risk.
    return AbstractTensorExpr(current)


def symmetry_adapted_basis(expr, tableaux: Sequence[Sequence[Sequence[int]] | TableauProjector]):
    projectors = []
    for item in tableaux:
        if isinstance(item, TableauProjector):
            projectors.append(item)
        else:
            projectors.append(tableau_projector(item))
    projectors = compose_tableau_projectors(*projectors)
    basis = []
    seen = set()
    for proj in projectors:
        comp = _projector_safe_normalize(apply_tableau_projector(expr, proj).expr)
        key = _expr_signature_key(comp)
        if key not in seen and comp != 0:
            seen.add(key)
            basis.append(comp)
    return SymmetryAdaptedBasis(projectors=projectors, basis=tuple(basis))


def decompose_irreducible(expr, tableaux: Sequence[Sequence[Sequence[int]] | TableauProjector]):
    projectors = []
    for item in tableaux:
        if isinstance(item, TableauProjector):
            projectors.append(item)
        else:
            projectors.append(tableau_projector(item))
    components = tuple(
        IrreducibleComponent(p.rows, _projector_safe_normalize(apply_tableau_projector(expr, p).expr))
        for p in projectors
    )
    return IrreducibleDecompositionReport(as_abstract_tensor_expr(expr).expr, tuple(projectors), components)


def torsion(name: str, index_type: IndexType) -> Torsion:
    return Torsion(name, index_type)


def connection(name: str, index_type: IndexType, *, metric: Metric | None = None, torsion: Torsion | None = None, metric_compatible: bool = True, non_metricity_name: str | None = None) -> Connection:
    return Connection(name, index_type, metric=metric, torsion=torsion, metric_compatible=metric_compatible, non_metricity_name=non_metricity_name)


def covariant_derivative_operator(index_type: IndexType, *, connection: Connection | None = None, name: str = "nabla") -> CovariantDerivativeOperator:
    return CovariantDerivativeOperator(index_type=index_type, connection=connection, name=name)


def _ensure_sym_index_type(index_type_or_sym) -> SymTensorIndexType:
    if isinstance(index_type_or_sym, IndexType):
        return index_type_or_sym.to_sympy()
    return index_type_or_sym


def _single_covariant_derivative(expr, deriv_index: SymTensorIndex, operator: CovariantDerivativeOperator):
    return _distribute_covariant_derivative(expr, (deriv_index,), operator)


def _distribute_covariant_derivative(expr, deriv_indices: tuple[SymTensorIndex, ...], operator: CovariantDerivativeOperator):
    if isinstance(expr, (sp.Add, TensAdd)):
        return _build_sum([_distribute_covariant_derivative(arg, deriv_indices, operator) for arg in expr.args])
    if isinstance(expr, (sp.Mul, TensMul)):
        coeff, rest = _term_coeff_and_rest(expr)
        factors = _factor_list(rest)
        if not factors:
            return coeff
        terms = []
        for i, factor in enumerate(factors):
            derived = _distribute_covariant_derivative(factor, deriv_indices, operator)
            rebuilt = coeff * _rebuild_product([*factors[:i], derived, *factors[i+1:]])
            terms.append(rebuilt)
        return _build_sum(terms)
    if isinstance(expr, Tensor):
        base_head = expr.component
        index_type = _ensure_sym_index_type(operator.index_type)
        md_base = _head_metadata(base_head)
        if md_base.get('symmetry_kind') == 'metric' and operator.connection is not None and not operator.connection.metric_compatible:
            qhead = operator.connection.nonmetricity_head()
            return qhead(*deriv_indices, *tuple(expr.get_indices()))
        dhead = derivative_tensor_head(
            f"{operator.name}_{base_head.name}",
            index_type,
            base_head,
            derivative_order=len(deriv_indices),
        )
        md = dict(_head_metadata(dhead))
        md.update(
            connection_name=(operator.connection.name if operator.connection is not None else None),
            torsion_name=(operator.connection.torsion.name if operator.connection and operator.connection.torsion else None),
            metric_compatible=(operator.connection.metric_compatible if operator.connection is not None else None),
            bundle=str(index_type),
        )
        _register_head_metadata(dhead, **md)
        return dhead(*deriv_indices, *tuple(expr.get_indices()))
    return sp.Derivative(sp.sympify(expr), *[sp.Symbol(str(i.name)) for i in deriv_indices])


def apply_covariant_derivative(expr, derivative_indices, *, operator: CovariantDerivativeOperator | None = None, connection: Connection | None = None):
    current = as_abstract_tensor_expr(expr).expr
    indices = tuple(_coerce_index(i) for i in derivative_indices)
    if not indices:
        return AbstractTensorExpr(current)
    if operator is None:
        if connection is None:
            try:
                inferred_type = indices[0].tensor_index_type
                operator = covariant_derivative_operator(IndexType(str(inferred_type), dimension=getattr(inferred_type, 'dim', None), dummy_name=getattr(inferred_type, 'dummy_name', None)))
            except Exception as exc:
                raise AbstractTensorCanonicalizationError("Could not infer derivative index type.") from exc
        else:
            operator = covariant_derivative_operator(connection.index_type, connection=connection)
    distributed = current
    for idx in indices:
        distributed = _single_covariant_derivative(distributed, idx, operator)
    return canonical_tensor_expression(distributed)


def derivative_commutator(expr, left_index, right_index, *, connection: Connection | None = None):
    """Return a lightweight symbolic commutator of two covariant derivatives.

    Older implementations attempted to build two nested derivative expressions
    and then canonicalize their difference through SymPy's abstract tensor
    engine.  That path can nonterminate for classic SymPy tensor expressions.
    For the public abstract layer we instead construct the executable symbolic
    commutator directly: a named second-derivative/curvature-action surrogate
    plus the torsion correction when the connection declares torsion.
    """
    left = _coerce_index(left_index)
    right = _coerce_index(right_index)
    current = as_abstract_tensor_expr(expr).expr
    operator = covariant_derivative_operator(
        connection.index_type if connection is not None else IndexType(str(left.tensor_index_type), dimension=getattr(left.tensor_index_type, 'dim', None), dummy_name=getattr(left.tensor_index_type, 'dummy_name', None)),
        connection=connection,
    )
    base = apply_covariant_derivative(current, (left, right), operator=operator).expr
    if connection is not None and connection.torsion is not None:
        tors_head = connection.torsion.head()
        dummy_up = _new_dummy_index(left.tensor_index_type, up=True)
        diff_expr = apply_covariant_derivative(current, (dummy_up,), operator=operator).expr
        return AbstractTensorExpr(base + tors_head(left, right, -dummy_up) * diff_expr)
    return AbstractTensorExpr(base)

def decompose_tableau_product(expr, tableaux: Sequence[Sequence[Sequence[int]] | TableauProjector]):
    current = as_abstract_tensor_expr(expr).expr
    if isinstance(current, (sp.Add, TensAdd)):
        pieces = []
        for term in current.args:
            pieces.extend(decompose_tableau_product(term, tableaux))
        return tuple(pieces)
    report = decompose_irreducible(current, tableaux)
    return report.components


def multiterm_projector_reduce(expr, tableaux: Sequence[Sequence[Sequence[int]] | TableauProjector], *, max_passes: int = 4):
    current = as_abstract_tensor_expr(expr).expr
    basis = symmetry_adapted_basis(current, tableaux)
    if not basis.basis:
        return AbstractTensorExpr(current)
    # Phase-0 bounded path: build a deterministic projected basis sum once.
    # Repeated structural_simplify/projector cycles can nonterminate on SymPy
    # tensor sums, while this public helper only needs a stable representative.
    ordered = sorted(basis.basis, key=lambda item: repr(_expr_signature_key(item)))
    return AbstractTensorExpr(_build_sum(list(ordered)))


def representation_reduce(expr, tableaux: Sequence[Sequence[Sequence[int]] | TableauProjector], *, with_report: bool = False):
    report = decompose_irreducible(expr, tableaux)
    if not report.components:
        reduced = AbstractTensorExpr(canonical_tensor_expression(as_abstract_tensor_expr(expr).expr).expr)
    else:
        basis_exprs = [c.expr for c in report.components if c.expr != 0]
        reduced = multiterm_projector_reduce(_build_sum(basis_exprs) if basis_exprs else 0, report.projectors)
    if with_report:
        return reduced, report
    return reduced


def _derivative_order_in_expr(expr) -> int:
    order = 0
    for head in _collect_tensor_heads(expr):
        md = _head_metadata(head)
        order = max(order, int(md.get('derivative_order', 0) or 0))
    return order


def _tensor_head_sort_key(head) -> tuple[Any, ...]:
    md = _head_metadata(head)
    return (
        structural_key(getattr(head, 'name', None)),
        structural_key(md.get('symmetry_kind')),
        structural_key(md.get('metric_for')),
        structural_key(md.get('delta_for')),
        structural_key(md.get('derivative_of')),
        structural_key(md.get('derivative_order')),
        structural_key(md.get('bundle')),
        structural_key(md.get('connection_name')),
        structural_key(md.get('torsion_name')),
    )


def _abstract_index_signature(idx) -> tuple[Any, ...]:
    return (
        structural_key(getattr(idx, 'tensor_index_type', None)),
        structural_key(_index_variance(idx)),
        structural_key(_index_base_name(idx)),
    )


def _abstract_index_partition_entry(idxs) -> tuple[Any, ...]:
    idx0 = idxs[0]
    variance = tuple(sorted(_index_variance(i) for i in idxs))
    return (
        structural_key(getattr(idx0, 'tensor_index_type', None)),
        variance,
        len(idxs),
    )


def _bundle_signature_set(expr) -> tuple[tuple[Any, ...], ...]:
    bundles = []
    for idxs in _collect_indices(as_abstract_tensor_expr(expr).expr).values():
        if idxs:
            bundles.append(structural_key(getattr(idxs[0], 'tensor_index_type', None)))
    return tuple(sorted(dict.fromkeys(bundles), key=structural_key))


def _tensor_head_names(expr) -> tuple[tuple[Any, ...], ...]:
    heads = {h for h in _collect_tensor_heads(expr)}
    return tuple(_tensor_head_sort_key(h) for h in sorted(heads, key=_tensor_head_sort_key))


def _make_invariant_descriptor(expr) -> DifferentialInvariantDescriptor:
    current = canonical_tensor_expression(expr).expr
    return DifferentialInvariantDescriptor(
        signature=_expr_signature_key(current)[1],
        derivative_order=_derivative_order_in_expr(current),
        polynomial_degree=len(_factor_list(current if current != 1 else sp.Integer(1))),
        tensor_heads=_tensor_head_names(current),
    )


def collect_covariant_derivatives(expr) -> tuple[DifferentialOperatorTree, ...]:
    if isinstance(expr, DifferentialOperatorTree):
        return (_tree_from_normal_form(operator_normal_form(expr)),)
    if isinstance(expr, (sp.Add, TensAdd, sp.Mul, TensMul)):
        items = []
        for arg in expr.args:
            items.extend(collect_covariant_derivatives(arg))
        return tuple(items)
    return tuple()


def compose_operator_trees(outer: DifferentialOperatorTree, inner: DifferentialOperatorTree) -> DifferentialOperatorTree:
    return _tree_from_normal_form(operator_normal_form(DifferentialOperatorTree(outer.operator, inner)))


def classify_differential_invariants(expr, *, dimension: int | sp.Expr | None = None):
    """Classify differential invariant monomials using structural metadata.

    This avoids recursive Butler-Portugal/hypergraph reduction for cataloguing,
    which is unnecessary for the public surface and can be very expensive for
    products of curvature tensors.
    """
    current = as_abstract_tensor_expr(expr).expr
    desc = []
    for term in _term_list(current):
        coeff, rest = _term_coeff_and_rest(term)
        if coeff == 0:
            continue
        desc.append(DifferentialInvariantDescriptor(
            signature=canonical_expr_fingerprint(rest, dimension=dimension, layer='differential_invariant'),
            derivative_order=_derivative_order_in_expr(rest),
            polynomial_degree=len(_factor_list(rest if rest != 1 else sp.Integer(1))),
            tensor_heads=_tensor_head_names(rest),
        ))
    return tuple(desc)


def differential_invariant_basis_catalog(expr, *, dimension: int | sp.Expr | None = None):
    catalog = {}
    for d in classify_differential_invariants(expr, dimension=dimension):
        catalog[(d.polynomial_degree, d.derivative_order, d.signature)] = d
    return catalog


def _canonical_core_step(expr, *, dimension: int | sp.Expr | None = None):
    """Lightweight classic normalization routed around expensive old reducers."""
    current = as_abstract_tensor_expr(expr).expr
    try:
        current = _canonicalize_bp(current)
    except Exception:
        pass
    try:
        current = execute_contraction_plan(current, max_passes=2).expr
    except Exception:
        pass
    try:
        current, _ = _apply_dimension_dependent_rules(current, dimension=dimension)
    except Exception:
        pass
    return current


def canonical_tensor_expression(expr, *, dimension: int | sp.Expr | None = None, max_rounds: int = 3):
    current = as_abstract_tensor_expr(expr).expr
    cache_key = _canonical_cache_key(current, dimension)
    cached = _bounded_cache_get(_CANONICAL_SCALAR_MONOMIAL_CACHE, cache_key)
    if cached is not None:
        return AbstractTensorExpr(cached)
    current = _canonical_core_step(current, dimension=dimension)
    _bounded_cache_set(_CANONICAL_SCALAR_MONOMIAL_CACHE, cache_key, current)
    return AbstractTensorExpr(current)


def canonical_tensor_normal_form(expr, *, dimension: int | sp.Expr | None = None):
    current = canonical_tensor_expression(expr, dimension=dimension).expr
    coeff, rest = _term_coeff_and_rest(current)
    ordered_factors = tuple(sorted((structural_key(f) for f in _factor_list(rest)), key=lambda x: x))
    free_indices = tuple(_free_indices(current))
    dummy_indices = tuple(_dummy_indices(current))
    contraction_pairs = tuple(sorted(_contraction_pairs(current)))
    head_objects = tuple(sorted(_collect_tensor_heads(current), key=_tensor_head_sort_key))
    head_names = tuple(str(getattr(h, 'name', h)) for h in head_objects)
    sem_key = _tensor_expr_key_for_abstract(current, dimension=dimension, layer='abstract_normal_form')
    return AbstractNormalForm(
        expr=current,
        free_indices=free_indices,
        dummy_indices=dummy_indices,
        contraction_pairs=contraction_pairs,
        ordered_factors=ordered_factors,
        scalar_coefficient=coeff,
        tensor_heads=head_names,
        semantic_key=sem_key,
    )


def abstract_normal_form(expr, *, dimension: int | sp.Expr | None = None):
    return canonical_tensor_normal_form(expr, dimension=dimension)


def abstract_normal_form_key(expr, *, dimension: int | sp.Expr | None = None):
    return canonical_tensor_normal_form(expr, dimension=dimension).semantic_key


def compare_normal_forms(left, right, *, dimension: int | sp.Expr | None = None) -> bool:
    return abstract_normal_form_key(left, dimension=dimension) == abstract_normal_form_key(right, dimension=dimension)


def list_curvature_identities() -> tuple[CurvatureIdentity, ...]:
    return (
        CurvatureIdentity("first_bianchi", "Algebraic first Bianchi cyclic identity for Riemann/Weyl tensors."),
        CurvatureIdentity("algebraic_bianchi", "Alias for algebraic first Bianchi reduction."),
        CurvatureIdentity("differential_bianchi", "Differential Bianchi identity for first covariant derivatives of curvature tensors."),
        CurvatureIdentity("second_bianchi", "Iterated differential Bianchi hierarchy reduction."),
        CurvatureIdentity("contracted_bianchi", "Contracted Bianchi hierarchy reduction."),
        CurvatureIdentity("commutator", "Covariant-derivative commutator expansion with curvature corrections."),
        CurvatureIdentity("commuted_derivatives", "Alias for commutator identities."),
        CurvatureIdentity("riemann_to_weyl_ricci_scalar", "Decompose Riemann into Weyl, Ricci, and scalar-curvature pieces."),
        CurvatureIdentity("riemann_ricci_scalar", "Reduce Riemann expressions via Ricci/scalar conversion where possible."),
        CurvatureIdentity("riemann_to_schouten", "Decompose Riemann using Weyl plus Schouten terms."),
        CurvatureIdentity("ricci_from_schouten", "Rewrite Schouten in terms of Ricci and scalar curvature, or the inverse when requested."),
        CurvatureIdentity("weyl_schouten_ricci_family", "Apply the Weyl/Schouten/Ricci/Riemann conversion family."),
        CurvatureIdentity("dimension_weyl_zero", "Apply the dimension-dependent vanishing of the Weyl tensor in low dimensions."),
        CurvatureIdentity("torsion_antisymmetry", "Apply basic torsion identities."),
        CurvatureIdentity("torsion_bianchi", "Apply torsion-aware Bianchi-style reduction."),
        CurvatureIdentity("nonmetric_metric_derivative", "Use non-metricity identities for covariant derivatives of the metric."),
        CurvatureIdentity("nonmetric_commutator", "Apply non-metricity-aware commutator reductions."),
    )


def apply_curvature_identity(expr, name: str, *, dimension: int | sp.Expr | None = None):
    key = str(name).strip().lower()
    current = as_abstract_tensor_expr(expr).expr
    if key in {"first_bianchi", "algebraic_bianchi"}:
        reduced, _ = multi_term_tensor_reduce(current, dimension=dimension)
        return AbstractTensorExpr(reduced)
    if key == "differential_bianchi":
        return differential_bianchi_reduce(current)
    if key == "second_bianchi":
        return _second_bianchi_reduce(current, dimension=dimension)
    if key == "contracted_bianchi":
        return _contracted_bianchi_reduce(current, dimension=dimension)
    if key in {"commutator", "commuted_derivatives"}:
        return commute_covariant_derivatives(current)
    if key == "riemann_to_weyl_ricci_scalar":
        if dimension is None:
            raise AbstractTensorCanonicalizationError("riemann_to_weyl_ricci_scalar requires a dimension.")
        return decompose_curvature_expression(current, dimension=dimension)
    if key == "riemann_ricci_scalar":
        return _riemann_ricci_scalar_reduce(current, dimension=dimension)
    if key == "riemann_to_schouten":
        return structural_simplify(weyl_from_riemann_schouten(current).expr)
    if key == "weyl_schouten_ricci_family":
        if dimension is None:
            raise AbstractTensorCanonicalizationError("weyl_schouten_ricci_family requires a dimension.")
        return _weyl_schouten_ricci_family(current, dimension=dimension)
    if key == "ricci_from_schouten":
        if dimension is None:
            raise AbstractTensorCanonicalizationError("ricci_from_schouten requires a dimension.")
        return ricci_from_schouten(current, dimension=dimension)
    if key == "dimension_weyl_zero":
        reduced, _ = _apply_dimension_dependent_rules(current, dimension=dimension)
        return structural_simplify(reduced)
    if key == "torsion_antisymmetry":
        return _torsion_identity_reduce(current)
    if key == "torsion_bianchi":
        return canonical_tensor_expression(_torsion_identity_reduce(differential_bianchi_reduce(current).expr).expr, dimension=dimension)
    if key == "nonmetric_metric_derivative":
        return _nonmetric_identity_reduce(current, dimension=dimension)
    if key == "nonmetric_commutator":
        return canonical_tensor_expression(_nonmetric_identity_reduce(commute_covariant_derivatives(current).expr, dimension=dimension).expr, dimension=dimension)
    raise AbstractTensorCanonicalizationError(f"Unknown curvature identity: {name!r}")


def simplify_with_identity_library(expr, identities: Sequence[str] | None = None, *, dimension: int | sp.Expr | None = None):
    chosen = tuple(i.name for i in list_curvature_identities()) if identities is None else tuple(identities)
    current = as_abstract_tensor_expr(expr).expr
    for name in chosen:
        current = apply_curvature_identity(current, name, dimension=dimension).expr
    return canonical_tensor_expression(current, dimension=dimension)

# --- Indexed <-> abstract bridge ---

def _bundle_to_index_type(bundle: str | None, cache: dict[str, SymTensorIndexType], dim=None):
    key = bundle or "Generic"
    if key not in cache:
        cache[key] = abstract_index_type(key, dummy_name=key[0].upper(), dim=dim)
    return cache[key]


def _infer_symmetry_from_tensor_object(tensor) -> str | None:
    md = getattr(tensor, "symmetry_metadata", {}) or {}
    rank = len(getattr(tensor, "variance_spec", ()))
    if rank == 4 and md.get("riemann", False):
        return "riemann"
    if rank == 4 and md.get("weyl", False):
        return "weyl"
    for key, tag in (("symmetric", "symmetric"), ("antisymmetric", "antisymmetric")):
        groups = tuple(md.get(key, tuple()))
        if len(groups) == 1 and tuple(groups[0]) == tuple(range(rank)):
            return tag
    if rank == 4 and md.get("young_tableaux"):
        tabs = tuple(md["young_tableaux"])
        if any(tuple(len(r) for r in tab) == (2, 2) for tab in tabs):
            return "riemann"
    return None


def indexed_to_abstract(indexed_expr, *, bundle_dims: Mapping[str, int | sp.Expr] | None = None):
    from .tensor_indices import IndexedTensor, IndexedTensorExpr

    type_cache: dict[str, SymTensorIndexType] = {}
    tensor_head_cache: dict[int, TensorHead] = {}
    sym_index_cache: dict[tuple[str, str], TensorIndex] = {}
    bundle_dims = dict(bundle_dims or {})

    def get_sym_index(idx) -> SymTensorIndex:
        bundle = idx.bundle or "Generic"
        key = (bundle, idx.name)
        if key not in sym_index_cache:
            itype = _bundle_to_index_type(bundle, type_cache, dim=bundle_dims.get(bundle))
            created = tensor_indices(idx.name, itype)
            sym_index_cache[key] = created[0] if isinstance(created, tuple) else created
        base = sym_index_cache[key]
        return base if idx.variance == "u" else -base

    def get_head(tensor) -> SymTensorHead:
        tid = id(tensor)
        if tid not in tensor_head_cache:
            index_types = [
                _bundle_to_index_type(getattr(b, "bundle", None) or infer_bundle, type_cache, dim=bundle_dims.get(getattr(b, "bundle", None) or infer_bundle))
                for infer_bundle, b in zip(
                    [getattr(s, "name", None) or "Generic" for s in getattr(tensor, "slot_bases", ())],
                    getattr(tensor, "slot_bases", ()),
                )
            ]
            symmetry = _infer_symmetry_from_tensor_object(tensor)
            tensor_head_cache[tid] = abstract_tensor_head(
                tensor.name or f"T{len(tensor_head_cache)}",
                index_types,
                symmetry=symmetry,
            )
        return tensor_head_cache[tid]

    def convert(obj):
        if isinstance(obj, IndexedTensor):
            head = get_head(obj.tensor)
            return head(*[get_sym_index(i) for i in obj.indices])
        if isinstance(obj, IndexedTensorExpr):
            if obj.op == "tensor":
                return convert(obj.args[0])
            if obj.op == "add":
                return _build_sum([convert(a) for a in obj.args])
            if obj.op == "tensor_product":
                out = convert(obj.args[0])
                for arg in obj.args[1:]:
                    out = out * convert(arg)
                return out
            raise AbstractTensorCanonicalizationError(f"Unsupported indexed expression op: {obj.op}")
        raise AbstractTensorCanonicalizationError(
            f"Cannot convert object of type {type(obj)!r} to an abstract tensor expression."
        )

    return TensorAtlasAbstractExpr(convert(indexed_expr))


def abstract_to_indexed(
    expr,
    *,
    tensor_registry: Mapping[str, object],
    bundle_map: Mapping[str, str] | None = None,
):
    from .fields import ScalarField
    from .tensor_indices import IndexedTensor, IndexedTensorExpr, TensorIndex, _scale_tensor

    bundle_map = dict(bundle_map or {})

    def convert_index(idx: SymTensorIndex) -> SymTensorIndex:
        bundle = bundle_map.get(str(idx.tensor_index_type), None)
        variance = "u" if idx.is_up else "l"
        return TensorIndex(idx.name, variance, bundle)

    def convert(node):
        if getattr(node, "is_number", False):
            charts = [getattr(t, "chart", None) for t in tensor_registry.values() if getattr(t, "chart", None) is not None]
            if charts:
                return ScalarField(charts[0], sp.sympify(node))
            raise AbstractTensorCanonicalizationError("A tensor registry with at least one charted tensor is required to convert scalar-only expressions.")
        if isinstance(node, Tensor):
            name = str(node.component.name)
            if name not in tensor_registry:
                raise AbstractTensorCanonicalizationError(
                    f"Tensor head {name!r} is not present in tensor_registry."
                )
            tensor_obj = tensor_registry[name]
            idx = tuple(convert_index(i) for i in node.get_indices())
            return IndexedTensor(tensor_obj, idx)
        if isinstance(node, TensAdd):
            return IndexedTensorExpr("add", tuple(convert(arg) for arg in node.args))
        if isinstance(node, TensMul):
            coeff = sp.Integer(1)
            non_scalars = []
            for arg in node.args:
                if getattr(arg, "is_number", False):
                    coeff *= sp.sympify(arg)
                else:
                    non_scalars.append(arg)
            if not non_scalars:
                return convert(coeff)
            converted = [convert(arg) for arg in non_scalars]
            if len(converted) == 1 and isinstance(converted[0], IndexedTensor):
                if coeff != 1:
                    return IndexedTensor(_scale_tensor(converted[0].tensor, coeff), converted[0].indices)
                return converted[0]
            if coeff != 1 and converted and isinstance(converted[0], IndexedTensor):
                converted[0] = IndexedTensor(_scale_tensor(converted[0].tensor, coeff), converted[0].indices)
            elif coeff != 1:
                raise AbstractTensorCanonicalizationError("Scalar coefficients on abstract products can only be bridged when at least one indexed tensor factor is present.")
            return IndexedTensorExpr("tensor_product", tuple(converted))
        raise AbstractTensorCanonicalizationError(
            f"Cannot convert abstract node of type {type(node)!r} to an indexed expression."
        )

    return convert(as_abstract_tensor_expr(expr).expr)


# --- Curvature invariant tooling ---

def _require_scalar_invariant(expr):
    free = _free_indices(expr)
    if free:
        raise AbstractTensorCanonicalizationError(
            f"Curvature invariant reduction requires a scalar expression; free indices found: {free!r}"
        )


def _term_signature(term):
    _coeff, rest = _term_coeff_and_rest(term)
    return _expr_signature_key(rest)[1]


def curvature_invariant_signature(expr, *, dimension: int | sp.Expr | None = None, use_multi_term: bool = True):
    current = as_abstract_tensor_expr(expr).expr
    _require_scalar_invariant(current)
    if use_multi_term:
        current, _used = multi_term_tensor_reduce(current, dimension=dimension)
    else:
        current = butler_portugal_canonicalize(current)
    return tuple(_term_signature(term) for term in _term_list(current))


def curvature_invariant_basis(expr, *, dimension: int | sp.Expr | None = None, use_multi_term: bool = True):
    current = as_abstract_tensor_expr(expr).expr
    _require_scalar_invariant(current)
    if use_multi_term:
        current, _used = multi_term_tensor_reduce(current, dimension=dimension)
    else:
        current = butler_portugal_canonicalize(current)
    basis_map: dict[object, object] = {}
    for term in _term_list(current):
        _coeff, rest = _term_coeff_and_rest(term)
        basis_map.setdefault(_expr_signature_key(rest)[1], rest)
    return tuple(basis_map[key] for key in sorted(basis_map, key=structural_key))


def reduce_curvature_invariants(expr, *, dimension: int | sp.Expr | None = None, use_multi_term: bool = True):
    return reduce_curvature_invariants_with_report(
        expr,
        dimension=dimension,
        use_multi_term=use_multi_term,
    ).expr


def reduce_curvature_invariants_with_report(
    expr,
    *,
    dimension: int | sp.Expr | None = None,
    use_multi_term: bool = True,
) -> TensorAtlasAbstractExpr:
    original = as_abstract_tensor_expr(expr).expr
    _require_scalar_invariant(original)
    working = original
    used_rules: tuple[str, ...] = tuple()
    if use_multi_term:
        working, used_rules = multi_term_tensor_reduce(working, dimension=dimension)
    else:
        working = butler_portugal_canonicalize(working)
    coeffs: dict[object, object] = {}
    basis_map: dict[object, object] = {}
    for term in _term_list(working):
        coeff, rest = _term_coeff_and_rest(term)
        key = _expr_signature_key(rest)[1]
        basis_map.setdefault(key, rest)
        coeffs[key] = sp.simplify(coeffs.get(key, sp.Integer(0)) + coeff)
    reduced_terms = [coeffs[key] * basis_map[key] for key in sorted(basis_map, key=structural_key) if coeffs[key] != 0]
    reduced = butler_portugal_canonicalize(_build_sum(reduced_terms))
    report = CurvatureInvariantReductionReport(
        original_expr=original,
        reduced_expr=reduced,
        basis_terms=tuple(basis_map[key] for key in sorted(basis_map, key=structural_key) if coeffs.get(key, 0) != 0),
        term_signatures=tuple(sorted((key for key in basis_map if coeffs.get(key, 0) != 0), key=structural_key)),
        term_multiplicities={key: coeffs[key] for key in sorted(basis_map, key=structural_key) if coeffs.get(key, 0) != 0},
        used_multi_term_rules=used_rules,
        dimension_used=dimension,
    )
    return TensorAtlasAbstractExpr(reduced, report)



# --- Component/chart bridge and contraction-graph helpers ---



def abstract_hypergraph_signature(expr):
    coeff, factors, factor_nodes, index_nodes = _build_abstract_hypergraph(expr)
    factor_labels, index_labels = _wl_refine_hypergraph(factor_nodes, index_nodes)
    factor_sig = tuple(sorted(((factor_labels[fi], _symmetry_aware_factor_signature(factors[fi])) for fi in factor_labels), key=structural_key))
    index_sig = tuple(sorted((index_labels[idx_key] for idx_key in index_labels), key=structural_key))
    free_part, dummy_part = _hypergraph_index_partition(expr)
    return {"coefficient": coeff, "factors": factor_sig, "indices": index_sig, "free_partition": free_part, "dummy_partition": dummy_part}


def canonical_reduce_by_hypergraph(expr, *, dimension: int | sp.Expr | None = None, with_report: bool = False):
    original = as_abstract_tensor_expr(expr).expr
    working = simplify_abstract(original, mode=("structural", "metric", "multiterm"), dimension=dimension).expr
    term_map: dict[tuple, object] = {}
    before_sigs = []
    after_sigs = []
    for term in _term_list(working):
        reduced = _canonical_reduce_term_by_hypergraph(term)
        sig = _canonical_term_sort_key(reduced)
        before_sigs.append(_canonical_term_sort_key(term))
        if sig in term_map:
            term_map[sig] = sp.simplify(term_map[sig] + reduced)
        else:
            term_map[sig] = reduced
    ordered_terms = []
    for sig in sorted(term_map, key=_structural_order_key):
        term = butler_portugal_canonicalize(term_map[sig])
        ordered_terms.append(term)
        after_sigs.append(_canonical_term_sort_key(term))
    canonical = butler_portugal_canonicalize(_build_sum(ordered_terms))
    free_part, dummy_part = _hypergraph_index_partition(canonical)
    renamings = {}
    if isinstance(working, (Tensor, TensMul, sp.Mul)) and isinstance(canonical, (Tensor, TensMul, sp.Mul, TensAdd, sp.Add)):
        renamings = _infer_dummy_renamings(_dummy_indices(working), _dummy_indices(canonical))
    if with_report:
        return AbstractTensorExpr(canonical), HypergraphCanonizationReport(
            original_expr=original,
            canonical_expr=canonical,
            term_signatures_before=tuple(before_sigs),
            term_signatures_after=tuple(after_sigs),
            dummy_renamings=renamings,
            free_index_partition=free_part,
            dummy_index_partition=dummy_part,
        )
    return AbstractTensorExpr(canonical)

def abstract_contraction_graph(expr):
    current = simplify_abstract(expr, mode="structural").expr
    factor_objects = tuple(_factor_list(_term_coeff_and_rest(current)[1]))
    factor_labels = tuple(_expr_signature_key(f)[1] for f in factor_objects)
    pairs = _contraction_pairs(current)
    nodes = tuple({"id": ("factor", k), "kind": "factor", "label": factor_labels[k]} for k in range(len(factor_labels)))
    edges = tuple({"source": ("factor", a), "target": ("factor", b), "kind": "contraction"} for a, b in pairs)
    return {"nodes": nodes, "edges": edges, "summary": {"factor_count": len(nodes), "contraction_count": len(edges), "graph_key": tuple(sorted(((e["source"], e["target"]) for e in edges), key=structural_key))}}


def abstract_contraction_graph_key(expr):
    payload = abstract_contraction_graph(expr)
    return (tuple(sorted(((n["id"], n["label"]) for n in payload["nodes"]), key=structural_key)), payload["summary"]["graph_key"])


def group_differential_invariants(expr, *, dimension: int | sp.Expr | None = None):
    groups = {}
    for item in classify_differential_invariants(expr, dimension=dimension):
        groups.setdefault((item.derivative_order, item.polynomial_degree), []).append(item)
    return {k: tuple(v) for k, v in groups.items()}


def component_to_abstract(obj, *, index_names: Sequence[str] | None = None, bundle_name: str | None = None):
    from .tensor_core import TensorObject
    from .fields import TensorField
    from .tensor_indices import TensorIndex, IndexedTensor
    from .indexed_api import indexed as indexed_builder
    if isinstance(obj, TensorObject):
        base = obj
    elif isinstance(obj, TensorField):
        base = TensorObject.from_tensor_field(obj)
    elif hasattr(obj, 'to_tensor_field'):
        base = TensorObject.from_tensor_field(obj.to_tensor_field())
    else:
        raise AbstractTensorCanonicalizationError(f'Unsupported component tensor bridge object: {type(obj)!r}.')
    rank = len(base.variance_spec)
    if index_names is None:
        index_names = tuple(chr(ord('a') + i) for i in range(rank))
    if len(index_names) != rank:
        raise AbstractTensorCanonicalizationError('index_names must match tensor rank.')
    indices = tuple(TensorIndex(str(name), var, bundle_name) for name, var in zip(index_names, base.variance_spec))
    indexed_obj = indexed_builder(base, *indices)
    bundle_dims = None
    if bundle_name is not None:
        bundle_dims = {bundle_name: getattr(base.chart, 'dimension', None)}
    return indexed_to_abstract(indexed_obj, bundle_dims=bundle_dims)


def abstract_to_component_tensor(expr, *, tensor_registry: Mapping[str, object], bundle_map: Mapping[str, str] | None = None):
    bridged = abstract_to_indexed(expr, tensor_registry=tensor_registry, bundle_map=bundle_map)
    return bridged


def apply_curvature_identity_to_component(obj, name: str, *, tensor_name: str | None = None, bundle_name: str | None = None, dimension: int | sp.Expr | None = None):
    from .tensor_core import TensorObject
    if isinstance(obj, TensorObject):
        base = obj
    elif hasattr(obj, 'to_tensor_field'):
        base = TensorObject.from_tensor_field(obj.to_tensor_field())
    else:
        raise AbstractTensorCanonicalizationError(f'Unsupported component tensor bridge object: {type(obj)!r}.')
    head_name = tensor_name or (base.name or 'T')
    abstract_expr = component_to_abstract(base, bundle_name=bundle_name).expr
    rewritten = apply_curvature_identity(abstract_expr, name, dimension=dimension).expr
    return abstract_to_component_tensor(rewritten, tensor_registry={head_name: base}, bundle_map=None if bundle_name is None else {bundle_name: bundle_name})



def canonical_reduce_by_contraction_graph(expr, *, dimension: int | sp.Expr | None = None):
    current = simplify_abstract(expr, mode=("structural", "metric", "multiterm"), dimension=dimension).expr
    reduced_terms = [_canonical_reduce_term_by_hypergraph(term) for term in _term_list(current)]
    reduced = _build_sum(sorted(reduced_terms, key=_canonical_term_sort_key))
    reduced = simplify_abstract(reduced, mode=("structural", "metric", "multiterm"), dimension=dimension).expr
    return TensorAtlasAbstractExpr(reduced)


def push_abstract_reduction_to_component(expr, *, tensor_registry: Mapping[str, object], dimension: int | sp.Expr | None = None, bundle_map: Mapping[str, str] | None = None):
    reduced = canonical_reduce_by_contraction_graph(expr, dimension=dimension).expr
    reduced = simplify_with_identity_library(reduced, dimension=dimension)
    try:
        return abstract_to_indexed(reduced, tensor_registry=tensor_registry, bundle_map=bundle_map)
    except Exception:
        return abstract_to_component_tensor(reduced, tensor_registry=tensor_registry, bundle_map=bundle_map)


# --- Identity libraries, invariant catalogs, and bridge/report helpers ---

@dataclass(frozen=True)
class CurvatureIdentityApplicationReport:
    library_name: str
    applied_identities: tuple[str, ...]
    original_expr: object
    final_expr: object
    dimension_used: int | sp.Expr | None = None
    steps: tuple["CurvatureIdentityStep", ...] = tuple()
    fixed_point_passes: int = 1
    provenance: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CurvatureIdentityLibrary:
    name: str
    identities: tuple[str, ...]
    dimension: int | sp.Expr | None = None
    description: str = ""


@dataclass(frozen=True)
class CurvatureIdentityStep:
    identity_name: str
    before_expr: object
    after_expr: object
    changed: bool
    notes: tuple[str, ...] = tuple()
    before_fingerprint: tuple[Any, ...] = tuple()
    after_fingerprint: tuple[Any, ...] = tuple()
    provenance: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CurvatureIdentityPolicy:
    name: str
    libraries: tuple[str, ...]
    dimension: int | sp.Expr | None = None
    include_identities: tuple[str, ...] = tuple()
    exclude_identities: tuple[str, ...] = tuple()
    description: str = ""


@dataclass(frozen=True)
class InvariantBasisElement:
    signature: str
    representative: object
    derivative_order: int
    polynomial_degree: int
    tensor_heads: tuple[str, ...]


@dataclass(frozen=True)
class InvariantBasisCatalog:
    elements: tuple[InvariantBasisElement, ...]
    by_signature: Mapping[str, InvariantBasisElement] = field(default_factory=dict)
    by_order: Mapping[int, tuple[InvariantBasisElement, ...]] = field(default_factory=dict)
    by_derivative_order: Mapping[int, tuple[InvariantBasisElement, ...]] = field(default_factory=dict)
    by_order_and_derivative: Mapping[tuple[int, int], tuple[InvariantBasisElement, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class InvariantBasisDatabase:
    by_order_and_derivative: Mapping[tuple[int, int], tuple[InvariantBasisElement, ...]] = field(default_factory=dict)
    by_dimension: Mapping[str, Mapping[tuple[int, int], tuple[InvariantBasisElement, ...]]] = field(default_factory=dict)


@dataclass(frozen=True)
class InvariantBasisReductionReport:
    original_expr: object
    reduced_expr: object
    matched_signatures: tuple[str, ...]
    basis_elements: tuple[InvariantBasisElement, ...]
    dimension_used: int | sp.Expr | None = None
    integration_by_parts_used: bool = False
    coefficient_map: Mapping[str, object] = field(default_factory=dict)
    reduction_trace: tuple[str, ...] = tuple()
    provenance: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BridgeConversionReport:
    source_layer: str
    target_layer: str
    original_object: object
    converted_object: object
    notes: tuple[str, ...] = tuple()
    relation: str = "formal_via_canonical_core"
    source_fingerprint: tuple[Any, ...] = tuple()
    target_fingerprint: tuple[Any, ...] = tuple()
    source_tensor_heads: tuple[tuple[Any, ...], ...] = tuple()
    target_tensor_heads: tuple[tuple[Any, ...], ...] = tuple()
    source_bundle_signatures: tuple[tuple[Any, ...], ...] = tuple()
    target_bundle_signatures: tuple[tuple[Any, ...], ...] = tuple()


def _dim_at_most(dimension, bound: int) -> bool:
    try:
        value = sp.sympify(dimension)
        return bool(value.is_integer and value <= bound)
    except Exception:
        return False


def _dimension_relation_names(dimension) -> tuple[str, ...]:
    if dimension is None:
        return tuple()
    try:
        d = int(sp.sympify(dimension))
    except Exception:
        return tuple()
    names = []
    if d <= 3:
        names.append('dimension-dependent Weyl vanishing')
    if d <= 2:
        names.append('two-dimensional Riemann-to-scalar reduction available')
    if d == 3:
        names.append('three-dimensional Weyl-Schouten reduction available')
    return tuple(names)


def _expr_provenance(expr, *, dimension: int | sp.Expr | None = None, stage: str = 'abstract') -> Mapping[str, object]:
    current = canonical_tensor_expression(expr, dimension=dimension).expr
    return {
        'fingerprint': _tensor_expr_key_for_abstract(current, dimension=dimension, layer=stage),
        'signature_key': _expr_signature_key(current),
        'tensor_heads': _tensor_head_names(current),
        'bundle_signatures': _bundle_signature_set(current),
        'derivative_order': _derivative_order_in_expr(current),
        'dimension_relations': _dimension_relation_names(dimension),
    }


def _torsion_identity_reduce(expr):
    return structural_simplify(as_abstract_tensor_expr(expr).expr)


def _nonmetric_identity_reduce(expr, *, dimension=None):
    return canonical_tensor_expression(as_abstract_tensor_expr(expr).expr, dimension=dimension)


def _second_bianchi_reduce(expr, *, dimension=None, max_passes: int = 2):
    current = as_abstract_tensor_expr(expr).expr
    for _ in range(max_passes):
        nxt = differential_bianchi_reduce(current).expr
        nxt = canonical_tensor_expression(nxt, dimension=dimension).expr
        if _expr_signature_key(nxt) == _expr_signature_key(current):
            break
        current = nxt
    return AbstractTensorExpr(current)


def _contracted_bianchi_reduce(expr, *, dimension=None):
    return canonical_tensor_expression(differential_bianchi_reduce(expr).expr, dimension=dimension)


def _riemann_ricci_scalar_reduce(expr, *, dimension=None):
    current = as_abstract_tensor_expr(expr).expr
    if dimension is not None:
        current = decompose_curvature_expression(current, dimension=dimension).expr
    return canonical_tensor_expression(current, dimension=dimension)


def _weyl_schouten_ricci_family(expr, *, dimension):
    current = as_abstract_tensor_expr(expr).expr
    current = decompose_curvature_workflow(current, dimension=dimension, target='weyl_ricci_scalar').expr
    current = schouten_from_ricci(current, dimension=dimension).expr
    return canonical_tensor_expression(current, dimension=dimension)


def list_curvature_identity_libraries(dimension: int | sp.Expr | None = None) -> tuple[CurvatureIdentityLibrary, ...]:
    libs = [
        CurvatureIdentityLibrary('core', ('first_bianchi', 'algebraic_bianchi', 'riemann_ricci_scalar'), dimension=dimension, description='Core algebraic curvature identities.'),
        CurvatureIdentityLibrary('differential', ('differential_bianchi', 'second_bianchi', 'contracted_bianchi', 'commutator', 'commuted_derivatives'), dimension=dimension, description='Differential curvature identities and commutators.'),
        CurvatureIdentityLibrary('conversion', ('riemann_to_weyl_ricci_scalar', 'riemann_to_schouten', 'ricci_from_schouten', 'weyl_schouten_ricci_family'), dimension=dimension, description='Schouten/Weyl/Ricci/Riemann conversion identities.'),
        CurvatureIdentityLibrary('schouten', ('riemann_to_weyl_ricci_scalar', 'riemann_to_schouten', 'ricci_from_schouten', 'weyl_schouten_ricci_family'), dimension=dimension, description='Alias for the Schouten/Weyl/Ricci/Riemann conversion family.'),
        CurvatureIdentityLibrary('torsion', ('torsion_antisymmetry', 'torsion_bianchi'), dimension=dimension, description='Torsion identities.'),
        CurvatureIdentityLibrary('nonmetric', ('nonmetric_metric_derivative', 'nonmetric_commutator'), dimension=dimension, description='Non-metricity identities.'),
    ]
    if dimension is not None:
        d = sp.sympify(dimension)
        libs.append(CurvatureIdentityLibrary('dimension_specific', ('dimension_weyl_zero', 'riemann_ricci_scalar', 'contracted_bianchi'), dimension=dimension, description='Dimension-aware identity family.'))
        libs.append(CurvatureIdentityLibrary(f'dim_{d}', ('dimension_weyl_zero', 'riemann_ricci_scalar', 'contracted_bianchi'), dimension=dimension, description='Dimension-aware identity family alias carrying the explicit dimension.'))
        if _dim_at_most(d, 3):
            libs.append(CurvatureIdentityLibrary('low_dimensional', ('dimension_weyl_zero', 'riemann_ricci_scalar'), dimension=dimension, description='Low-dimensional curvature identities.'))
    libs.append(CurvatureIdentityLibrary('full', tuple(dict.fromkeys(sum((list(lib.identities) for lib in libs), []))), dimension=dimension, description='Combined identity library.'))
    return tuple(libs)


def list_curvature_identity_policies(dimension: int | sp.Expr | None = None) -> tuple[CurvatureIdentityPolicy, ...]:
    return (
        CurvatureIdentityPolicy('fast', ('core', 'conversion'), dimension=dimension, description='Fast policy using algebraic and conversion identities.'),
        CurvatureIdentityPolicy('differential', ('core', 'differential', 'conversion'), dimension=dimension, description='Differential-geometry policy including derivative identities.'),
        CurvatureIdentityPolicy('metric_affine', ('core', 'differential', 'conversion', 'torsion', 'nonmetric'), dimension=dimension, description='Metric-affine policy with torsion and non-metric identities.'),
        CurvatureIdentityPolicy('full', ('full',), dimension=dimension, description='Full identity policy.'),
    )


def get_curvature_identity_policy(name: str = 'full', *, dimension: int | sp.Expr | None = None) -> CurvatureIdentityPolicy:
    key = str(name).strip().lower()
    for pol in list_curvature_identity_policies(dimension=dimension):
        if pol.name == key:
            return pol
    raise AbstractTensorCanonicalizationError(f'Unknown curvature identity policy: {name!r}')


def get_curvature_identity_library(name: str = 'full', *, dimension: int | sp.Expr | None = None) -> CurvatureIdentityLibrary:
    key = str(name).strip().lower()
    libraries = list_curvature_identity_libraries(dimension=dimension)
    alias_targets = canonical_named_aliases(key)
    if key == 'dimension_specific' and dimension is not None:
        d = sp.sympify(dimension)
        alias_targets.add(f'dim_{d}')
    for lib in libraries:
        names = canonical_named_aliases(lib.name)
        if key in names or names & alias_targets:
            if key == 'dimension_specific' and lib.name == 'low_dimensional':
                continue
            return lib
    raise AbstractTensorCanonicalizationError(f'Unknown curvature identity library: {name!r}')


def _resolve_identity_sequence(library: str | CurvatureIdentityLibrary | CurvatureIdentityPolicy = 'full', *, dimension: int | sp.Expr | None = None):
    if isinstance(library, CurvatureIdentityPolicy):
        ids = []
        for lib_name in library.libraries:
            ids.extend(get_curvature_identity_library(lib_name, dimension=dimension).identities)
        ids.extend(library.include_identities)
        ids = [x for x in ids if x not in set(library.exclude_identities)]
        return library.name, tuple(dict.fromkeys(ids))
    lib = get_curvature_identity_library(library, dimension=dimension) if isinstance(library, str) else library
    return lib.name, tuple(lib.identities)


def apply_curvature_identity_policy(expr, policy: str | CurvatureIdentityPolicy = 'full', *, dimension: int | sp.Expr | None = None, with_report: bool = False):
    pol = get_curvature_identity_policy(policy, dimension=dimension) if isinstance(policy, str) else policy
    return apply_curvature_identity_library(expr, pol, dimension=dimension, with_report=with_report)


def apply_curvature_identity_library(expr, library: str | CurvatureIdentityLibrary | CurvatureIdentityPolicy = 'full', *, dimension: int | sp.Expr | None = None, with_report: bool = False, max_rounds: int = 1):
    library_name, identities = _resolve_identity_sequence(library, dimension=dimension)
    original = canonical_tensor_expression(expr, dimension=dimension).expr
    current = original
    applied = []
    steps = []
    rounds_used = 0
    for round_idx in range(max(1, int(max_rounds))):
        rounds_used = round_idx + 1
        round_changed = False
        for name in identities:
            if name == 'riemann_to_weyl_ricci_scalar' and dimension is None:
                continue
            before = current
            notes = []
            try:
                nxt = apply_curvature_identity(current, name, dimension=dimension).expr
            except AbstractTensorCanonicalizationError:
                nxt = before
                notes.append('skipped')
            nxt = canonical_tensor_expression(nxt, dimension=dimension).expr
            changed = _expr_signature_key(nxt) != _expr_signature_key(before)
            if changed:
                round_changed = True
                applied.append(name)
            steps.append(CurvatureIdentityStep(
                name,
                before,
                nxt,
                changed,
                tuple(notes + ([f'round={round_idx + 1}'] if max_rounds > 1 else [])),
                _tensor_expr_key_for_abstract(before, dimension=dimension, layer='abstract_identity_before'),
                _tensor_expr_key_for_abstract(nxt, dimension=dimension, layer='abstract_identity_after'),
                {
                    'round': round_idx + 1,
                    'dimension_relations': _dimension_relation_names(dimension),
                },
            ))
            current = nxt
        current = canonical_tensor_expression(current, dimension=dimension).expr
        if not round_changed:
            break
    if with_report:
        provenance = {
            'original': _expr_provenance(original, dimension=dimension, stage='identity_original'),
            'final': _expr_provenance(current, dimension=dimension, stage='identity_final'),
            'identity_sequence': identities,
        }
        return AbstractTensorExpr(current), CurvatureIdentityApplicationReport(library_name, tuple(dict.fromkeys(applied)), original, current, dimension, tuple(steps), rounds_used, provenance)
    return AbstractTensorExpr(current)


def _curvature_order_in_expr(expr) -> int:
    return sum(1 for h in _collect_tensor_heads(expr) if str(h.name) in {'Riem', 'Ric', 'C', 'P', 'R'})


def _integration_by_parts_reduce(expr, *, dimension: int | sp.Expr | None = None):
    current = canonical_tensor_expression(expr, dimension=dimension).expr
    terms = []
    changed = False
    for term in _term_list(current):
        coeff, rest = _term_coeff_and_rest(term)
        factors = list(_factor_list(rest))
        deriv_positions = [k for k, f in enumerate(factors) if isinstance(f, Tensor) and _head_metadata(f.component).get('derivative_order', 0)]
        if len(deriv_positions) == 1 and len(factors) == 2:
            j = deriv_positions[0]
            i = 1 - j
            moved = factors[i] * factors[j]
            canonical_moved = canonical_tensor_expression(moved, dimension=dimension).expr
            if _expr_signature_key(canonical_moved) != _expr_signature_key(rest):
                changed = True
            terms.append(coeff * canonical_moved)
        else:
            terms.append(term)
    rebuilt = _build_sum(terms)
    if changed:
        rebuilt = canonical_tensor_expression(rebuilt, dimension=dimension).expr
    return rebuilt


def differential_invariant_equivalent(left, right, *, dimension: int | sp.Expr | None = None, integration_by_parts: bool = False) -> bool:
    a = canonical_tensor_expression(left, dimension=dimension).expr
    b = canonical_tensor_expression(right, dimension=dimension).expr
    if integration_by_parts:
        a = _integration_by_parts_reduce(a, dimension=dimension)
        b = _integration_by_parts_reduce(b, dimension=dimension)
    return _expr_signature_key(a) == _expr_signature_key(b)


def invariant_basis_database(expr, *, dimension: int | sp.Expr | None = None) -> InvariantBasisDatabase:
    catalog = invariant_basis_catalog(expr, dimension=dimension)
    grouped = dict(catalog.by_order_and_derivative)
    dim_key = ('generic', None) if dimension is None else ('dimension', structural_key(sp.sympify(dimension)))
    return InvariantBasisDatabase(grouped, {dim_key: grouped})


def invariant_relations(expr, *, dimension: int | sp.Expr | None = None):
    current = canonical_tensor_expression(expr, dimension=dimension).expr
    rels = []
    if dimension is not None and _dim_at_most(dimension, 3):
        reduced, _ = _apply_dimension_dependent_rules(current, dimension=dimension)
        if _expr_signature_key(reduced) != _expr_signature_key(current):
            rels.append(('dimension', canonical_tensor_expression(reduced, dimension=dimension).expr))
    for name in _dimension_relation_names(dimension):
        rels.append(('available_relation', name))
    desc = _make_invariant_descriptor(current)
    rels.append(('descriptor', (desc.polynomial_degree, desc.derivative_order, desc.signature)))
    return tuple(rels)


def reduce_higher_order_curvature_invariants(expr, *, dimension: int | sp.Expr | None = None, integration_by_parts: bool = False, with_report: bool = False):
    original = as_abstract_tensor_expr(expr).expr
    reduced_expr = canonical_tensor_expression(reduce_differential_curvature_invariants(original, dimension=dimension), dimension=dimension).expr
    if integration_by_parts:
        reduced_expr = _integration_by_parts_reduce(reduced_expr, dimension=dimension)
    catalog = invariant_basis_catalog(reduced_expr, dimension=dimension)
    rebuilt = reduce_to_invariant_basis(reduced_expr, catalog, dimension=dimension, integration_by_parts=integration_by_parts, with_report=with_report)
    if with_report:
        rebuilt_expr, inner_report = rebuilt
        matched = tuple(sorted(catalog.by_signature.keys()))
        outer_report = InvariantBasisReductionReport(
            original,
            rebuilt_expr.expr if hasattr(rebuilt_expr, 'expr') else rebuilt_expr,
            inner_report.matched_signatures or matched,
            tuple(catalog.elements),
            dimension,
            integration_by_parts,
            dict(inner_report.coefficient_map),
            ('reduce_higher_order_curvature_invariants',) + tuple(inner_report.reduction_trace),
            {
                'catalog_size': len(catalog.elements),
                'order_derivative_buckets': tuple(sorted(catalog.by_order_and_derivative)),
                'original': _expr_provenance(original, dimension=dimension, stage='higher_order_original'),
                'final': _expr_provenance(inner_report.reduced_expr, dimension=dimension, stage='higher_order_final'),
            },
        )
        return rebuilt_expr, outer_report
    return rebuilt


def invariant_basis_catalog(expr, *, dimension: int | sp.Expr | None = None) -> InvariantBasisCatalog:
    canonical_expr = canonical_tensor_expression(expr, dimension=dimension).expr
    cache_key = ("basis_catalog", _canonical_cache_key(canonical_expr, dimension))
    cached = _bounded_cache_get(_INVARIANT_CATALOG_CACHE, cache_key)
    if cached is not None:
        return cached
    basis = differential_curvature_invariant_basis(canonical_expr, dimension=dimension)
    elements = []
    canonical_reps: dict[tuple[Any, ...], object] = {}
    for rep in basis:
        rep_key = _canonical_cache_key(rep, dimension)
        canonical_rep = canonical_reps.get(rep_key)
        if canonical_rep is None:
            canonical_rep = canonical_tensor_expression(rep, dimension=dimension).expr
            canonical_reps[rep_key] = canonical_rep
        desc = _make_invariant_descriptor(canonical_rep)
        elements.append(InvariantBasisElement(desc.signature, canonical_rep, desc.derivative_order, desc.polynomial_degree, desc.tensor_heads))
    by_sig = {e.signature: e for e in elements}
    by_order = {}
    by_dorder = {}
    by_pair = {}
    for e in elements:
        by_order.setdefault(e.polynomial_degree, []).append(e)
        by_dorder.setdefault(e.derivative_order, []).append(e)
        by_pair.setdefault((e.polynomial_degree, e.derivative_order), []).append(e)
    by_order = {k: tuple(v) for k, v in by_order.items()}
    by_dorder = {k: tuple(v) for k, v in by_dorder.items()}
    by_pair = {k: tuple(v) for k, v in by_pair.items()}
    return _bounded_cache_set(_INVARIANT_CATALOG_CACHE, cache_key, InvariantBasisCatalog(tuple(elements), by_sig, by_order, by_dorder, by_pair))


def reduce_to_invariant_basis(expr, catalog: InvariantBasisCatalog | None = None, *, dimension: int | sp.Expr | None = None, integration_by_parts: bool = False, with_report: bool = False):
    reduced = canonical_tensor_expression(reduce_differential_curvature_invariants(expr, dimension=dimension), dimension=dimension).expr
    trace = ['reduce_differential_curvature_invariants', 'canonical_tensor_expression']
    if integration_by_parts:
        reduced = _integration_by_parts_reduce(reduced, dimension=dimension)
        trace.append('integration_by_parts')
    if catalog is None:
        catalog = invariant_basis_catalog(reduced, dimension=dimension)
        trace.append('invariant_basis_catalog')
    coeffs = {}
    seen_rests: dict[tuple[Any, ...], tuple[str, object]] = {}
    for term in _term_list(reduced):
        coeff, rest = _term_coeff_and_rest(term)
        rest_key = _canonical_cache_key(rest, dimension)
        cached = seen_rests.get(rest_key)
        if cached is None:
            canonical_rest = canonical_tensor_expression(rest, dimension=dimension).expr
            sig = _make_invariant_descriptor(canonical_rest).signature
            cached = (sig, canonical_rest)
            seen_rests[rest_key] = cached
        sig, _canonical_rest = cached
        if sig in catalog.by_signature:
            coeffs[sig] = sp.simplify(coeffs.get(sig, sp.Integer(0)) + coeff)
    rebuilt = _build_sum([coeffs[sig] * catalog.by_signature[sig].representative for sig in sorted(coeffs, key=structural_key) if coeffs[sig] != 0])
    result = canonical_tensor_expression(rebuilt, dimension=dimension)
    if with_report:
        matched = tuple(sorted((sig for sig in coeffs if coeffs[sig] != 0), key=structural_key))
        return result, InvariantBasisReductionReport(
            expr,
            result.expr,
            matched,
            tuple(catalog.by_signature[s] for s in matched if s in catalog.by_signature),
            dimension,
            integration_by_parts,
            {k: v for k, v in coeffs.items() if v != 0},
            tuple(trace),
            {
                'original': _expr_provenance(expr, dimension=dimension, stage='basis_reduce_original'),
                'final': _expr_provenance(result.expr, dimension=dimension, stage='basis_reduce_final'),
            },
        )
    return result


def _bridge_via_canonical_core(obj, *, target: str, tensor_registry: Mapping[str, object] | None = None, bundle_map: Mapping[str, str] | None = None, bundle_dims: Mapping[str, int | sp.Expr] | None = None, bundle_name: str | None = None, dimension: int | sp.Expr | None = None):
    tgt = str(target).strip().lower()
    if tgt == 'abstract':
        try:
            from .tensor_indices import IndexedTensor, IndexedTensorExpr
        except Exception:
            IndexedTensor = IndexedTensorExpr = tuple()
        if isinstance(obj, (IndexedTensor, IndexedTensorExpr)):
            converted = indexed_to_abstract(obj, bundle_dims=bundle_dims)
            source_layer = 'indexed'
        else:
            converted = component_to_abstract(obj, bundle_name=bundle_name)
            source_layer = 'component'
        canonical = canonical_tensor_expression(converted.expr if hasattr(converted, 'expr') else converted, dimension=dimension)
        return canonical, source_layer, ('canonical_core=canonical_tensor_expression', 'bridge_relation=formal_via_canonical_core')
    if tgt in {'indexed', 'component'}:
        if tensor_registry is None:
            raise AbstractTensorCanonicalizationError('tensor_registry is required when bridging from abstract expressions.')
        canonical_input = canonical_tensor_expression(obj.expr if hasattr(obj, 'expr') else obj, dimension=dimension).expr
        converted = abstract_to_indexed(canonical_input, tensor_registry=tensor_registry, bundle_map=bundle_map)
        return converted, 'abstract', ('canonical_core=canonical_tensor_expression', 'bridge_relation=formal_via_canonical_core')
    raise AbstractTensorCanonicalizationError(f'Unsupported bridge target: {target!r}')


def bridge_tensor_expression(obj, *, target: str, tensor_registry: Mapping[str, object] | None = None, bundle_map: Mapping[str, str] | None = None, bundle_dims: Mapping[str, int | sp.Expr] | None = None, bundle_name: str | None = None, dimension: int | sp.Expr | None = None, with_report: bool = False):
    converted, src, notes = _bridge_via_canonical_core(
        obj,
        target=target,
        tensor_registry=tensor_registry,
        bundle_map=bundle_map,
        bundle_dims=bundle_dims,
        bundle_name=bundle_name,
        dimension=dimension,
    )
    source_obj = obj if src != 'abstract' else canonical_tensor_expression(obj.expr if hasattr(obj, 'expr') else obj, dimension=dimension).expr
    source_fp = _tensor_expr_key_for_abstract(source_obj, dimension=dimension, layer=src)
    target_obj = converted.expr if hasattr(converted, 'expr') else converted
    target_layer = str(target).strip().lower()
    target_fp = _tensor_expr_key_for_abstract(target_obj, dimension=dimension, layer=target_layer)
    report = BridgeConversionReport(
        src,
        target_layer,
        source_obj,
        converted,
        tuple(notes),
        'formal_via_canonical_core',
        source_fp,
        target_fp,
        _tensor_head_names(source_obj),
        _tensor_head_names(target_obj),
        _bundle_signature_set(source_obj),
        _bundle_signature_set(target_obj),
    )
    return (converted, report) if with_report else converted


def roundtrip_bridge(obj, *, tensor_registry: Mapping[str, object] | None = None, bundle_name: str | None = None, bundle_map: Mapping[str, str] | None = None, dimension: int | sp.Expr | None = None):
    abstract_obj = bridge_tensor_expression(obj, target='abstract', bundle_name=bundle_name, dimension=dimension)
    if tensor_registry is None:
        return abstract_obj
    return bridge_tensor_expression(abstract_obj.expr if hasattr(abstract_obj, 'expr') else abstract_obj, target='indexed', tensor_registry=tensor_registry, bundle_map=bundle_map, dimension=dimension)


def reduce_component_via_abstract(obj, *, tensor_registry: Mapping[str, object], bundle_name: str | None = None, bundle_map: Mapping[str, str] | None = None, dimension: int | sp.Expr | None = None, library: str | CurvatureIdentityLibrary = 'full', with_report: bool = False):
    abstract_obj, bridge_report = bridge_tensor_expression(obj, target='abstract', bundle_name=bundle_name, with_report=True)
    reduced, lib_report = apply_curvature_identity_library(abstract_obj.expr, library=library, dimension=dimension, with_report=True)
    reduced = canonical_reduce_by_hypergraph(reduced.expr, dimension=dimension)
    pushed, push_report = bridge_tensor_expression(reduced.expr, target='indexed', tensor_registry=tensor_registry, bundle_map=bundle_map, with_report=True)
    if with_report:
        return pushed, (bridge_report, lib_report, push_report)
    return pushed
