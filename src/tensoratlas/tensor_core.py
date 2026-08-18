from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import sympy as sp

from .simplification_core import light_simplify, canonical_simplify
from .simplification_policy import cheap_simplify, normal_simplify

from .basis import (
    TensorBasis,
    basis_transformation_matrix,
    basis_transformation_matrix_tnf,
    cotangent_basis,
    dual_basis,
    tangent_basis,
    transformed_basis,
)
from .fields import ScalarField, TensorField, VectorField
from .mappings import CoordinateMap
from .normal_forms import TNFMatrix, TNFTensorArray, as_tnf_array, as_tnf_matrix, tnf_build_array, tnf_column_from_entries, tnf_scalar_array


SymmetrySpec = Dict[str, Tuple[Tuple[int, ...], ...]]


def _core_simplify_expr(expr):
    if isinstance(expr, TNFMatrix):
        return expr.map_entries(_core_simplify_expr)
    if isinstance(expr, TNFTensorArray):
        return expr.applyfunc(_core_simplify_expr)
    expr = sp.sympify(expr)
    return cheap_simplify(sp.expand(expr))


@dataclass(frozen=True)
class TensorObject:
    """Basis-aware tensor container.

    slot_bases carries basis information for each slot. For a type (r,s) tensor,
    upper slots typically use tangent bases and lower slots use cotangent bases.
    symmetry_metadata may contain keys like 'symmetric' or 'antisymmetric'
    mapping to tuples of slot groups, e.g. {'symmetric': ((0,1),)}.
    domain_metadata can hold optional user-level metadata such as
    manifold name, signature, units, or application-specific tags.
    """

    chart: object
    components: TNFTensorArray
    variance_spec: str
    slot_bases: Tuple[TensorBasis, ...]
    name: Optional[str] = None
    symmetry_metadata: SymmetrySpec = field(default_factory=dict)
    domain_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", as_tnf_array(self.components))
        if len(self.variance_spec) != self.components.rank():
            raise ValueError("Tensor rank must match variance_spec length.")
        if len(self.slot_bases) != len(self.variance_spec):
            raise ValueError("slot_bases must have one entry per tensor slot.")
        if self.components.shape != (self.chart.dimension,) * len(self.variance_spec):
            raise ValueError("components shape must be (dimension,)*rank.")

    @classmethod
    def from_tensor_field(cls, tensor: TensorField, name: Optional[str] = None, symmetry_metadata: Optional[SymmetrySpec] = None, domain_metadata: Optional[Dict[str, Any]] = None) -> "TensorObject":
        slot_bases = []
        for kind in tensor.variance_spec:
            slot_bases.append(tangent_basis(tensor.chart) if kind == 'u' else cotangent_basis(tensor.chart))
        return cls(tensor.chart, tensor.components, tensor.variance_spec, tuple(slot_bases), name=name, symmetry_metadata=symmetry_metadata or {}, domain_metadata=domain_metadata or {})

    @classmethod
    def from_vector_field(cls, vector: VectorField, name: Optional[str] = None, domain_metadata: Optional[Dict[str, Any]] = None) -> "TensorObject":
        tensor = vector.as_tensor()
        return cls.from_tensor_field(tensor, name=name, domain_metadata=domain_metadata)

    @classmethod
    def from_scalar_field(cls, scalar: ScalarField, name: Optional[str] = None, domain_metadata: Optional[Dict[str, Any]] = None) -> "TensorObject":
        return cls(scalar.chart, tnf_scalar_array(scalar.expr, cleaner=normal_simplify), "", tuple(), name=name, domain_metadata=domain_metadata or {})

    def to_tensor_field(self) -> TensorField:
        return TensorField(self.chart, self.components, self.variance_spec)

    def to_vector_field(self) -> VectorField:
        return self.to_tensor_field().as_vector()

    def simplify(self, assumptions=None, canonicalize_symmetry: bool = True) -> "TensorObject":
        arr = tnf_build_array(self.components.shape, lambda idx: normal_simplify(sp.refine(self.components[idx], assumptions) if assumptions is not None else self.components[idx]))
        out = TensorObject(self.chart, arr, self.variance_spec, self.slot_bases, self.name, dict(self.symmetry_metadata))
        return out.canonicalize_symmetry(assumptions=assumptions) if canonicalize_symmetry else out

    def canonicalize_symmetry(self, assumptions=None) -> "TensorObject":
        rank = len(self.variance_spec)
        if rank == 0:
            return self

        def _canon_entry(idx):
            canon_idx, sign, forced_zero = _canonicalize_component_index(idx, self.symmetry_metadata)
            if forced_zero:
                return sp.Integer(0)
            expr = self.components[canon_idx]
            if sign == -1:
                expr = -expr
            if assumptions is not None:
                expr = sp.refine(expr, assumptions)
            return normal_simplify(expr)

        arr = tnf_build_array(self.components.shape, _canon_entry)
        return TensorObject(self.chart, arr, self.variance_spec, self.slot_bases, self.name, dict(self.symmetry_metadata), dict(self.domain_metadata))

    def permute_slots(self, perm: Sequence[int]) -> "TensorObject":
        perm = tuple(perm)
        if sorted(perm) != list(range(len(self.variance_spec))):
            raise ValueError("perm must be a permutation of slot indices.")
        new_shape = tuple(self.components.shape[p] for p in perm)
        arr = self.components.permutedims(perm)
        new_spec = ''.join(self.variance_spec[p] for p in perm)
        new_bases = tuple(self.slot_bases[p] for p in perm)
        return TensorObject(self.chart, arr, new_spec, new_bases, self.name, _permute_symmetry_metadata(self.symmetry_metadata, perm))

    def contract_slots(self, slot1: int, slot2: int):
        tf = self.to_tensor_field().contract(slot1, slot2)
        if isinstance(tf, ScalarField):
            return tf
        return TensorObject.from_tensor_field(tf, name=self.name)

    def raise_slots(self, slots: Iterable[int]) -> "TensorObject":
        tf = self.to_tensor_field()
        slot_bases = list(self.slot_bases)
        for slot in slots:
            tf = tf.raise_index(slot)
            slot_bases[slot] = dual_basis(slot_bases[slot])
        return TensorObject(self.chart, tf.components, tf.variance_spec, tuple(slot_bases), self.name, dict(self.symmetry_metadata))

    def lower_slots(self, slots: Iterable[int]) -> "TensorObject":
        tf = self.to_tensor_field()
        slot_bases = list(self.slot_bases)
        for slot in slots:
            tf = tf.lower_index(slot)
            slot_bases[slot] = dual_basis(slot_bases[slot])
        return TensorObject(self.chart, tf.components, tf.variance_spec, tuple(slot_bases), self.name, dict(self.symmetry_metadata))

    def change_basis(self, new_slot_bases: Sequence[TensorBasis], coords: Optional[Tuple[sp.Symbol, ...]] = None) -> "TensorObject":
        if len(new_slot_bases) != len(self.slot_bases):
            raise ValueError("new_slot_bases must match tensor rank.")
        if coords is None and getattr(self.chart, 'symbols', None) is not None:
            coords = self.chart.symbols()
        rank = len(self.variance_spec)
        dim = self.chart.dimension
        if rank == 0:
            return TensorObject(self.chart, _rank0_tensor_array(self.components[()]), self.variance_spec, tuple(new_slot_bases), self.name, dict(self.symmetry_metadata))
        matrices = [basis_transformation_matrix_tnf(self.slot_bases[s], new_slot_bases[s], coords) for s in range(rank)]
        def _change_basis_entry(new_idx):
            total = sp.Integer(0)
            for old_idx in product(range(dim), repeat=rank):
                coeff = sp.prod(matrices[s][new_idx[s], old_idx[s]] for s in range(rank))
                total += coeff * self.components[old_idx]
            return light_simplify(total)
        arr = tnf_build_array((dim,) * rank, _change_basis_entry)
        return TensorObject(self.chart, arr, self.variance_spec, tuple(new_slot_bases), self.name, dict(self.symmetry_metadata), dict(self.domain_metadata)).simplify(
            assumptions=self.chart.assumptions(coords) if coords is not None else None
        )

    def transform(self, mapping: CoordinateMap) -> "TensorObject":
        tf = self.to_tensor_field().transform(mapping)
        new_bases = tuple(transformed_basis(b, mapping) for b in self.slot_bases)
        return TensorObject(mapping.target, tf.components, tf.variance_spec, new_bases, self.name, dict(self.symmetry_metadata), dict(self.domain_metadata)).canonicalize_symmetry(
            assumptions=mapping.target.assumptions(mapping.target.symbols()) if getattr(mapping.target, 'assumptions', None) is not None else None
        )

    def push_forward(self, mapping: CoordinateMap) -> "TensorObject":
        if self.chart != mapping.source:
            raise ValueError('push_forward expects a tensor on mapping.source.')
        return self.transform(mapping)

    def pull_back(self, mapping: CoordinateMap) -> "TensorObject":
        if self.chart != mapping.source:
            raise ValueError('pull_back expects a tensor on mapping.source.')
        return self.transform(mapping)

    def with_indices(self, *indices):
        from .tensor_indices import indexed
        return indexed(self, *indices)

    def symmetrize_slots(self, slots: Sequence[int]) -> "TensorObject":
        return _symmetrize_like(self, tuple(slots), antisymmetric=False)

    def antisymmetrize_slots(self, slots: Sequence[int]) -> "TensorObject":
        return _symmetrize_like(self, tuple(slots), antisymmetric=True)

    def young_project_slots(self, tableau: Sequence[Sequence[int]]) -> "TensorObject":
        """Apply a basic Young symmetrizer defined by a tableau of slot indices.

        Rows are symmetrized, then columns are antisymmetrized. This is a practical
        first-order implementation rather than full Young-projector canonicalization.
        """
        out = self
        rows = [tuple(row) for row in tableau if len(tuple(row)) > 1]
        cols = []
        max_len = max((len(tuple(r)) for r in tableau), default=0)
        for j in range(max_len):
            col = tuple(row[j] for row in tableau if j < len(tuple(row)))
            if len(col) > 1:
                cols.append(col)
        for row in rows:
            out = out.symmetrize_slots(row)
        for col in cols:
            out = out.antisymmetrize_slots(col)
        md = dict(out.symmetry_metadata)
        existing = list(md.get('young_tableaux', tuple()))
        existing.append(tuple(tuple(r) for r in tableau))
        md['young_tableaux'] = tuple(existing)
        return TensorObject(out.chart, out.components, out.variance_spec, out.slot_bases, out.name, md, dict(out.domain_metadata)).simplify(canonicalize_symmetry=False)


    def trace(self, slot1: int = 0, slot2: int = 1):
        return self.contract_slots(slot1, slot2)

    def compose(self, other: "TensorObject", pair: tuple[int, int] = (1, 0)) -> "TensorObject":
        """Compose two tensors by tensor product followed by one contraction pair.

        Default pair=(1,0) is matrix-like composition for rank-2 endomorphisms.
        """
        prod = self.tensor_product(other)
        res = prod.contract_slots(pair[0], len(self.variance_spec) + pair[1])
        if isinstance(res, ScalarField):
            return TensorObject(self.chart, _rank0_tensor_array(res.expr), '', tuple(), name=self.name, domain_metadata=dict(self.domain_metadata))
        return res

    def tensor_power(self, n: int) -> "TensorObject":
        if n < 0:
            raise ValueError('tensor_power requires n >= 0')
        if n == 0:
            from .tensor_algebra import identity_tensor
            return TensorObject.from_tensor_field(identity_tensor(self.chart, 'ul'))
        out = self
        for _ in range(n - 1):
            out = out.compose(self)
        return out

    def pseudoinverse(self) -> "TensorObject":
        mat = self.as_matrix()
        pinv = mat.pinv()
        arr = tnf_build_array(self.components.shape, lambda idx: normal_simplify(pinv[idx[0], idx[1]]))
        return TensorObject(self.chart, arr, self.variance_spec, self.slot_bases, self.name, dict(self.symmetry_metadata), dict(self.domain_metadata))

    def eigenvects(self):
        return self.as_matrix().eigenvects()

    def jordan_form(self):
        return self.as_matrix().jordan_form()

    def singular_values(self):
        mat = self.as_matrix()
        return tuple(sp.sqrt(ev) for ev in (mat.T * mat).eigenvals().keys())

    def quadratic_form(self, vector: "TensorObject"):
        if len(self.variance_spec) != 2:
            raise TypeError('quadratic_form requires a rank-2 tensor.')
        if len(vector.variance_spec) != 1:
            raise TypeError('quadratic_form expects a rank-1 tensor/vector.')
        v = tnf_column_from_entries(vector.components[(i,)] for i in range(vector.chart.dimension))
        m = self.as_matrix()
        return normal_simplify((v.T @ m @ v)[0, 0])

    def bilinear(self, left: "TensorObject", right: "TensorObject"):
        lv = tnf_column_from_entries(left.components[(i,)] for i in range(left.chart.dimension)) if len(left.variance_spec) == 1 else left.as_matrix()
        rv = tnf_column_from_entries(right.components[(i,)] for i in range(right.chart.dimension)) if len(right.variance_spec) == 1 else right.as_matrix()
        return normal_simplify((lv.T @ self.as_matrix() @ rv)[0, 0])

    def pfaffian(self):
        mat = self.as_matrix()
        if mat.rows % 2 != 0:
            raise ValueError('Pfaffian requires even dimension.')
        return normal_simplify(mat.pfaffian())


    def multi_contract(self, pairs: Sequence[Tuple[int, int]]):
        """Contract several upper/lower slot pairs in one call."""
        out = self
        removed = []
        for slot1, slot2 in pairs:
            shift1 = sum(1 for r in removed if r < slot1)
            shift2 = sum(1 for r in removed if r < slot2)
            a, b = slot1 - shift1, slot2 - shift2
            res = out.contract_slots(a, b)
            if isinstance(res, ScalarField):
                return res
            out = res
            removed.extend([slot1, slot2])
            removed.sort()
        return out

    def tensor_product(self, other: "TensorObject") -> "TensorObject":
        if self.chart != other.chart:
            raise ValueError("Tensor product requires matching charts.")
        rank1 = len(self.variance_spec)
        rank2 = len(other.variance_spec)
        dim = self.chart.dimension
        arr = tnf_build_array((dim,) * (rank1 + rank2), lambda idx: normal_simplify(
            self.components[idx[:rank1]] * other.components[idx[rank1:]]
        ))
        return TensorObject(
            self.chart,
            arr,
            self.variance_spec + other.variance_spec,
            self.slot_bases + other.slot_bases,
            name=None,
            symmetry_metadata={},
        )

    def wedge(self, other: "TensorObject") -> "TensorObject":
        if set(self.variance_spec) - {'l'} or set(other.variance_spec) - {'l'}:
            raise ValueError("wedge currently requires covariant-form slots only.")
        tf = self.to_tensor_field().wedge(other.to_tensor_field())
        bases = tuple(cotangent_basis(self.chart) for _ in range(len(tf.variance_spec)))
        metadata = {'antisymmetric': (tuple(range(len(tf.variance_spec))),)}
        return TensorObject(self.chart, tf.components, tf.variance_spec, bases, name=None, symmetry_metadata=metadata, domain_metadata=_merge_domain_metadata(self.domain_metadata, other.domain_metadata))

    def exterior_derivative(self) -> "TensorObject":
        tf = self.to_tensor_field().exterior_derivative()
        return TensorObject.from_tensor_field(tf, name=self.name, domain_metadata=dict(self.domain_metadata))

    def hodge_star(self) -> "TensorObject":
        tf = self.to_tensor_field().hodge_star()
        return TensorObject.from_tensor_field(tf, name=self.name, domain_metadata=dict(self.domain_metadata))


@dataclass(frozen=True)
class TensorExpr:
    op: str
    args: Tuple[object, ...]

    def evaluate(self):
        if self.op == 'tensor':
            return self.args[0]
        if self.op == 'add':
            left = _eval_expr(self.args[0])
            right = _eval_expr(self.args[1])
            return add_tensors(left, right)
        if self.op == 'tensor_product':
            left = _eval_expr(self.args[0])
            right = _eval_expr(self.args[1])
            return left.tensor_product(right)
        if self.op == 'wedge':
            left = _eval_expr(self.args[0])
            right = _eval_expr(self.args[1])
            return left.wedge(right)
        raise NotImplementedError(f"Unknown TensorExpr op {self.op!r}")

    def simplify(self):
        return self.evaluate().simplify()


def _eval_expr(obj):
    if isinstance(obj, TensorExpr):
        return obj.evaluate()
    return obj


def add_tensors(left: TensorObject, right: TensorObject) -> TensorObject:
    if left.chart != right.chart or left.variance_spec != right.variance_spec or left.slot_bases != right.slot_bases:
        raise ValueError("Tensor addition requires matching chart, variance, and slot bases.")
    arr = tnf_build_array(left.components.shape, lambda idx: normal_simplify(left.components[idx] + right.components[idx]))
    return TensorObject(left.chart, arr, left.variance_spec, left.slot_bases, name=None, symmetry_metadata=_merge_symmetry(left.symmetry_metadata, right.symmetry_metadata), domain_metadata=_merge_domain_metadata(left.domain_metadata, right.domain_metadata))


def tensor_simplify(obj: Any) -> TensorObject:
    return _eval_expr(obj).simplify()


def tensor_expand(obj: Any) -> TensorObject:
    return tensor_simplify(obj)


def tensor_reduce(obj: Any) -> TensorObject:
    out = tensor_simplify(obj)
    md = out.symmetry_metadata
    if 'antisymmetric' in md:
        for group in md['antisymmetric']:
            out = out.antisymmetrize_slots(group)
    if 'symmetric' in md:
        for group in md['symmetric']:
            out = out.symmetrize_slots(group)
    return out.canonicalize_symmetry()


def _symmetrize_like_basic(tensor: TensorObject, slots: Tuple[int, ...], antisymmetric: bool) -> TensorObject:
    rank = len(tensor.variance_spec)
    if any(s < 0 or s >= rank for s in slots):
        raise ValueError("slot index out of range")
    dim = tensor.chart.dimension
    perms = tuple(permutations(slots))
    arr = tnf_build_array(tensor.components.shape, lambda idx: normal_simplify(sum(
        (_perm_sign_from_images(slots, permuted_slots) if antisymmetric else 1) * tensor.components[tuple(_permute_index_slots(idx, slots, permuted_slots))]
        for permuted_slots in perms
    ) / len(perms)))
    md = dict(tensor.symmetry_metadata)
    key = 'antisymmetric' if antisymmetric else 'symmetric'
    existing = list(md.get(key, tuple()))
    existing.append(tuple(slots))
    md[key] = tuple(existing)
    return TensorObject(tensor.chart, arr, tensor.variance_spec, tensor.slot_bases, tensor.name, md)


def _perm_sign_from_images(domain: Sequence[int], image: Sequence[int]) -> int:
    pos = {d: i for i, d in enumerate(domain)}
    perm = tuple(pos[i] for i in image)
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return -1 if inv % 2 else 1


def _permute_symmetry_metadata(md: SymmetrySpec, perm: Sequence[int]) -> SymmetrySpec:
    inverse = {old: new for new, old in enumerate(perm)}
    out: SymmetrySpec = {}
    for key, groups in md.items():
        out[key] = tuple(tuple(inverse[s] for s in group) for group in groups)
    return out


def _merge_symmetry(left: SymmetrySpec, right: SymmetrySpec) -> SymmetrySpec:
    out = dict(left)
    for key, value in right.items():
        if key in out and out[key] != value:
            continue
        out[key] = value
    return out


def _canonicalize_component_index(index: Sequence[int], md: SymmetrySpec):
    idx = list(index)
    sign = 1
    forced_zero = False
    for group in md.get('antisymmetric', tuple()):
        group = tuple(group)
        values = [idx[s] for s in group]
        if len(set(values)) < len(values):
            forced_zero = True
            break
        desired = sorted(values)
        used = [False] * len(values)
        perm_local = []
        for val in desired:
            for pos, old in enumerate(values):
                if not used[pos] and old == val:
                    used[pos] = True
                    perm_local.append(pos)
                    break
        inv = 0
        for i in range(len(perm_local)):
            for j in range(i + 1, len(perm_local)):
                if perm_local[i] > perm_local[j]:
                    inv += 1
        if inv % 2:
            sign *= -1
        for slot, val in zip(group, desired):
            idx[slot] = val
    if forced_zero:
        return tuple(index), 1, True
    for group in md.get('symmetric', tuple()):
        group = tuple(group)
        desired = sorted(idx[s] for s in group)
        for slot, val in zip(group, desired):
            idx[slot] = val
    return tuple(idx), sign, False



def _merge_domain_metadata(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(left)
    for key, value in right.items():
        if key not in out:
            out[key] = value
        elif out[key] == value:
            continue
        else:
            out[f"right_{key}"] = value
    return out


def tensor_from_components(chart: object, components: Any, variance_spec: str, slot_bases: Sequence[TensorBasis] | None = None, *,
                           name: Optional[str] = None, symmetry_metadata: Optional[SymmetrySpec] = None,
                           domain_metadata: Optional[Dict[str, Any]] = None) -> TensorObject:
    if slot_bases is None:
        slot_bases = tuple(tangent_basis(chart) if k == 'u' else cotangent_basis(chart) for k in variance_spec)
    return TensorObject(chart, as_tnf_array(components), variance_spec, tuple(slot_bases), name=name,
                        symmetry_metadata=symmetry_metadata or {}, domain_metadata=domain_metadata or {})


@dataclass(frozen=True)
class StructuredTensorArray:
    shape: Tuple[int, ...]
    entries: Dict[Tuple[int, ...], sp.Expr]
    symmetry_metadata: SymmetrySpec = field(default_factory=dict)
    domain_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cleaned = {tuple(idx): normal_simplify(val) for idx, val in self.entries.items() if normal_simplify(val) != 0}
        object.__setattr__(self, 'entries', cleaned)

    @classmethod
    def from_dense(cls, array: TNFTensorArray, *, symmetry_metadata: Optional[SymmetrySpec] = None, domain_metadata: Optional[Dict[str, Any]] = None) -> 'StructuredTensorArray':
        dense = as_tnf_array(array)
        indices = [()] if not dense.shape else product(*[range(s) for s in dense.shape])
        entries = {tuple(idx): dense[tuple(idx)] for idx in indices if dense[tuple(idx)] != 0}
        return cls(dense.shape, entries, symmetry_metadata or {}, domain_metadata or {})

    def to_dense(self) -> TNFTensorArray:
        return tnf_build_array(self.shape, lambda idx: self.entries.get(tuple(idx), sp.Integer(0)))

    def sparsity(self) -> float:
        total = 1
        for s in self.shape:
            total *= s
        return 1.0 if total == 0 else 1.0 - (len(self.entries) / total)


def symmetry_canonicalize(obj: Any) -> TensorObject:
    """Apply declared slot symmetries repeatedly until stable."""
    out = tensor_reduce(obj)
    prev = None
    while prev is None or prev.components != out.components:
        prev = out
        md = out.symmetry_metadata
        if 'antisymmetric' in md:
            for group in md['antisymmetric']:
                out = out.antisymmetrize_slots(group)
        if 'symmetric' in md:
            for group in md['symmetric']:
                out = out.symmetrize_slots(group)
        out = out.canonicalize_symmetry()
        if prev.components == out.components:
            break
    return out


# --- Foundational tensor-algebra extensions (initial) ---

@dataclass(frozen=True)
class TensorProjector:
    """Reusable slot symmetrizer/antisymmetrizer.

    slots selects the slots to project. When antisymmetric=False this computes
    the average over permutations of the selected slots. When antisymmetric=True
    it computes the alternating average.
    """
    slots: Tuple[int, ...]
    antisymmetric: bool = False
    normalized: bool = True

    def apply(self, tensor: TensorObject) -> TensorObject:
        if self.antisymmetric:
            return _symmetrize_like(tensor, self.slots, antisymmetric=True, normalized=self.normalized)
        return _symmetrize_like(tensor, self.slots, antisymmetric=False, normalized=self.normalized)

    __call__ = apply


def symmetrizer(slots: Sequence[int], *, normalized: bool = True) -> TensorProjector:
    return TensorProjector(tuple(slots), antisymmetric=False, normalized=normalized)


def antisymmetrizer(slots: Sequence[int], *, normalized: bool = True) -> TensorProjector:
    return TensorProjector(tuple(slots), antisymmetric=True, normalized=normalized)


def _symmetrize_like(tensor: TensorObject, slots: Tuple[int, ...], antisymmetric: bool = False, normalized: bool = True) -> TensorObject:
    slots = tuple(slots)
    rank = len(tensor.variance_spec)
    if any(s < 0 or s >= rank for s in slots):
        raise ValueError("Invalid slot index in symmetrization request.")
    if len(set(slots)) != len(slots):
        raise ValueError("Slots must be distinct.")
    if len(slots) <= 1:
        return tensor
    dim = tensor.chart.dimension
    arr = tnf_build_array(tensor.components.shape, lambda idx: sp.Integer(0))
    perms = list(permutations(range(len(slots))))
    scale = sp.Integer(len(perms)) if normalized else sp.Integer(1)
    def _symmetrized_value(idx):
        total = sp.Integer(0)
        for perm in perms:
            moved = list(idx)
            perm_sign = 1
            values = [idx[s] for s in slots]
            permuted_values = [values[p] for p in perm]
            for target_slot, value in zip(slots, permuted_values):
                moved[target_slot] = value
            if antisymmetric:
                inv = 0
                for i in range(len(perm)):
                    for j in range(i + 1, len(perm)):
                        if perm[i] > perm[j]:
                            inv += 1
                perm_sign = -1 if inv % 2 else 1
            total += perm_sign * tensor.components[tuple(moved)]
        return total / scale if normalized else total

    arr = tnf_build_array(tensor.components.shape, _symmetrized_value)
    md = dict(tensor.symmetry_metadata)
    key = 'antisymmetric' if antisymmetric else 'symmetric'
    groups = list(md.get(key, tuple()))
    if slots not in groups:
        groups.append(slots)
    md[key] = tuple(groups)
    return TensorObject(tensor.chart, arr, tensor.variance_spec, tensor.slot_bases, tensor.name, md).canonicalize_symmetry()


def _tensorobject_as_matrix(self: TensorObject) -> TNFMatrix:
    if len(self.variance_spec) != 2:
        raise ValueError("Rank-2 tensor required.")
    dim = self.chart.dimension
    return TNFMatrix(dim, dim, tuple(tuple(normal_simplify(self.components[(i, j)]) for j in range(dim)) for i in range(dim)))


def _matrix_to_tensorobject(template: TensorObject, matrix, *, variance_spec: Optional[str] = None, slot_bases: Optional[Tuple[TensorBasis, ...]] = None, name: Optional[str] = None):
    dim = template.chart.dimension
    matrix = as_tnf_matrix(matrix)
    if matrix.shape != (dim, dim):
        raise ValueError("Matrix shape must match chart dimension.")
    arr = TNFTensorArray(tuple((dim, dim)), tuple(normal_simplify(matrix[i, j]) for i in range(dim) for j in range(dim)))
    return TensorObject(template.chart, arr, variance_spec or template.variance_spec, slot_bases or template.slot_bases, name or template.name, dict(template.symmetry_metadata))


def _tensorobject_inverse(self: TensorObject) -> TensorObject:
    if self.variance_spec not in {'ul', 'lu', 'll', 'uu'}:
        raise ValueError("inverse() currently supports rank-2 tensors only.")
    mat = self.as_matrix()
    inv = mat.inv()
    if self.variance_spec == 'ul':
        new_spec = 'ul'
    elif self.variance_spec == 'lu':
        new_spec = 'lu'
    elif self.variance_spec == 'll':
        new_spec = 'uu'
    else:
        new_spec = 'll'
    new_bases = tuple(dual_basis(b) for b in self.slot_bases)
    return _matrix_to_tensorobject(self, inv, variance_spec=new_spec, slot_bases=new_bases)


def _tensorobject_determinant(self: TensorObject):
    return self.as_matrix().det().to_sympy()


def _tensorobject_characteristic_polynomial(self: TensorObject, lam=None):
    lam = lam or sp.Symbol('lambda')
    return sp.expand(self.as_matrix().charpoly(lam).as_expr())


def _tensorobject_eigenvals(self: TensorObject):
    return self.as_matrix().eigenvals()


def _tensorobject_symmetric_part(self: TensorObject) -> TensorObject:
    if len(self.variance_spec) != 2:
        raise ValueError("symmetric_part() requires a rank-2 tensor.")
    return self.symmetrize_slots((0, 1))


def _tensorobject_skew_part(self: TensorObject) -> TensorObject:
    if len(self.variance_spec) != 2:
        raise ValueError("skew_part() requires a rank-2 tensor.")
    return self.antisymmetrize_slots((0, 1))


def _tensorobject_commutator(self: TensorObject, other: TensorObject) -> TensorObject:
    if self.chart != other.chart or self.variance_spec != other.variance_spec:
        raise ValueError("Commutator requires tensors on the same chart with matching type.")
    if self.variance_spec not in {'ul', 'lu'}:
        raise ValueError("Commutator currently supports endomorphism-type rank-2 tensors only.")
    a = self.as_matrix()
    b = other.as_matrix()
    return _matrix_to_tensorobject(self, _core_simplify_expr(a * b - b * a), variance_spec=self.variance_spec, slot_bases=self.slot_bases, name=None)


def _tensorobject_anticommutator(self: TensorObject, other: TensorObject) -> TensorObject:
    if self.chart != other.chart or self.variance_spec != other.variance_spec:
        raise ValueError("Anticommutator requires tensors on the same chart with matching type.")
    if self.variance_spec not in {'ul', 'lu'}:
        raise ValueError("Anticommutator currently supports endomorphism-type rank-2 tensors only.")
    a = self.as_matrix()
    b = other.as_matrix()
    return _matrix_to_tensorobject(self, _core_simplify_expr(a * b + b * a), variance_spec=self.variance_spec, slot_bases=self.slot_bases, name=None)


def _tensorobject_contract_all_possible(self: TensorObject):
    out = self
    changed = True
    while changed and isinstance(out, TensorObject):
        changed = False
        for i in range(len(out.variance_spec)):
            for j in range(i + 1, len(out.variance_spec)):
                if out.variance_spec[i] != out.variance_spec[j]:
                    out = out.contract_slots(i, j)
                    changed = True
                    break
            if changed:
                break
    return out


def _tensorobject_contract_by_pairs(self: TensorObject, *pairs: Tuple[int, int]):
    return self.multi_contract(pairs)


def direct_sum_rank2(*tensors: TensorObject, name: Optional[str] = None) -> TensorObject:
    if not tensors:
        raise ValueError("direct_sum_rank2 requires at least one tensor.")
    if any(len(t.variance_spec) != 2 for t in tensors):
        raise ValueError("direct_sum_rank2 currently supports rank-2 tensors only.")
    first = tensors[0]
    if any(t.variance_spec != first.variance_spec for t in tensors[1:]):
        raise ValueError("All tensors must have the same variance.")
    dims = [t.chart.dimension for t in tensors]
    total_dim = sum(dims)
    offsets = []
    offset = 0
    for dim in dims:
        offsets.append(offset)
        offset += dim
    arr = tnf_build_array((total_dim, total_dim), lambda idx: sp.Integer(0))
    entries = list(arr.entries)
    def _flat_index(shape, idx):
        acc = 0
        mult = 1
        for size, pos in zip(reversed(shape), reversed(idx)):
            acc += pos * mult
            mult *= size
        return acc
    for tensor_obj, dim, base in zip(tensors, dims, offsets):
        mat = tensor_obj.as_matrix()
        for i in range(dim):
            for j in range(dim):
                entries[_flat_index((total_dim, total_dim), (base + i, base + j))] = normal_simplify(mat[i, j])
    arr = TNFTensorArray((total_dim, total_dim), tuple(entries))
    from .basis import TensorBasis
    basis_chart = None
    basis0 = TensorBasis(f"{first.slot_bases[0].name}⊕", first.slot_bases[0].kind, basis_chart, total_dim, first.slot_bases[0].dual_name)
    basis1 = TensorBasis(f"{first.slot_bases[1].name}⊕", first.slot_bases[1].kind, basis_chart, total_dim, first.slot_bases[1].dual_name)
    chart = type("AbstractDirectSumChart", (), {"dimension": total_dim, "chart_name": "DirectSum", "metric_name": "Abstract"})()
    return TensorObject(chart, arr, first.variance_spec, (basis0, basis1), name=name or "⊕".join(t.name or "T" for t in tensors), symmetry_metadata={})


def diagonal_tensor(chart: Any, diagonal_entries: Sequence[object], variance_spec: str = 'll', *, name: Optional[str] = None) -> TensorObject:
    if len(diagonal_entries) != chart.dimension:
        raise ValueError("Need one diagonal entry per chart dimension.")
    arr = tnf_build_array((chart.dimension, chart.dimension), lambda idx: sp.sympify(diagonal_entries[idx[0]]) if idx[0] == idx[1] else sp.Integer(0))
    bases = tuple(tangent_basis(chart) if kind == 'u' else cotangent_basis(chart) for kind in variance_spec)
    return TensorObject(chart, arr, variance_spec, bases, name=name, symmetry_metadata={'symmetric': ((0, 1),)})


def block_tensor(blocks: Sequence[Sequence[TensorObject]], *, name: Optional[str] = None) -> TensorObject:
    rows = [list(row) for row in blocks]
    if not rows or not rows[0]:
        raise ValueError("block_tensor requires a non-empty rectangular block grid.")
    row_count = len(rows)
    col_count = len(rows[0])
    if any(len(r) != col_count for r in rows):
        raise ValueError("Blocks must form a rectangular grid.")
    first = rows[0][0]
    if len(first.variance_spec) != 2:
        raise ValueError("block_tensor currently supports rank-2 tensors only.")
    row_dims = [rows[i][0].chart.dimension for i in range(row_count)]
    col_dims = [rows[0][j].chart.dimension for j in range(col_count)]
    if any(rows[i][j].variance_spec != first.variance_spec for i in range(row_count) for j in range(col_count)):
        raise ValueError("All blocks must have the same variance.")
    total_rows = sum(row_dims)
    total_cols = sum(col_dims)
    if total_rows != total_cols:
        raise ValueError("block_tensor currently returns square rank-2 tensors only.")
    entries = [sp.Integer(0)] * (total_rows * total_cols)
    row_offset = 0
    for i, rd in enumerate(row_dims):
        col_offset = 0
        for j, cd in enumerate(col_dims):
            block = rows[i][j]
            mat = block.as_matrix()
            if mat.shape != (rd, cd):
                raise ValueError("Incompatible block shape.")
            for r in range(rd):
                base = (row_offset + r) * total_cols + col_offset
                for c in range(cd):
                    entries[base + c] = normal_simplify(mat[r, c])
            col_offset += cd
        row_offset += rd
    arr = TNFTensorArray((total_rows, total_cols), tuple(entries))
    from .basis import TensorBasis
    chart = type("AbstractBlockChart", (), {"dimension": total_rows, "chart_name": "Block", "metric_name": "Abstract"})()
    basis0 = TensorBasis(f"{first.slot_bases[0].name}[block]", first.slot_bases[0].kind, None, total_rows, first.slot_bases[0].dual_name)
    basis1 = TensorBasis(f"{first.slot_bases[1].name}[block]", first.slot_bases[1].kind, None, total_rows, first.slot_bases[1].dual_name)
    return TensorObject(chart, arr, first.variance_spec, (basis0, basis1), name=name or "BlockTensor", symmetry_metadata={})




def _default_slot_basis(chart, variance: str):
    return tangent_basis(chart) if variance == 'u' else cotangent_basis(chart)


def _tensorobject_canonical_basis_form(self: TensorObject, coords=None) -> TensorObject:
    """Return a symmetry-canonical form in standard coordinate bases when possible."""
    if coords is None and getattr(self.chart, 'symbols', None) is not None:
        coords = self.chart.symbols()
    new_bases = []
    can_change = True
    for basis, var in zip(self.slot_bases, self.variance_spec):
        target = _default_slot_basis(self.chart, var)
        try:
            basis_transformation_matrix_tnf(basis, target, coords)
            new_bases.append(target)
        except Exception:
            if basis == target:
                new_bases.append(target)
            else:
                can_change = False
                new_bases.append(basis)
    out = self.change_basis(tuple(new_bases), coords) if can_change and tuple(new_bases) != self.slot_bases else self
    return symmetry_canonicalize(out).simplify(assumptions=self.chart.assumptions(coords) if coords is not None and getattr(self.chart, 'assumptions', None) is not None else None)


def _tensorobject_equivalent(self: TensorObject, other: TensorObject, *, modulo_basis: bool = True, modulo_symmetry: bool = True) -> bool:
    if not isinstance(other, TensorObject):
        return False
    if self.chart != other.chart or self.variance_spec != other.variance_spec:
        return False
    left = self.canonical_basis_form() if modulo_basis else self
    right = other.canonical_basis_form() if modulo_basis else other
    if modulo_symmetry:
        left = symmetry_canonicalize(left)
        right = symmetry_canonicalize(right)
    left = left.simplify()
    right = right.simplify()
    if left.slot_bases != right.slot_bases:
        try:
            right = right.change_basis(left.slot_bases)
        except Exception:
            return False
    if left.components.rank() == 0:
        return normal_simplify(left.components[()] - right.components[()]) == 0
    for idx in product(*[range(s) for s in left.components.shape]):
        if normal_simplify(left.components[idx] - right.components[idx]) != 0:
            return False
    return True


def direct_sum_tensor(*tensors: TensorObject, name: Optional[str] = None) -> TensorObject:
    """Direct sum for arbitrary same-rank tensors with matching variance.

    Each tensor is embedded as a diagonal hyperblock in a larger tensor whose
    dimension is the sum of the input dimensions.
    """
    if not tensors:
        raise ValueError('direct_sum_tensor requires at least one tensor.')
    first = tensors[0]
    rank = len(first.variance_spec)
    if any(len(t.variance_spec) != rank or t.variance_spec != first.variance_spec for t in tensors[1:]):
        raise ValueError('All tensors must have the same rank and variance.')
    dims = [t.chart.dimension for t in tensors]
    total_dim = sum(dims)
    shape = (total_dim,) * rank
    if rank == 0:
        arr = tnf_build_array((), lambda _: normal_simplify(sum(t.components[()] for t in tensors)))
    else:
        boundaries = []
        offset = 0
        for dim in dims:
            boundaries.append((offset, offset + dim))
            offset += dim
        def _entry(idx):
            for tensor_obj, (start, end) in zip(tensors, boundaries):
                if all(start <= value < end for value in idx):
                    local = tuple(value - start for value in idx)
                    return normal_simplify(tensor_obj.components[local])
            return sp.Integer(0)
        arr = tnf_build_array(shape, _entry)
    basis_chart = None
    slot_bases = tuple(TensorBasis(f'{b.name}⊕', b.kind, basis_chart, total_dim, b.dual_name) for b in first.slot_bases)
    chart = type('AbstractDirectSumChart', (), {'dimension': total_dim, 'chart_name': 'DirectSum', 'metric_name': 'Abstract'})()
    md = dict(first.symmetry_metadata)
    return TensorObject(chart, arr, first.variance_spec, slot_bases, name=name or '⊕'.join(t.name or 'T' for t in tensors), symmetry_metadata=md)


def block_tensor_rank2(blocks: Sequence[Sequence[TensorObject]], *, name: Optional[str] = None, allow_rectangular: bool = False) -> TensorObject:
    rows = [list(row) for row in blocks]
    if not rows or not rows[0]:
        raise ValueError('block_tensor_rank2 requires a non-empty rectangular block grid.')
    row_count = len(rows)
    col_count = len(rows[0])
    if any(len(r) != col_count for r in rows):
        raise ValueError('Blocks must form a rectangular grid.')
    first = rows[0][0]
    if len(first.variance_spec) != 2:
        raise ValueError('block_tensor_rank2 currently supports rank-2 tensors only.')
    row_dims = [rows[i][0].chart.dimension for i in range(row_count)]
    col_dims = [rows[0][j].chart.dimension for j in range(col_count)]
    if any(rows[i][j].variance_spec != first.variance_spec for i in range(row_count) for j in range(col_count)):
        raise ValueError('All blocks must have the same variance.')
    total_rows = sum(row_dims)
    total_cols = sum(col_dims)
    if total_rows != total_cols and not allow_rectangular:
        raise ValueError('Rectangular block tensors require allow_rectangular=True.')
    entries = [sp.Integer(0)] * (total_rows * total_cols)
    def _flat_index(shape, idx):
        acc = 0
        mult = 1
        for size, pos in zip(reversed(shape), reversed(idx)):
            acc += pos * mult
            mult *= size
        return acc
    row_offset = 0
    for i, rd in enumerate(row_dims):
        col_offset = 0
        for j, cd in enumerate(col_dims):
            block = rows[i][j]
            mat = block.as_matrix()
            if mat.shape != (rd, cd):
                raise ValueError('Incompatible block shape.')
            for r in range(rd):
                for c in range(cd):
                    entries[_flat_index((total_rows, total_cols), (row_offset + r, col_offset + c))] = normal_simplify(mat[r, c])
            col_offset += cd
        row_offset += rd
    arr = TNFTensorArray((total_rows, total_cols), tuple(entries))
    chart_dim = max(total_rows, total_cols)
    chart = type('AbstractBlockChart', (), {'dimension': chart_dim, 'chart_name': 'Block', 'metric_name': 'Abstract'})()
    b0 = TensorBasis(f'{first.slot_bases[0].name}[block]', first.slot_bases[0].kind, None, total_rows, first.slot_bases[0].dual_name)
    b1 = TensorBasis(f'{first.slot_bases[1].name}[block]', first.slot_bases[1].kind, None, total_cols, first.slot_bases[1].dual_name)
    if total_rows != total_cols:
        return TNFMatrix(total_rows, total_cols, tuple(tuple(arr[(r, c)] for c in range(total_cols)) for r in range(total_rows)))
    return TensorObject(chart, arr, first.variance_spec, (b0, b1), name=name or 'BlockTensor', symmetry_metadata={})

TensorObject.as_matrix = _tensorobject_as_matrix
TensorObject.inverse = _tensorobject_inverse
TensorObject.determinant = _tensorobject_determinant
TensorObject.characteristic_polynomial = _tensorobject_characteristic_polynomial
TensorObject.eigenvals = _tensorobject_eigenvals
TensorObject.symmetric_part = _tensorobject_symmetric_part
TensorObject.skew_part = _tensorobject_skew_part
TensorObject.commutator = _tensorobject_commutator
TensorObject.anticommutator = _tensorobject_anticommutator
TensorObject.contract_all_possible = _tensorobject_contract_all_possible
TensorObject.contract_by_pairs = _tensorobject_contract_by_pairs
TensorObject.canonical_basis_form = _tensorobject_canonical_basis_form
TensorObject.equivalent = _tensorobject_equivalent


def _tensorobject_add(self: TensorObject, other):
    if isinstance(other, TensorObject):
        return add_tensors(self, other)
    return NotImplemented

def _tensorobject_sub(self: TensorObject, other):
    if isinstance(other, TensorObject):
        neg_arr = tnf_build_array(other.components.shape, lambda idx: -other.components[idx])
        neg = TensorObject(other.chart, neg_arr, other.variance_spec, other.slot_bases, other.name, dict(other.symmetry_metadata))
        return add_tensors(self, neg)
    return NotImplemented

TensorObject.__add__ = _tensorobject_add
TensorObject.__sub__ = _tensorobject_sub


block_tensor_general = block_tensor_rank2
direct_sum_general = direct_sum_tensor


def composition(*tensors: TensorObject, pair: tuple[int,int] = (1,0)) -> TensorObject:
    if not tensors:
        raise ValueError('Need at least one tensor.')
    out = tensors[0]
    for t in tensors[1:]:
        out = out.compose(t, pair=pair)
    return out





def _tensor_scale(tensor: "TensorObject", scalar):
    arr = tnf_build_array(tensor.components.shape, lambda idx: normal_simplify(scalar * tensor.components[idx]))
    return TensorObject(tensor.chart, arr, tensor.variance_spec, tensor.slot_bases, tensor.name, dict(tensor.symmetry_metadata))

# --- v18 stronger Young-projector / irreducible-symmetry utilities ---

def _hook_lengths_from_tableau(tableau):
    hooks = {}
    rows = [tuple(r) for r in tableau]
    for i, row in enumerate(rows):
        for j, slot in enumerate(row):
            right = len(row) - j - 1
            below = sum(1 for r in rows[i + 1:] if j < len(r))
            hooks[slot] = right + below + 1
    return hooks

def young_projector_data(tableau: Sequence[Sequence[int]]) -> tuple[Tuple[int, ...], ...]:
    rows = [tuple(r) for r in tableau]
    cols = []
    max_len = max((len(r) for r in rows), default=0)
    for j in range(max_len):
        col = tuple(r[j] for r in rows if j < len(r))
        if col:
            cols.append(col)
    hooks = _hook_lengths_from_tableau(rows)
    hook_product = sp.Integer(1)
    for v in hooks.values():
        hook_product *= v
    size = sum(len(r) for r in rows)
    return {
        'rows': tuple(rows),
        'cols': tuple(cols),
        'shape': tuple(len(r) for r in rows),
        'hook_lengths': hooks,
        'hook_product': normal_simplify(hook_product),
        'size': size,
    }

def young_normalized_project_slots(tensor: TensorObject, tableau: Sequence[Sequence[int]]) -> TensorObject:
    data = young_projector_data(tableau)
    out = tensor
    for row in data['rows']:
        if len(row) > 1:
            out = out.symmetrize_slots(row)
    for col in data['cols']:
        if len(col) > 1:
            out = out.antisymmetrize_slots(col)
    scale = sp.factorial(data['size']) / data['hook_product']
    out = _tensor_scale(out, scale)
    md = dict(out.symmetry_metadata)
    existing = list(md.get('young_tableaux', tuple()))
    existing.append(tuple(tuple(r) for r in tableau))
    md['young_tableaux'] = tuple(existing)
    return TensorObject(out.chart, out.components, out.variance_spec, out.slot_bases, out.name, md, dict(out.domain_metadata)).simplify(canonicalize_symmetry=False)

def young_irreducible_canonicalize(tensor: TensorObject) -> TensorObject:
    md = dict(tensor.symmetry_metadata)
    out = tensor
    tableaux = md.get('young_tableaux', tuple())
    if tableaux:
        for tab in tableaux:
            out = young_normalized_project_slots(out, tab)
        return out.simplify(canonicalize_symmetry=False)
    for _ in range(2):
        for group in md.get('symmetric', tuple()):
            out = out.symmetrize_slots(tuple(group))
        for group in md.get('antisymmetric', tuple()):
            out = out.antisymmetrize_slots(tuple(group))
    return out.simplify(canonicalize_symmetry=False)


def _coerce_tensor_object(obj):
    if isinstance(obj, TensorObject):
        return obj
    if isinstance(obj, TensorField):
        return TensorObject.from_tensor_field(obj)
    raise TypeError("Expected a TensorObject or TensorField.")


def double_contraction(left, right):
    left_obj = _coerce_tensor_object(left)
    right_obj = _coerce_tensor_object(right)
    if left_obj.chart != right_obj.chart or left_obj.components.shape != right_obj.components.shape:
        raise ValueError("double_contraction requires tensors on the same chart with the same component shape.")
    total = sp.Integer(0)
    for idx in product(*[range(n) for n in left_obj.components.shape]):
        total += left_obj.components[idx] * right_obj.components[idx]
    return ScalarField(left_obj.chart, normal_simplify(total))


def deviatoric_part(obj):
    tensor = _coerce_tensor_object(obj)
    if tensor.components.rank() != 2 or tensor.components.shape[0] != tensor.components.shape[1]:
        raise ValueError("deviatoric_part expects a square rank-2 tensor.")
    n = tensor.chart.dimension
    tf = tensor.to_tensor_field()
    metric = tensor.chart.metric(tensor.chart.symbols())
    ginv = tensor.chart.inverse_metric(tensor.chart.symbols())
    if metric is None or ginv is None:
        raise ValueError("Chart does not define a metric.")
    trace = sum(ginv[i, j] * tf.components[i, j] for i in range(n) for j in range(n))
    out = tnf_build_array((n, n), lambda idx: normal_simplify(tf.components[idx] - metric[idx] * trace / n))
    return TensorObject(tensor.chart, out, tf.variance_spec, tensor.slot_bases, name=tensor.name, symmetry_metadata=dict(tensor.symmetry_metadata))


def trace_free_part(obj):
    return deviatoric_part(obj)


def hydrostatic_part(obj):
    tensor = _coerce_tensor_object(obj)
    if tensor.components.rank() != 2 or tensor.components.shape[0] != tensor.components.shape[1]:
        raise ValueError("hydrostatic_part expects a square rank-2 tensor.")
    n = tensor.chart.dimension
    tf = tensor.to_tensor_field()
    metric = tensor.chart.metric(tensor.chart.symbols())
    ginv = tensor.chart.inverse_metric(tensor.chart.symbols())
    if metric is None or ginv is None:
        raise ValueError("Chart does not define a metric.")
    trace = sum(ginv[i, j] * tf.components[i, j] for i in range(n) for j in range(n))
    out = tnf_build_array((n, n), lambda idx: normal_simplify(metric[idx] * trace / n))
    return TensorObject(tensor.chart, out, tf.variance_spec, tensor.slot_bases, name=tensor.name, symmetry_metadata=dict(tensor.symmetry_metadata))


def principal_invariants(obj):
    tensor = _coerce_tensor_object(obj)
    if tensor.components.rank() != 2 or tensor.components.shape[0] != tensor.components.shape[1]:
        raise ValueError("principal_invariants expects a square rank-2 tensor.")
    M = sp.Matrix(tensor.components.to_sympy().tolist())
    n = M.shape[0]
    I1 = normal_simplify(M.trace())
    I2 = normal_simplify(sp.Rational(1, 2) * (I1**2 - (M * M).trace()))
    if n == 2:
        I3 = normal_simplify(M.det())
    else:
        I3 = normal_simplify(M.det())
    return {"I1": I1, "I2": I2, "I3": I3}


def principal_values(obj):
    tensor = _coerce_tensor_object(obj)
    if tensor.components.rank() != 2 or tensor.components.shape[0] != tensor.components.shape[1]:
        raise ValueError("principal_values expects a square rank-2 tensor.")
    M = sp.Matrix(tensor.components.to_sympy().tolist())
    vals = []
    for val, mult in M.eigenvals().items():
        vals.extend([normal_simplify(val)] * int(mult))
    return tuple(vals)


def traction_vector(stress, normal):
    stress_obj = _coerce_tensor_object(stress)
    if stress_obj.components.rank() != 2:
        raise ValueError("traction_vector expects a rank-2 stress tensor.")
    nvec = sp.Matrix(normal)
    if len(nvec) != stress_obj.chart.dimension:
        raise ValueError("normal must match chart dimension.")
    out = sp.Matrix(stress_obj.components.to_sympy().tolist()) * nvec
    arr = tnf_build_array((stress_obj.chart.dimension,), lambda idx: normal_simplify(out[idx[0], 0]))
    return TensorObject(stress_obj.chart, arr, 'u', (stress_obj.slot_bases[0],), symmetry_metadata={}, domain_metadata=dict(stress_obj.domain_metadata))


def to_mandel(obj):
    tensor = _coerce_tensor_object(obj)
    dim = tensor.chart.dimension
    root2 = sp.sqrt(2)
    if tensor.components.rank() == 2:
        if dim == 2:
            i = tensor.components
            return as_tnf_matrix(sp.Matrix([i[0,0], i[1,1], root2*i[0,1]]))
        if dim == 3:
            i = tensor.components
            return as_tnf_matrix(sp.Matrix([i[0,0], i[1,1], i[2,2], root2*i[1,2], root2*i[0,2], root2*i[0,1]]))
    if tensor.components.rank() == 4:
        V = to_voigt(tensor).to_sympy()
        if dim == 2:
            S = sp.diag(1,1,root2)
        elif dim == 3:
            S = sp.diag(1,1,1,root2,root2,root2)
        else:
            raise ValueError("to_mandel expects dimension 2 or 3.")
        return as_tnf_matrix(S * V * S)
    raise ValueError("to_mandel expects a rank-2 or rank-4 tensor.")


def voigt_map(dimension: int):
    if dimension == 2:
        return ((0, 0), (1, 1), (0, 1))
    if dimension == 3:
        return ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    raise ValueError("Voigt notation is implemented for dimensions 2 and 3.")


def to_voigt(obj):
    tensor = _coerce_tensor_object(obj)
    dim = tensor.chart.dimension
    mapping = voigt_map(dim)
    if tensor.components.rank() == 2:
        return as_tnf_matrix(sp.Matrix([tensor.components[i, j] for (i, j) in mapping]))
    if tensor.components.rank() == 4:
        matrix = sp.Matrix([[tensor.components[i, j, k, l] for (k, l) in mapping] for (i, j) in mapping])
        return as_tnf_matrix(matrix)
    raise ValueError("to_voigt expects a rank-2 or rank-4 tensor.")
