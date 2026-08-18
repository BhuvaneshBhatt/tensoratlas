"""Backend-light differential-form operations over component bases."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from typing import Any, Mapping, Sequence

from .components import ComponentTensor, Basis, IndexTuple, Scalar, _require_sympy, _simplify_scalar, coordinate_symbols
from .manifolds import TensorKernelError
from .tensor_heads import TensorHead


def _permutation_sign(items: Sequence[int]) -> int:
    inversions = 0
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if items[left] > items[right]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _canonical_form_key(key: IndexTuple) -> tuple[int, IndexTuple]:
    if len(set(key)) < len(key):
        return 0, tuple(key)
    ordered = tuple(sorted(key))
    return _permutation_sign(key), ordered


@dataclass(frozen=True, slots=True)
class DifferentialForm:
    """Sparse differential form in a chosen basis.

    Components use the conventional antisymmetric tensor normalization: a
    one-form alpha has components alpha_i, and wedge products satisfy
    dx^i wedge dx^j = -dx^j wedge dx^i with component (i,j)=1 for i<j.
    """

    basis: Basis
    degree: int
    components: Mapping[IndexTuple, Scalar]
    default: Scalar = 0

    def __post_init__(self) -> None:
        if self.degree < 0:
            raise TensorKernelError("Form degree must be non-negative.")
        cleaned: dict[IndexTuple, Scalar] = {}
        for key, value in self.components.items():
            ckey = tuple(key)
            if len(ckey) != self.degree:
                raise TensorKernelError("Differential-form component key has the wrong degree.")
            for item in ckey:
                self.basis._validate_component_index((item,))
            sign, canonical = _canonical_form_key(ckey)
            if sign == 0:
                continue
            stored = value if sign == 1 else -value
            if stored != self.default:
                cleaned[canonical] = _simplify_scalar(cleaned.get(canonical, 0) + stored)
        object.__setattr__(self, "components", {k: v for k, v in cleaned.items() if v != self.default})

    @property
    def dimension(self) -> int:
        return self.basis.dimension

    @classmethod
    def zero(cls, basis: Basis, degree: int) -> "DifferentialForm":
        """Return the zero form of a specified degree."""
        return cls(basis, degree, {})

    @classmethod
    def basis_one_form(cls, basis: Basis, slot: int) -> "DifferentialForm":
        """Return the elementary one-form dual to the selected basis vector."""
        basis._validate_component_index((slot,))
        return cls(basis, 1, {(slot,): 1})

    def _check_same_space(self, other: "DifferentialForm") -> None:
        if self.basis != other.basis or self.degree != other.degree:
            raise TensorKernelError("Differential forms must have the same basis and degree.")

    def __add__(self, other: "DifferentialForm") -> "DifferentialForm":
        self._check_same_space(other)
        keys = set(self.components) | set(other.components)
        out = {key: _simplify_scalar(self.component(*key) + other.component(*key)) for key in keys}
        return DifferentialForm(self.basis, self.degree, {k: v for k, v in out.items() if v != self.default}, default=self.default)

    def __sub__(self, other: "DifferentialForm") -> "DifferentialForm":
        return self + (-other)

    def __neg__(self) -> "DifferentialForm":
        return DifferentialForm(self.basis, self.degree, {key: _simplify_scalar(-value) for key, value in self.components.items()}, default=self.default)

    def __mul__(self, scalar: Scalar) -> "DifferentialForm":
        if isinstance(scalar, DifferentialForm):
            return NotImplemented
        return DifferentialForm(self.basis, self.degree, {key: _simplify_scalar(value * scalar) for key, value in self.components.items()}, default=self.default)

    def __rmul__(self, scalar: Scalar) -> "DifferentialForm":
        return self * scalar

    def component(self, *index: int) -> Scalar:
        if len(index) != self.degree:
            raise TensorKernelError("Component lookup degree mismatch.")
        sign, canonical = _canonical_form_key(tuple(index))
        if sign == 0:
            return self.default
        value = self.components.get(canonical, self.default)
        return value if sign == 1 else -value

    def to_component_tensor(self, *, name: str = "omega") -> ComponentTensor:
        head = TensorHead(name, (self.basis.index_type,) * self.degree, symmetry="antisymmetric", variance=("down",) * self.degree)
        return ComponentTensor(head, self.basis, self.components, variance=("down",) * self.degree)

    @classmethod
    def from_component_tensor(cls, tensor: ComponentTensor) -> "DifferentialForm":
        if any(slot != "down" for slot in tensor.variance):
            raise TensorKernelError("Differential forms require covariant component tensors.")
        return cls(tensor.basis, tensor.rank, tensor.components, default=tensor.default)

    @classmethod
    def scalar(cls, basis: Basis, value: Scalar) -> "DifferentialForm":
        return cls(basis, 0, {(): value} if value != 0 else {})

    def wedge(self, other: "DifferentialForm") -> "DifferentialForm":
        if self.basis != other.basis:
            raise TensorKernelError("Wedge product requires forms on the same basis.")
        degree = self.degree + other.degree
        if degree > self.dimension:
            return DifferentialForm(self.basis, degree, {})
        out: dict[IndexTuple, Scalar] = {}
        for left_key, left_value in self.components.items():
            for right_key, right_value in other.components.items():
                combined = tuple(left_key) + tuple(right_key)
                sign, canonical = _canonical_form_key(combined)
                if sign == 0:
                    continue
                out[canonical] = _simplify_scalar(out.get(canonical, 0) + sign * left_value * right_value)
        return DifferentialForm(self.basis, degree, out, default=self.default)

    def exterior_derivative(self, *, coordinates: Sequence[Scalar] | None = None) -> "DifferentialForm":
        """Return the exterior derivative.

        Coordinate bases use the usual alternating coordinate derivative.
        Non-coordinate frames additionally use the structure coefficients
        ``[e_i,e_j]=c^m_{ij}e_m``. Derivatives of frame components are taken
        only when explicit coordinate symbols are supplied; otherwise that part
        is treated as zero for bounded algebraic moving-frame calculations.
        """
        if self.degree == self.dimension:
            return DifferentialForm(self.basis, self.degree + 1, {})
        coords = tuple(coordinates) if coordinates is not None else (
            coordinate_symbols(self.basis.coordinates) if self.basis.kind == "coordinate" else ()
        )
        sp = _require_sympy() if coords else None
        out_degree = self.degree + 1
        out: dict[IndexTuple, Scalar] = {}
        for key in combinations(range(self.dimension), out_degree):
            total = 0
            if coords:
                for pos, deriv_index in enumerate(key):
                    rest = key[:pos] + key[pos + 1:]
                    total += (-1) ** pos * sp.diff(self.component(*rest), coords[deriv_index])
            for left_pos in range(out_degree):
                for right_pos in range(left_pos + 1, out_degree):
                    left_index = key[left_pos]
                    right_index = key[right_pos]
                    rest = key[:left_pos] + key[left_pos + 1:right_pos] + key[right_pos + 1:]
                    sign = (-1) ** (left_pos + right_pos)
                    for upper in range(self.dimension):
                        coeff = self.basis.structure_coefficient(upper, left_index, right_index)
                        if coeff != 0:
                            total += sign * coeff * self.component(upper, *rest)
            total = _simplify_scalar(total)
            if total != 0:
                out[tuple(key)] = total
        return DifferentialForm(self.basis, out_degree, out, default=self.default)

    def interior_product(self, vector: ComponentTensor) -> "DifferentialForm":
        if vector.basis != self.basis or vector.rank != 1 or vector.variance != ("up",):
            raise TensorKernelError("Interior product requires a contravariant vector on the same basis.")
        if self.degree == 0:
            return DifferentialForm(self.basis, 0, {})
        out: dict[IndexTuple, Scalar] = {}
        for key in product(range(self.dimension), repeat=self.degree - 1):
            if len(set(key)) < len(key):
                continue
            total = 0
            for vec_index in range(self.dimension):
                total += vector.component(vec_index) * self.component(vec_index, *key)
            total = _simplify_scalar(total)
            if total != 0:
                sign, canonical = _canonical_form_key(tuple(key))
                if sign != 0:
                    out[canonical] = _simplify_scalar(out.get(canonical, 0) + sign * total)
        return DifferentialForm(self.basis, self.degree - 1, out, default=self.default)

    def hodge_star(self, metric: ComponentTensor, inverse_metric: ComponentTensor | None = None) -> "DifferentialForm":
        if metric.basis != self.basis or metric.variance != ("down", "down"):
            raise TensorKernelError("Hodge star requires a covariant metric on the same basis.")
        sp = _require_sympy()
        inv = inverse_metric
        if inv is None:
            from .components import inverse_metric_from_metric
            inv = inverse_metric_from_metric(metric)
        if inv.basis != self.basis or inv.variance != ("up", "up"):
            raise TensorKernelError("Inverse metric must be contravariant and use the same basis.")
        dense_metric = sp.Matrix(metric.to_dense())
        sqrt_det = sp.sqrt(sp.Abs(dense_metric.det()))
        out_degree = self.dimension - self.degree
        out: dict[IndexTuple, Scalar] = {}
        for out_key in product(range(self.dimension), repeat=out_degree):
            if len(set(out_key)) < len(out_key):
                continue
            total = 0
            for low_key in product(range(self.dimension), repeat=self.degree) if self.degree else [()]:
                if len(set(low_key)) < len(low_key):
                    continue
                raised_total = 0
                for high_key in product(range(self.dimension), repeat=self.degree) if self.degree else [()]:
                    weight = 1
                    for slot in range(self.degree):
                        weight *= inv.component(low_key[slot], high_key[slot])
                    raised_total += weight * self.component(*high_key)
                total += sp.LeviCivita(*(tuple(low_key) + tuple(out_key))) * raised_total
            total = _simplify_scalar(sqrt_det * total / sp.factorial(self.degree))
            if total != 0:
                sign, canonical = _canonical_form_key(tuple(out_key))
                if sign != 0:
                    out[canonical] = _simplify_scalar(out.get(canonical, 0) + sign * total)
        return DifferentialForm(self.basis, out_degree, out, default=self.default)

    def codifferential(self, metric: ComponentTensor, inverse_metric: ComponentTensor | None = None) -> "DifferentialForm":
        if self.degree == 0:
            raise TensorKernelError("Codifferential is defined on positive-degree forms.")
        result = self.hodge_star(metric, inverse_metric).exterior_derivative().hodge_star(metric, inverse_metric)
        sign = (-1) ** (self.dimension * (self.degree + 1) + 1)
        return DifferentialForm(result.basis, result.degree, {k: _simplify_scalar(sign * v) for k, v in result.components.items()})

    def de_rham_laplacian(self, metric: ComponentTensor, inverse_metric: ComponentTensor | None = None) -> "DifferentialForm":
        terms: list[DifferentialForm] = []
        if self.degree > 0:
            terms.append(self.codifferential(metric, inverse_metric).exterior_derivative())
        if self.degree < self.dimension:
            terms.append(self.exterior_derivative().codifferential(metric, inverse_metric))
        if not terms:
            raise TensorKernelError("The Hodge Laplacian is not defined for this form.")
        out: dict[IndexTuple, Scalar] = {}
        for term in terms:
            for key, value in term.components.items():
                out[key] = _simplify_scalar(out.get(key, 0) + value)
        return DifferentialForm(self.basis, self.degree, out)

    def pullback(self, transform_matrix: Sequence[Sequence[Scalar]], target_basis: Basis | None = None) -> "DifferentialForm":
        """Pull back a covariant form with a matrix acting on one-form slots."""
        dim = self.dimension
        matrix = tuple(tuple(row) for row in transform_matrix)
        if len(matrix) != dim or any(len(row) != dim for row in matrix):
            raise TensorKernelError("Pullback matrix must be square with dimension matching the form basis.")
        basis = target_basis or self.basis
        out: dict[IndexTuple, Scalar] = {}
        for new_key in product(range(dim), repeat=self.degree):
            if len(set(new_key)) < len(new_key):
                continue
            total = 0
            for old_key in product(range(dim), repeat=self.degree):
                coeff = 1
                for slot, (old_i, new_i) in enumerate(zip(old_key, new_key)):
                    coeff *= matrix[old_i][new_i]
                total += coeff * self.component(*old_key)
            total = _simplify_scalar(total)
            if total != 0:
                sign, canonical = _canonical_form_key(tuple(new_key))
                if sign != 0:
                    out[canonical] = _simplify_scalar(out.get(canonical, 0) + sign * total)
        return DifferentialForm(basis, self.degree, out, default=self.default)



def basis_one_form(basis: Basis, slot: int) -> DifferentialForm:
    """Return the elementary covector e^slot."""
    return DifferentialForm.basis_one_form(basis, slot)


def volume_form(basis: Basis, density: Scalar = 1) -> DifferentialForm:
    """Return density times the oriented coordinate volume form."""
    return DifferentialForm(basis, basis.dimension, {tuple(range(basis.dimension)): density})


def lie_derivative(form: DifferentialForm, vector: ComponentTensor, *, coordinates: Sequence[Scalar] | None = None) -> DifferentialForm:
    """Compute the Lie derivative of a form using Cartan's formula."""
    if form.basis != vector.basis:
        raise TensorKernelError("Lie derivative requires a vector and form on the same basis.")
    first = form.exterior_derivative(coordinates=coordinates).interior_product(vector)
    second = form.interior_product(vector).exterior_derivative(coordinates=coordinates)
    return first + second


def hodge_star(form: DifferentialForm, metric: ComponentTensor, inverse_metric: ComponentTensor | None = None) -> DifferentialForm:
    """Functional wrapper for the Hodge star."""
    return form.hodge_star(metric, inverse_metric)


def codifferential(form: DifferentialForm, metric: ComponentTensor, inverse_metric: ComponentTensor | None = None) -> DifferentialForm:
    """Functional wrapper for the codifferential."""
    return form.codifferential(metric, inverse_metric)


def de_rham_laplacian(form: DifferentialForm, metric: ComponentTensor, inverse_metric: ComponentTensor | None = None) -> DifferentialForm:
    """Functional wrapper for the Hodge-de Rham Laplacian."""
    return form.de_rham_laplacian(metric, inverse_metric)

def form_from_components(basis: Basis, degree: int, components: Mapping[IndexTuple, Scalar]) -> DifferentialForm:
    return DifferentialForm(basis, degree, components)


def wedge(left: DifferentialForm, right: DifferentialForm) -> DifferentialForm:
    return left.wedge(right)


def exterior_derivative(form: DifferentialForm, *, coordinates: Sequence[Scalar] | None = None) -> DifferentialForm:
    return form.exterior_derivative(coordinates=coordinates)


def exterior_derivative_squared(form: DifferentialForm, *, coordinates: Sequence[Scalar] | None = None) -> DifferentialForm:
    """Return ``d(d(form))`` as a nilpotency regression helper."""
    return form.exterior_derivative(coordinates=coordinates).exterior_derivative(coordinates=coordinates)


def cartan_identity_residual(form: DifferentialForm, vector: ComponentTensor, *, coordinates: Sequence[Scalar] | None = None) -> DifferentialForm:
    """Return ``L_X form - i_X d form - d i_X form``."""
    lhs = lie_derivative(form, vector, coordinates=coordinates)
    rhs = form.exterior_derivative(coordinates=coordinates).interior_product(vector) + form.interior_product(vector).exterior_derivative(coordinates=coordinates)
    return lhs - rhs


def hodge_star_squared(form: DifferentialForm, metric: ComponentTensor, inverse_metric: ComponentTensor | None = None) -> DifferentialForm:
    """Apply the Hodge star twice."""
    return form.hodge_star(metric, inverse_metric).hodge_star(metric, inverse_metric)


def hodge_star_square_sign(dimension: int, degree: int, *, timelike_directions: int = 0) -> int:
    """Return ``(-1)^(p(n-p)+s)`` for the double-Hodge-star convention."""
    return -1 if (degree * (dimension - degree) + timelike_directions) % 2 else 1


def pullback_form(form: DifferentialForm, transform_matrix: Sequence[Sequence[Scalar]], target_basis: Basis | None = None) -> DifferentialForm:
    """Functional wrapper for covariant pullback of a differential form."""
    return form.pullback(transform_matrix, target_basis)


def form_inner_product(left: DifferentialForm, right: DifferentialForm, metric: ComponentTensor, inverse_metric: ComponentTensor | None = None) -> Scalar:
    """Return the pointwise inner product of two equal-degree forms.

    The convention is ``<alpha,beta> = 1/p! alpha_{i1...ip} beta^{i1...ip}``.
    """
    if left.basis != right.basis or left.degree != right.degree:
        raise TensorKernelError("Form inner product requires forms with the same basis and degree.")
    if metric.basis != left.basis:
        raise TensorKernelError("Metric must live on the same basis as the forms.")
    inv = inverse_metric
    if inv is None:
        from .components import inverse_metric_from_metric
        inv = inverse_metric_from_metric(metric)
    if left.degree == 0:
        return _simplify_scalar(left.component() * right.component())
    from math import factorial
    total = 0
    for low_key in product(range(left.dimension), repeat=left.degree):
        if len(set(low_key)) < len(low_key):
            continue
        raised_sum = 0
        for high_key in product(range(left.dimension), repeat=left.degree):
            if len(set(high_key)) < len(high_key):
                continue
            weight = 1
            for slot in range(left.degree):
                weight *= inv.component(low_key[slot], high_key[slot])
            raised_sum += weight * right.component(*high_key)
        total += left.component(*low_key) * raised_sum
    return _simplify_scalar(total / factorial(left.degree))


def wedge_hodge_inner_product(left: DifferentialForm, right: DifferentialForm, metric: ComponentTensor, inverse_metric: ComponentTensor | None = None) -> DifferentialForm:
    """Return ``left wedge *right`` as a top-degree form."""
    return left.wedge(right.hodge_star(metric, inverse_metric))


def is_closed(form: DifferentialForm, *, coordinates: Sequence[Scalar] | None = None) -> bool:
    """Return True when the stored exterior derivative is structurally zero."""
    return not form.exterior_derivative(coordinates=coordinates).components


def pullback_from_coordinate_functions(
    form: DifferentialForm,
    source_basis: Basis,
    target_coordinates: Sequence[Scalar],
    source_coordinates: Sequence[Scalar] | None = None,
) -> DifferentialForm:
    """Pull back a form using old coordinates expressed in source coordinates.

    ``target_coordinates`` are the coordinates of ``form.basis`` written as
    functions on ``source_basis``.  The induced one-form matrix is
    ``partial x_old_i / partial y_new_j``.
    """
    sp = _require_sympy()
    src = tuple(source_coordinates) if source_coordinates is not None else coordinate_symbols(source_basis.coordinates)
    tgt = tuple(target_coordinates)
    if len(src) != source_basis.dimension or len(tgt) != form.dimension:
        raise TensorKernelError("Coordinate-function pullback has incompatible dimensions.")
    matrix = tuple(tuple(sp.diff(tgt_i, src_j) for src_j in src) for tgt_i in tgt)
    return form.pullback(matrix, source_basis)
