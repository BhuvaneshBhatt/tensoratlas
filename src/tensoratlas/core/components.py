"""Coordinate, basis, frame, and component-tensor realization objects.

This module keeps component-level data separate from the abstract tensor
expression kernel.  It is backend-neutral: component values may be numbers,
SymPy expressions, or any scalar-like object supporting ordinary arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product
from typing import Any, Callable, Mapping, Sequence

from .indices import IndexType
from .manifolds import Manifold, TensorKernelError
from .tensor_heads import TensorHead

Scalar = Any
IndexTuple = tuple[int, ...]


def _as_square_matrix(values: Sequence[Sequence[Scalar]], dimension: int, *, name: str) -> tuple[tuple[Scalar, ...], ...]:
    rows = tuple(tuple(row) for row in values)
    if len(rows) != dimension or any(len(row) != dimension for row in rows):
        raise TensorKernelError(f"{name} must be a {dimension} x {dimension} matrix.")
    return rows


def _zero_like() -> int:
    return 0


def _canonical_component_key(kind: str, key: IndexTuple) -> tuple[int, IndexTuple]:
    """Return sign and canonical key for built-in component symmetries."""
    if kind == "none" or len(key) < 2:
        return 1, key
    if kind == "symmetric":
        return 1, tuple(sorted(key))
    if kind == "antisymmetric":
        if len(set(key)) < len(key):
            return 0, key
        ordered = tuple(sorted(key))
        inversions = 0
        for left in range(len(key)):
            for right in range(left + 1, len(key)):
                if key[left] > key[right]:
                    inversions += 1
        return (-1 if inversions % 2 else 1), ordered
    if kind == "antisym_last2":
        if key[-2] == key[-1]:
            return 0, key
        if key[-1] < key[-2]:
            return -1, key[:-2] + (key[-1], key[-2])
        return 1, key
    if kind in {"riemann", "weyl"} and len(key) == 4:
        a, b, c, d = key
        if a == b or c == d:
            return 0, key
        sign = 1
        first = (a, b)
        second = (c, d)
        if first[1] < first[0]:
            first = (first[1], first[0])
            sign *= -1
        if second[1] < second[0]:
            second = (second[1], second[0])
            sign *= -1
        if second < first:
            first, second = second, first
        return sign, first + second
    return 1, key


def _sum_values(left: Scalar, right: Scalar) -> Scalar:
    value = left + right
    try:
        import sympy as sp  # type: ignore

        return sp.cancel(value)
    except Exception:
        return value


def _simplify_scalar(value: Scalar) -> Scalar:
    try:
        import sympy as sp  # type: ignore

        return sp.cancel(value)
    except Exception:
        return value


@dataclass(frozen=True, slots=True)
class CoordinateSystem:
    """A named coordinate system on a manifold."""

    name: str
    manifold: Manifold
    coordinate_names: tuple[str, ...]
    index_type: IndexType | None = None
    domain: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)
    coordinate_symbols: tuple[Scalar, ...] | None = field(default=None, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise TensorKernelError("Coordinate-system name must be non-empty.")
        if not self.coordinate_names:
            raise TensorKernelError("Coordinate system must contain at least one coordinate.")
        if isinstance(self.manifold.dimension, int) and len(self.coordinate_names) != self.manifold.dimension:
            raise TensorKernelError("Coordinate count must match the manifold dimension.")
        itype = self.index_type or self.manifold.index_type(f"{self.name}_coord")
        if itype.manifold != self.manifold:
            raise TensorKernelError("Coordinate index type must belong to the coordinate-system manifold.")
        if self.coordinate_symbols is None:
            try:
                import sympy as sp  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise TensorKernelError("CoordinateSystem construction requires SymPy when coordinate_symbols are omitted.") from exc
            symbols = tuple(sp.Symbol(name, real=True) for name in self.coordinate_names)
        else:
            symbols = tuple(self.coordinate_symbols)
        if len(symbols) != len(self.coordinate_names):
            raise TensorKernelError("Coordinate-symbol count must match coordinate-name count.")
        object.__setattr__(self, "index_type", itype)
        object.__setattr__(self, "coordinate_names", tuple(self.coordinate_names))
        object.__setattr__(self, "domain", dict(self.domain))
        object.__setattr__(self, "coordinate_symbols", symbols)

    @property
    def dimension(self) -> int:
        return len(self.coordinate_names)

    def coordinate_basis(self, name: str | None = None) -> "Basis":
        """Return the coordinate basis associated with this coordinate system."""
        return Basis(name or f"d/d{self.name}", self, kind="coordinate")


@dataclass(frozen=True, slots=True)
class Basis:
    """A vector basis over a coordinate system.

    ``structure_coefficients`` stores coefficients ``c^i_{jk}`` for
    ``[e_j, e_k] = c^i_{jk} e_i``.  Coordinate bases default to zero structure
    coefficients.
    """

    name: str
    coordinates: CoordinateSystem
    kind: str = "coordinate"
    structure_coefficients: Mapping[tuple[int, int, int], Scalar] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise TensorKernelError("Basis name must be non-empty.")
        if self.kind not in {"coordinate", "frame", "orthonormal", "noncoordinate"}:
            raise TensorKernelError(f"Unsupported basis kind: {self.kind!r}.")
        cleaned: dict[tuple[int, int, int], Scalar] = {}
        for key, value in self.structure_coefficients.items():
            if len(key) != 3:
                raise TensorKernelError("Structure-coefficient keys must have form (upper, lower1, lower2).")
            self._validate_component_index(key)
            if value != 0:
                cleaned[tuple(key)] = value
        object.__setattr__(self, "structure_coefficients", cleaned)

    @property
    def dimension(self) -> int:
        return self.coordinates.dimension

    @property
    def index_type(self) -> IndexType:
        return self.coordinates.index_type

    def structure_coefficient(self, upper: int, lower_left: int, lower_right: int) -> Scalar:
        """Return ``c^upper_{lower_left lower_right}``, enforcing antisymmetry."""
        self._validate_component_index((upper, lower_left, lower_right))
        direct = self.structure_coefficients.get((upper, lower_left, lower_right))
        if direct is not None:
            return direct
        reverse = self.structure_coefficients.get((upper, lower_right, lower_left))
        if reverse is not None:
            return -reverse
        return 0

    def _validate_component_index(self, index: Sequence[int]) -> None:
        for item in index:
            if item < 0 or item >= self.dimension:
                raise TensorKernelError(f"Component index {item!r} is outside basis dimension {self.dimension}.")


@dataclass(frozen=True, slots=True)
class BasisTransform:
    """A linear transformation between bases over the same coordinate system.

    ``matrix[i][j]`` is the coefficient mapping source basis vector ``j`` to
    target basis vector ``i``.  The optional inverse matrix is used for covariant
    slots.
    """

    source: Basis
    target: Basis
    matrix: tuple[tuple[Scalar, ...], ...]
    inverse_matrix: tuple[tuple[Scalar, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.source.coordinates.manifold != self.target.coordinates.manifold:
            raise TensorKernelError("Basis transforms require a shared manifold.")
        if self.source.dimension != self.target.dimension:
            raise TensorKernelError("Basis transforms require equal dimensions.")
        dim = self.source.dimension
        object.__setattr__(self, "matrix", _as_square_matrix(self.matrix, dim, name="Basis transform"))
        if self.inverse_matrix is not None:
            object.__setattr__(self, "inverse_matrix", _as_square_matrix(self.inverse_matrix, dim, name="Inverse basis transform"))

    @classmethod
    def identity(cls, basis: Basis) -> "BasisTransform":
        dim = basis.dimension
        matrix = tuple(tuple(1 if i == j else 0 for j in range(dim)) for i in range(dim))
        return cls(basis, basis, matrix, matrix)

    def matrix_for_variance(self, variance: str) -> tuple[tuple[Scalar, ...], ...]:
        if variance == "up":
            return self.matrix
        if self.inverse_matrix is None:
            raise TensorKernelError("Covariant component transformation requires an inverse matrix.")
        return self.inverse_matrix


@dataclass(frozen=True, slots=True)
class ComponentTensor:
    """Sparse component realization of a tensor head in a chosen basis."""

    head: TensorHead
    basis: Basis
    components: Mapping[IndexTuple, Scalar]
    variance: tuple[str | None, ...] | None = None
    default: Scalar = 0

    def __post_init__(self) -> None:
        variance = tuple(self.variance if self.variance is not None else self.head.variance)
        if len(variance) != self.head.rank:
            raise TensorKernelError("Component variance must match tensor rank.")
        if any(item not in {"up", "down", None} for item in variance):
            raise TensorKernelError("Component variance entries must be 'up', 'down', or None.")
        for itype in self.head.index_types:
            if itype != self.basis.index_type:
                raise TensorKernelError("Component tensor slots must use the basis index type.")
        cleaned: dict[IndexTuple, Scalar] = {}
        for key, value in self.components.items():
            ckey = tuple(key)
            self._validate_key(ckey)
            sign, canonical_key = _canonical_component_key(self.head.symmetry.kind, ckey)
            if sign == 0:
                continue
            stored_value = value if sign == 1 else -value
            if stored_value != self.default:
                if canonical_key in cleaned:
                    previous = cleaned[canonical_key]
                    if _simplify_scalar(previous - stored_value) != 0:
                        raise TensorKernelError(
                            f"Conflicting values supplied for symmetry-equivalent component {ckey!r}."
                        )
                else:
                    cleaned[canonical_key] = stored_value
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "components", cleaned)

    @property
    def rank(self) -> int:
        return self.head.rank

    @property
    def dimension(self) -> int:
        return self.basis.dimension

    @classmethod
    def zero(cls, head: TensorHead, basis: Basis, *, variance: tuple[str | None, ...] | None = None, default: Scalar = 0) -> "ComponentTensor":
        return cls(head, basis, {}, variance=variance, default=default)

    @classmethod
    def from_dense(
        cls,
        head: TensorHead,
        basis: Basis,
        values: Any,
        *,
        variance: tuple[str | None, ...] | None = None,
        default: Scalar = 0,
    ) -> "ComponentTensor":
        """Build a sparse component tensor from nested sequences."""
        components: dict[IndexTuple, Scalar] = {}
        for key in product(range(basis.dimension), repeat=head.rank):
            value = values
            for item in key:
                value = value[item]
            if value != default:
                components[key] = value
        return cls(head, basis, components, variance=variance, default=default)

    @classmethod
    def from_function(
        cls,
        head: TensorHead,
        basis: Basis,
        function: Callable[[IndexTuple], Scalar],
        *,
        variance: tuple[str | None, ...] | None = None,
        default: Scalar = 0,
    ) -> "ComponentTensor":
        components = {key: function(key) for key in product(range(basis.dimension), repeat=head.rank)}
        return cls(head, basis, components, variance=variance, default=default)

    def component(self, *index: int) -> Scalar:
        key = tuple(index)
        self._validate_key(key)
        sign, canonical_key = _canonical_component_key(self.head.symmetry.kind, key)
        if sign == 0:
            return self.default
        value = self.components.get(canonical_key, self.default)
        return value if sign == 1 else -value

    def with_component(self, index: Sequence[int], value: Scalar) -> "ComponentTensor":
        key = tuple(index)
        self._validate_key(key)
        new = dict(self.components)
        if value == self.default:
            new.pop(key, None)
        else:
            new[key] = value
        return ComponentTensor(self.head, self.basis, new, variance=self.variance, default=self.default)

    def to_dense(self) -> Any:
        """Return nested Python lists with all components materialized."""
        def build(prefix: tuple[int, ...], depth: int):
            if depth == self.rank:
                return self.component(*prefix)
            return [build(prefix + (i,), depth + 1) for i in range(self.dimension)]
        if self.rank == 0:
            return self.components.get((), self.default)
        return build((), 0)

    def lowered_with(self, metric: "ComponentTensor", slot: int, *, head: TensorHead | None = None) -> "ComponentTensor":
        """Lower one contravariant slot using a covariant metric."""
        if metric.rank != 2 or metric.variance != ("down", "down"):
            raise TensorKernelError("Lowering requires a covariant rank-two metric tensor.")
        if metric.basis != self.basis:
            raise TensorKernelError("Metric and tensor must use the same basis.")
        if slot < 0 or slot >= self.rank or self.variance[slot] != "up":
            raise TensorKernelError("Can only lower an existing contravariant slot.")
        new_variance = list(self.variance)
        new_variance[slot] = "down"
        new_head = head or TensorHead(self.head.name, self.head.index_types, symmetry=self.head.symmetry, variance=tuple(new_variance), commutative=self.head.commutative)
        out: dict[IndexTuple, Scalar] = {}
        for new_key in product(range(self.dimension), repeat=self.rank):
            total = 0
            for old_i in range(self.dimension):
                old_key = tuple(old_i if pos == slot else value for pos, value in enumerate(new_key))
                total = total + metric.component(new_key[slot], old_i) * self.component(*old_key)
            total = _simplify_scalar(total)
            if total != self.default:
                out[new_key] = total
        return ComponentTensor(new_head, self.basis, out, variance=tuple(new_variance), default=self.default)

    def raised_with(self, inverse_metric: "ComponentTensor", slot: int, *, head: TensorHead | None = None) -> "ComponentTensor":
        """Raise one covariant slot using a contravariant inverse metric."""
        if inverse_metric.rank != 2 or inverse_metric.variance != ("up", "up"):
            raise TensorKernelError("Raising requires a contravariant rank-two inverse metric tensor.")
        if inverse_metric.basis != self.basis:
            raise TensorKernelError("Inverse metric and tensor must use the same basis.")
        if slot < 0 or slot >= self.rank or self.variance[slot] != "down":
            raise TensorKernelError("Can only raise an existing covariant slot.")
        new_variance = list(self.variance)
        new_variance[slot] = "up"
        new_head = head or TensorHead(self.head.name, self.head.index_types, symmetry=self.head.symmetry, variance=tuple(new_variance), commutative=self.head.commutative)
        out: dict[IndexTuple, Scalar] = {}
        for new_key in product(range(self.dimension), repeat=self.rank):
            total = 0
            for old_i in range(self.dimension):
                old_key = tuple(old_i if pos == slot else value for pos, value in enumerate(new_key))
                total = total + inverse_metric.component(new_key[slot], old_i) * self.component(*old_key)
            total = _simplify_scalar(total)
            if total != self.default:
                out[new_key] = total
        return ComponentTensor(new_head, self.basis, out, variance=tuple(new_variance), default=self.default)

    def transform(self, transform: BasisTransform) -> "ComponentTensor":
        """Transform all slots to the target basis."""
        if transform.source != self.basis:
            raise TensorKernelError("Basis transform source must match the component tensor basis.")
        target_head = self.head
        if any(itype != transform.target.index_type for itype in self.head.index_types):
            target_head = TensorHead(
                self.head.name,
                (transform.target.index_type,) * self.rank,
                symmetry=self.head.symmetry,
                variance=self.variance,
                commutative=self.head.commutative,
                role=self.head.role,
            )
        if self.rank == 0:
            return ComponentTensor(target_head, transform.target, dict(self.components), variance=self.variance, default=self.default)
        matrices = tuple(transform.matrix_for_variance(v or "up") for v in self.variance)
        out: dict[IndexTuple, Scalar] = {}
        for new_key in product(range(self.dimension), repeat=self.rank):
            total = 0
            for old_key in product(range(self.dimension), repeat=self.rank):
                coeff = 1
                for slot, (new_i, old_i) in enumerate(zip(new_key, old_key)):
                    coeff = coeff * matrices[slot][new_i][old_i]
                    if coeff == 0:
                        break
                if coeff != 0:
                    total = total + coeff * self.component(*old_key)
            if total != self.default:
                out[new_key] = total
        return ComponentTensor(target_head, transform.target, out, variance=self.variance, default=self.default)

    def _validate_key(self, key: IndexTuple) -> None:
        if len(key) != self.rank:
            raise TensorKernelError(f"Component key {key!r} has rank {len(key)}, expected {self.rank}.")
        for item in key:
            if item < 0 or item >= self.dimension:
                raise TensorKernelError(f"Component index {item!r} is outside dimension {self.dimension}.")


@dataclass(frozen=True, slots=True)
class ConnectionCoefficients:
    """Connection coefficients in a specified basis.

    Components are stored as ``Gamma^i_{jk}``.  The torsion helper subtracts
    structure coefficients so non-coordinate frames are handled correctly.
    """

    basis: Basis
    components: Mapping[tuple[int, int, int], Scalar]
    default: Scalar = 0

    def __post_init__(self) -> None:
        cleaned: dict[tuple[int, int, int], Scalar] = {}
        for key, value in self.components.items():
            if len(key) != 3:
                raise TensorKernelError("Connection-coefficient keys must have form (upper, lower1, lower2).")
            self.basis._validate_component_index(key)
            if value != self.default:
                cleaned[tuple(key)] = value
        object.__setattr__(self, "components", cleaned)

    def coefficient(self, upper: int, lower_left: int, lower_right: int) -> Scalar:
        self.basis._validate_component_index((upper, lower_left, lower_right))
        return self.components.get((upper, lower_left, lower_right), self.default)

    def torsion_component(self, upper: int, lower_left: int, lower_right: int) -> Scalar:
        """Return ``T^upper_{lower_left lower_right}``."""
        return (
            self.coefficient(upper, lower_left, lower_right)
            - self.coefficient(upper, lower_right, lower_left)
            - self.basis.structure_coefficient(upper, lower_left, lower_right)
        )

    def torsion_tensor(self, head: TensorHead | None = None) -> ComponentTensor:
        itype = self.basis.index_type
        tensor_head = head or TensorHead.torsion("T", itype)
        comps = {
            key: self.torsion_component(*key)
            for key in product(range(self.basis.dimension), repeat=3)
        }
        return ComponentTensor(tensor_head, self.basis, comps, variance=("up", "down", "down"), default=self.default)


def metric_component_tensor(name: str, basis: Basis, values: Sequence[Sequence[Scalar]]) -> ComponentTensor:
    """Construct a covariant metric component tensor in ``basis``."""
    head = TensorHead.metric(name, basis.index_type)
    matrix = _as_square_matrix(values, basis.dimension, name="Metric components")
    return ComponentTensor.from_dense(head, basis, matrix, variance=("down", "down"))


def inverse_metric_component_tensor(name: str, basis: Basis, values: Sequence[Sequence[Scalar]]) -> ComponentTensor:
    """Construct a contravariant inverse-metric component tensor in ``basis``."""
    head = TensorHead.inverse_metric(name, basis.index_type)
    matrix = _as_square_matrix(values, basis.dimension, name="Inverse metric components")
    return ComponentTensor.from_dense(head, basis, matrix, variance=("up", "up"))

@dataclass(frozen=True, slots=True)
class CoordinateTransform:
    """Coordinate change represented by an explicit Jacobian.

    ``jacobian[i][j]`` is ``d target_i / d source_j``.  The inverse Jacobian
    is required for covariant slot transformation.  The object intentionally
    stores matrices rather than deriving them from symbolic maps so callers can
    use exact, numeric, or externally computed Jacobians.
    """

    source: CoordinateSystem
    target: CoordinateSystem
    jacobian: tuple[tuple[Scalar, ...], ...]
    inverse_jacobian: tuple[tuple[Scalar, ...], ...] | None = None

    def __post_init__(self) -> None:
        if self.source.manifold != self.target.manifold:
            raise TensorKernelError("Coordinate transforms require a shared manifold.")
        if self.source.dimension != self.target.dimension:
            raise TensorKernelError("Coordinate transforms require equal dimensions.")
        dim = self.source.dimension
        object.__setattr__(self, "jacobian", _as_square_matrix(self.jacobian, dim, name="Coordinate Jacobian"))
        if self.inverse_jacobian is not None:
            object.__setattr__(self, "inverse_jacobian", _as_square_matrix(self.inverse_jacobian, dim, name="Inverse coordinate Jacobian"))

    def basis_transform(self, *, source_name: str | None = None, target_name: str | None = None) -> BasisTransform:
        """Return the induced transform between coordinate bases."""
        return BasisTransform(
            self.source.coordinate_basis(source_name),
            self.target.coordinate_basis(target_name),
            self.jacobian,
            self.inverse_jacobian,
        )

@dataclass(frozen=True, slots=True)
class Coframe:
    """Dual covector basis associated with a vector basis."""

    basis: Basis
    name: str | None = None

    @property
    def dimension(self) -> int:
        return self.basis.dimension

    @property
    def coordinates(self) -> CoordinateSystem:
        return self.basis.coordinates

    @property
    def index_type(self) -> IndexType:
        return self.basis.index_type

    def pairing(self, vector_index: int, covector_index: int) -> int:
        """Return theta^covector_index(e_vector_index)."""
        self.basis._validate_component_index((vector_index,))
        self.basis._validate_component_index((covector_index,))
        return 1 if vector_index == covector_index else 0


def dual_coframe(basis: Basis, *, name: str | None = None) -> Coframe:
    """Return the formal dual coframe of ``basis``."""
    return Coframe(basis, name=name or f"{basis.name}^*")


@dataclass(frozen=True, slots=True)
class Vielbein:
    """Frame/coframe data relating an internal frame to a coordinate basis."""

    coordinate_basis: Basis
    frame_basis: Basis
    frame_to_coordinate: tuple[tuple[Scalar, ...], ...]
    coordinate_to_frame: tuple[tuple[Scalar, ...], ...]
    signature: tuple[Scalar, ...] | None = None

    def __post_init__(self) -> None:
        if self.coordinate_basis.kind != "coordinate":
            raise TensorKernelError("Vielbein requires a coordinate source basis.")
        if self.coordinate_basis.coordinates.manifold != self.frame_basis.coordinates.manifold:
            raise TensorKernelError("Vielbein bases must live on the same manifold.")
        dim = self.coordinate_basis.dimension
        object.__setattr__(self, "frame_to_coordinate", _as_square_matrix(self.frame_to_coordinate, dim, name="Frame-to-coordinate matrix"))
        object.__setattr__(self, "coordinate_to_frame", _as_square_matrix(self.coordinate_to_frame, dim, name="Coordinate-to-frame matrix"))
        if self.signature is not None and len(self.signature) != dim:
            raise TensorKernelError("Metric signature length must match the basis dimension.")
        for left in range(dim):
            for right in range(dim):
                total = 0
                for mid in range(dim):
                    total += self.coordinate_to_frame[left][mid] * self.frame_to_coordinate[mid][right]
                if _simplify_scalar(total - (1 if left == right else 0)) != 0:
                    raise TensorKernelError("Vielbein matrices must be mutual inverses.")

    @property
    def dimension(self) -> int:
        return self.coordinate_basis.dimension

    def coordinate_to_frame_transform(self) -> BasisTransform:
        return BasisTransform(self.coordinate_basis, self.frame_basis, self.coordinate_to_frame, self.frame_to_coordinate)

    def frame_to_coordinate_transform(self) -> BasisTransform:
        return BasisTransform(self.frame_basis, self.coordinate_basis, self.frame_to_coordinate, self.coordinate_to_frame)

    def metric_from_signature(self, *, name: str = "g") -> ComponentTensor:
        """Build coordinate metric components from an orthonormal-frame signature."""
        if self.signature is None:
            raise TensorKernelError("metric_from_signature requires an explicit signature.")
        values: list[list[Scalar]] = []
        for mu in range(self.dimension):
            row: list[Scalar] = []
            for nu in range(self.dimension):
                total = 0
                for internal in range(self.dimension):
                    total += self.signature[internal] * self.coordinate_to_frame[internal][mu] * self.coordinate_to_frame[internal][nu]
                row.append(_simplify_scalar(total))
            values.append(row)
        return metric_component_tensor(name, self.coordinate_basis, values)


@dataclass(frozen=True, slots=True)
class SpinConnectionCoefficients:
    """Spin connection coefficients omega^a{}_{b mu} in a frame."""

    vielbein: Vielbein
    components: Mapping[tuple[int, int, int], Scalar]
    default: Scalar = 0

    def __post_init__(self) -> None:
        cleaned: dict[tuple[int, int, int], Scalar] = {}
        for key, value in self.components.items():
            if len(key) != 3:
                raise TensorKernelError("Spin-connection keys must have form (frame_up, frame_down, coordinate_down).")
            a, b, mu = key
            self.vielbein.frame_basis._validate_component_index((a, b))
            self.vielbein.coordinate_basis._validate_component_index((mu,))
            if value != self.default:
                cleaned[tuple(key)] = value
        object.__setattr__(self, "components", cleaned)

    def coefficient(self, frame_up: int, frame_down: int, coordinate_down: int) -> Scalar:
        self.vielbein.frame_basis._validate_component_index((frame_up, frame_down))
        self.vielbein.coordinate_basis._validate_component_index((coordinate_down,))
        return self.components.get((frame_up, frame_down, coordinate_down), self.default)


def spin_connection_from_vielbein(vielbein: Vielbein, connection: ConnectionCoefficients, *, coordinates: Sequence[Scalar] | None = None) -> SpinConnectionCoefficients:
    """Compute omega^a{}_{b mu} from a vielbein and coordinate connection."""
    if connection.basis != vielbein.coordinate_basis:
        raise TensorKernelError("Spin connection construction requires the coordinate-basis connection for the vielbein.")
    sp = _require_sympy()
    coords = tuple(coordinates) if coordinates is not None else coordinate_symbols(vielbein.coordinate_basis.coordinates)
    if len(coords) != vielbein.dimension:
        raise TensorKernelError("Coordinate list length must match the vielbein dimension.")
    comps: dict[tuple[int, int, int], Scalar] = {}
    for frame_up, frame_down, coord_mu in product(range(vielbein.dimension), repeat=3):
        total = 0
        for coord_nu in range(vielbein.dimension):
            inner = sp.diff(vielbein.frame_to_coordinate[coord_nu][frame_down], coords[coord_mu])
            for coord_rho in range(vielbein.dimension):
                inner += connection.coefficient(coord_nu, coord_mu, coord_rho) * vielbein.frame_to_coordinate[coord_rho][frame_down]
            total += vielbein.coordinate_to_frame[frame_up][coord_nu] * inner
        total = _simplify_scalar(total)
        if total != 0:
            comps[(frame_up, frame_down, coord_mu)] = total
    return SpinConnectionCoefficients(vielbein, comps)


def covariant_derivative_components(tensor: ComponentTensor, connection: ConnectionCoefficients, *, coordinates: Sequence[Scalar] | None = None, head: TensorHead | None = None) -> ComponentTensor:
    """Return coordinate covariant derivative components with a final derivative slot."""
    if tensor.basis != connection.basis:
        raise TensorKernelError("Tensor and connection must use the same basis.")
    if connection.basis.kind != "coordinate":
        raise TensorKernelError("Component covariant derivatives are currently implemented for coordinate bases.")
    sp = _require_sympy()
    coords = tuple(coordinates) if coordinates is not None else coordinate_symbols(tensor.basis.coordinates)
    dim = tensor.dimension
    if len(coords) != dim:
        raise TensorKernelError("Coordinate list length must match the basis dimension.")
    variance = tuple(tensor.variance) + ("down",)
    tensor_head = head or TensorHead(f"D{tensor.head.name}", tensor.head.index_types + (tensor.basis.index_type,), variance=variance, commutative=tensor.head.commutative)
    out: dict[IndexTuple, Scalar] = {}
    for base_key in product(range(dim), repeat=tensor.rank):
        for deriv in range(dim):
            total = sp.diff(tensor.component(*base_key), coords[deriv])
            for slot, slot_variance in enumerate(tensor.variance):
                for mid in range(dim):
                    replaced = tuple(mid if pos == slot else item for pos, item in enumerate(base_key))
                    if slot_variance == "up":
                        total += connection.coefficient(base_key[slot], deriv, mid) * tensor.component(*replaced)
                    elif slot_variance == "down":
                        total -= connection.coefficient(mid, deriv, base_key[slot]) * tensor.component(*replaced)
            total = _simplify_scalar(total)
            if total != 0:
                out[base_key + (deriv,)] = total
    return ComponentTensor(tensor_head, tensor.basis, out, variance=variance)


def _require_sympy():
    try:
        import sympy as sp  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only without SymPy
        raise TensorKernelError("This component calculation requires SymPy-compatible scalar expressions.") from exc
    return sp


def coordinate_symbols(coordinates: CoordinateSystem) -> tuple[Scalar, ...]:
    """Return SymPy symbols for the coordinate names."""
    sp = _require_sympy()
    return tuple(sp.Symbol(name) for name in coordinates.coordinate_names)


def inverse_metric_from_metric(metric: ComponentTensor, *, name: str = "ginv") -> ComponentTensor:
    """Compute inverse metric components with SymPy matrix inversion."""
    if metric.rank != 2 or metric.variance != ("down", "down"):
        raise TensorKernelError("inverse_metric_from_metric expects a covariant metric tensor.")
    sp = _require_sympy()
    dense = sp.Matrix(metric.to_dense())
    inverse = dense.inv()
    values = tuple(tuple(_simplify_scalar(inverse[i, j]) for j in range(metric.dimension)) for i in range(metric.dimension))
    return inverse_metric_component_tensor(name, metric.basis, values)


def christoffel_symbols_from_metric(
    metric: ComponentTensor,
    inverse_metric: ComponentTensor | None = None,
    *,
    coordinates: Sequence[Scalar] | None = None,
) -> ConnectionCoefficients:
    """Compute Levi-Civita connection coefficients from a coordinate metric."""
    if metric.rank != 2 or metric.variance != ("down", "down"):
        raise TensorKernelError("Christoffel construction expects a covariant metric tensor.")
    if metric.basis.kind != "coordinate":
        raise TensorKernelError("Metric-derived Christoffel symbols are currently implemented for coordinate bases.")
    inv = inverse_metric or inverse_metric_from_metric(metric)
    if inv.basis != metric.basis or inv.variance != ("up", "up"):
        raise TensorKernelError("Inverse metric must be contravariant and use the same basis.")
    sp = _require_sympy()
    coords = tuple(coordinates) if coordinates is not None else coordinate_symbols(metric.basis.coordinates)
    if len(coords) != metric.dimension:
        raise TensorKernelError("Coordinate list length must match the basis dimension.")
    dim = metric.dimension
    comps: dict[tuple[int, int, int], Scalar] = {}
    for upper, lower_left, lower_right in product(range(dim), repeat=3):
        total = 0
        for ell in range(dim):
            total += inv.component(upper, ell) * (
                sp.diff(metric.component(lower_right, ell), coords[lower_left])
                + sp.diff(metric.component(lower_left, ell), coords[lower_right])
                - sp.diff(metric.component(lower_left, lower_right), coords[ell])
            ) / 2
        total = _simplify_scalar(total)
        if total != 0:
            comps[(upper, lower_left, lower_right)] = total
    return ConnectionCoefficients(metric.basis, comps)


def riemann_component_tensor(
    connection: ConnectionCoefficients,
    *,
    coordinates: Sequence[Scalar] | None = None,
    head: TensorHead | None = None,
) -> ComponentTensor:
    """Compute mixed Riemann components ``R^i{}_{jkl}`` for a coordinate basis."""
    if connection.basis.kind != "coordinate":
        raise TensorKernelError("Riemann construction from coefficients is currently implemented for coordinate bases.")
    sp = _require_sympy()
    coords = tuple(coordinates) if coordinates is not None else coordinate_symbols(connection.basis.coordinates)
    dim = connection.basis.dimension
    if len(coords) != dim:
        raise TensorKernelError("Coordinate list length must match the basis dimension.")
    tensor_head = head or TensorHead.curvature("Riemann", connection.basis.index_type)
    comps: dict[IndexTuple, Scalar] = {}
    for upper, lower, left, right in product(range(dim), repeat=4):
        total = sp.diff(connection.coefficient(upper, lower, right), coords[left])
        total -= sp.diff(connection.coefficient(upper, lower, left), coords[right])
        for mid in range(dim):
            total += connection.coefficient(upper, mid, left) * connection.coefficient(mid, lower, right)
            total -= connection.coefficient(upper, mid, right) * connection.coefficient(mid, lower, left)
        total = _simplify_scalar(total)
        if total != 0:
            comps[(upper, lower, left, right)] = total
    return ComponentTensor(tensor_head, connection.basis, comps, variance=("up", "down", "down", "down"))


def lower_riemann_tensor(riemann: ComponentTensor, metric: ComponentTensor, *, head: TensorHead | None = None) -> ComponentTensor:
    """Lower the leading slot of a mixed Riemann tensor."""
    tensor_head = head or TensorHead.riemann("RiemannDown", riemann.basis.index_type, variance=("down", "down", "down", "down"))
    return riemann.lowered_with(metric, 0, head=tensor_head)


def ricci_component_tensor(riemann: ComponentTensor, *, head: TensorHead | None = None) -> ComponentTensor:
    """Contract ``R^i{}_{jil}`` to the covariant Ricci tensor."""
    if riemann.rank != 4 or riemann.variance != ("up", "down", "down", "down"):
        raise TensorKernelError("Ricci contraction expects mixed Riemann components R^i{}_{jkl}.")
    dim = riemann.dimension
    tensor_head = head or TensorHead.ricci("Ric", riemann.basis.index_type)
    comps: dict[IndexTuple, Scalar] = {}
    for lower, right in product(range(dim), repeat=2):
        total = 0
        for upper in range(dim):
            total += riemann.component(upper, lower, upper, right)
        total = _simplify_scalar(total)
        if total != 0:
            comps[(lower, right)] = total
    return ComponentTensor(tensor_head, riemann.basis, comps, variance=("down", "down"))


def scalar_curvature_component(ricci: ComponentTensor, inverse_metric: ComponentTensor) -> Scalar:
    """Contract inverse metric and Ricci tensor to the scalar curvature."""
    if ricci.basis != inverse_metric.basis:
        raise TensorKernelError("Ricci tensor and inverse metric must use the same basis.")
    if ricci.variance != ("down", "down") or inverse_metric.variance != ("up", "up"):
        raise TensorKernelError("Scalar curvature expects covariant Ricci and contravariant inverse metric.")
    total = 0
    for left, right in product(range(ricci.dimension), repeat=2):
        total += inverse_metric.component(left, right) * ricci.component(left, right)
    return _simplify_scalar(total)


def einstein_component_tensor(
    ricci: ComponentTensor,
    metric: ComponentTensor,
    scalar_curvature: Scalar,
    *,
    head: TensorHead | None = None,
) -> ComponentTensor:
    """Compute covariant Einstein tensor components ``G_ab = Ric_ab - g_ab R/2``."""
    if ricci.basis != metric.basis:
        raise TensorKernelError("Ricci tensor and metric must use the same basis.")
    tensor_head = head or TensorHead.einstein("G", ricci.basis.index_type)
    comps: dict[IndexTuple, Scalar] = {}
    for left, right in product(range(ricci.dimension), repeat=2):
        value = _simplify_scalar(ricci.component(left, right) - metric.component(left, right) * scalar_curvature / 2)
        if value != 0:
            comps[(left, right)] = value
    return ComponentTensor(tensor_head, ricci.basis, comps, variance=("down", "down"))


@dataclass(frozen=True, slots=True)
class MetricGeometry:
    """Cached component geometry derived from a coordinate-basis metric."""

    metric: ComponentTensor
    inverse_metric: ComponentTensor
    connection: ConnectionCoefficients
    riemann: ComponentTensor
    ricci: ComponentTensor
    scalar_curvature: Scalar
    einstein: ComponentTensor


def metric_geometry(metric: ComponentTensor, *, coordinates: Sequence[Scalar] | None = None) -> MetricGeometry:
    """Compute and cache standard Levi-Civita curvature objects for a metric."""
    inverse = inverse_metric_from_metric(metric)
    connection = christoffel_symbols_from_metric(metric, inverse, coordinates=coordinates)
    riemann = riemann_component_tensor(connection, coordinates=coordinates)
    ricci = ricci_component_tensor(riemann)
    scalar = scalar_curvature_component(ricci, inverse)
    einstein = einstein_component_tensor(ricci, metric, scalar)
    return MetricGeometry(metric, inverse, connection, riemann, ricci, scalar, einstein)


def _permutation_parity_from_values(values: tuple[int, ...]) -> int:
    inversions = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left] > values[right]:
                inversions += 1
    return -1 if inversions % 2 else 1


def levi_civita_symbol_component_tensor(
    name: str,
    basis: Basis,
    *,
    variance: tuple[str, ...] | None = None,
    orientation: int = 1,
) -> ComponentTensor:
    """Return sparse Levi-Civita symbol components in ``basis``.

    This is the alternating symbol, not the metric volume tensor.  Raising or
    lowering with a metric, including determinant factors, is a separate
    operation controlled by the convention layer.
    """

    if orientation not in {-1, 1}:
        raise TensorKernelError("Levi-Civita orientation must be +1 or -1.")
    use_variance = tuple(variance) if variance is not None else ("down",) * basis.dimension
    if len(use_variance) != basis.dimension or len(set(use_variance)) != 1:
        raise TensorKernelError("Levi-Civita symbol variance must be all-up or all-down with rank equal to dimension.")
    head = TensorHead.epsilon(name, basis.index_type, rank=basis.dimension, variance=use_variance[0])
    comps: dict[IndexTuple, Scalar] = {}
    for key in product(range(basis.dimension), repeat=basis.dimension):
        if len(set(key)) != basis.dimension:
            continue
        comps[key] = orientation * _permutation_parity_from_values(tuple(key))
    return ComponentTensor(head, basis, comps, variance=use_variance)


def generalized_delta_component_tensor(name: str, basis: Basis, order: int) -> ComponentTensor:
    """Return components of ``delta^{a1...ak}_{b1...bk}``."""

    if order <= 0 or order > basis.dimension:
        raise TensorKernelError("Generalized-delta order must be between 1 and the basis dimension.")
    head = TensorHead.generalized_delta(name, basis.index_type, order)
    comps: dict[IndexTuple, Scalar] = {}
    for uppers in product(range(basis.dimension), repeat=order):
        for lowers in product(range(basis.dimension), repeat=order):
            if len(set(uppers)) < order or len(set(lowers)) < order:
                continue
            value = 0
            for perm in permutations(range(order)):
                sign = _permutation_parity_from_values(tuple(perm))
                if all(uppers[pos] == lowers[perm[pos]] for pos in range(order)):
                    value += sign
            if value:
                comps[tuple(uppers + lowers)] = value
    return ComponentTensor(head, basis, comps, variance=("up",) * order + ("down",) * order)
