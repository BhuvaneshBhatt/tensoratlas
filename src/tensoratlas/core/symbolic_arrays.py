"""Symbolic array tensor operations analogous to common CAS tensor primitives."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Mapping, Sequence

from .manifolds import TensorKernelError
from tensoratlas.errors import ContractionError, TensorShapeError
from tensoratlas.validation import ValidationReport, invalid_report, valid_report
from .symbolic_utils import build_nested as _build_nested
from .symbolic_utils import cancel_safely as _simplify
from .symbolic_utils import get_nested as _get_nested
from .symbolic_utils import iter_indices
from .symbolic_utils import nested_shape as _nested_shape

Scalar = Any
Index = tuple[int, ...]

_CONTRACT_PAIR_MESSAGE = (
    "tensor_contract expected a pair like (1, 2) or a sequence of pairs "
    "like ((0, 2), (1, 3))."
)


@dataclass(frozen=True, slots=True)
class TensorArray:
    """A rectangular symbolic tensor array with lightweight property metadata."""

    components: Any
    variance: tuple[str | None, ...] = ()
    properties: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        shape = _nested_shape(self.components)
        if self.variance and len(self.variance) != len(shape):
            raise TensorShapeError("Variance length must match tensor rank.")
        if any(item not in {"up", "down", None} for item in self.variance):
            raise TensorShapeError("Variance entries must be 'up', 'down', or None.")
        object.__setattr__(self, "variance", tuple(self.variance) if self.variance else tuple(None for _ in shape))
        object.__setattr__(self, "properties", dict(self.properties))

    @property
    def dimensions(self) -> tuple[int, ...]:
        return _nested_shape(self.components)

    @property
    def rank(self) -> int:
        return len(self.dimensions)

    @property
    def shape(self) -> tuple[int, ...]:
        """Alias for :attr:`dimensions` for array-oriented users."""
        return self.dimensions

    @property
    def values(self) -> Any:
        """Alias for the nested component array."""
        return self.components

    def to_sympy_array(self):
        """Return components as a SymPy ``Array``."""
        import sympy as sp
        return sp.Array(self.components)

    @classmethod
    def from_sympy_array(cls, array: Any, *, variance: Sequence[str | None] | None = None, properties: Mapping[str, Any] | None = None) -> "TensorArray":
        """Build a :class:`TensorArray` from a SymPy array-like object."""
        try:
            components = array.tolist()
        except AttributeError:
            components = array
        return cls(components, tuple(variance or ()), dict(properties or {}))

    def summary(self) -> dict[str, Any]:
        """Return rank, shape, variance, and metadata for notebook display."""
        return {"rank": self.rank, "shape": self.dimensions, "variance": self.variance, "properties": dict(self.properties)}

    def validation_report(self) -> ValidationReport:
        """Return structured validation diagnostics without raising."""
        if self.variance and len(self.variance) != self.rank:
            return invalid_report("variance length must match tensor rank")
        return valid_report()

    def validate(self) -> bool:
        """Validate rectangular shape and variance metadata."""
        report = self.validation_report()
        if not report.ok:
            raise TensorShapeError("; ".join(report.errors))
        return True

    def tensor_product(self, *others: Any) -> "TensorArray":
        """Method form of :func:`tensor_product`."""
        return tensor_product(self, *others)

    def contract(self, *pairs: Any) -> "TensorArray":
        """Method form of :func:`tensor_contract`.

        ``T.contract(1, 2)`` and ``T.contract((1, 2))`` are both accepted.
        """
        if len(pairs) == 2 and all(isinstance(item, int) for item in pairs):
            pair_arg = (int(pairs[0]), int(pairs[1]))
        elif len(pairs) == 1:
            pair_arg = pairs[0]
        else:
            pair_arg = pairs
        return tensor_contract(self, pair_arg)

    def component(self, key: Index) -> Scalar:
        if len(key) != self.rank:
            raise TensorShapeError("Component key length must match tensor rank.")
        return _get_nested(self.components, key)

    def with_components(self, components: Any, *, variance: Sequence[str | None] | None = None, properties: Mapping[str, Any] | None = None) -> "TensorArray":
        return TensorArray(
            components,
            tuple(self.variance if variance is None else variance),
            dict(self.properties if properties is None else properties),
        )


def as_tensor_array(value: TensorArray | Any, *, variance: Sequence[str | None] | None = None, properties: Mapping[str, Any] | None = None) -> TensorArray:
    if isinstance(value, TensorArray):
        return value
    return TensorArray(value, tuple(variance or ()), dict(properties or {}))


def tensor_dimensions(tensor: TensorArray | Any) -> tuple[int, ...]:
    """Return dimensions of a rectangular symbolic tensor array."""
    return as_tensor_array(tensor).dimensions


def tensor_properties(tensor: TensorArray | Any) -> dict[str, Any]:
    """Return propagated symbolic-array metadata."""
    arr = as_tensor_array(tensor)
    return {"rank": arr.rank, "dimensions": arr.dimensions, "variance": arr.variance, **dict(arr.properties)}


def tensor_product(*tensors: TensorArray | Any) -> TensorArray:
    """Return the outer product of tensor arrays."""
    arrays = tuple(as_tensor_array(tensor) for tensor in tensors)
    if not arrays:
        return TensorArray(1, ())
    shape = tuple(dim for arr in arrays for dim in arr.dimensions)
    variance = tuple(var for arr in arrays for var in arr.variance)

    def getter(key: Index) -> Scalar:
        offset = 0
        value = 1
        for arr in arrays:
            part = key[offset: offset + arr.rank]
            value *= arr.component(part)
            offset += arr.rank
        return _simplify(value)

    properties = {"operation": "tensor_product", "factors": len(arrays)}
    return TensorArray(_build_nested(shape, getter), variance, properties)


def tensor_transpose(tensor: TensorArray | Any, permutation: Sequence[int]) -> TensorArray:
    """Permute tensor axes."""
    arr = as_tensor_array(tensor)
    perm = tuple(permutation)
    if sorted(perm) != list(range(arr.rank)):
        raise TensorKernelError("Tensor transpose permutation must permute all axes.")
    old_shape = arr.dimensions
    new_shape = tuple(old_shape[i] for i in perm)
    inverse = tuple(perm.index(i) for i in range(arr.rank))

    def getter(new_key: Index) -> Scalar:
        old_key = tuple(new_key[inverse[i]] for i in range(arr.rank))
        return arr.component(old_key)

    props = dict(arr.properties)
    props["last_transpose"] = perm
    return TensorArray(_build_nested(new_shape, getter), tuple(arr.variance[i] for i in perm), props)


def _normalize_contract_pairs(pairs: Sequence[tuple[int, int]] | tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Normalize one or more contraction pairs.

    ``tensor_contract(A, (0, 1))`` is interpreted as one contraction
    pair, while ``tensor_contract(A, ((0, 1), (2, 3)))`` contracts two
    independent pairs.  Strings and malformed sequences are rejected with a
    domain error instead of leaking a Python unpacking error.
    """
    if isinstance(pairs, (str, bytes)):
        raise ContractionError(_CONTRACT_PAIR_MESSAGE)
    try:
        items = tuple(pairs)
    except TypeError as exc:
        raise ContractionError(_CONTRACT_PAIR_MESSAGE) from exc
    if len(items) == 2 and all(isinstance(item, int) for item in items):
        return ((int(items[0]), int(items[1])),)
    normalized: list[tuple[int, int]] = []
    for item in items:
        if isinstance(item, (str, bytes)):
            raise ContractionError(_CONTRACT_PAIR_MESSAGE)
        try:
            a, b = item
        except (TypeError, ValueError) as exc:
            raise ContractionError(_CONTRACT_PAIR_MESSAGE) from exc
        normalized.append((int(a), int(b)))
    return tuple(normalized)


def tensor_contract(tensor: TensorArray | Any, pairs: Sequence[tuple[int, int]] | tuple[int, int]) -> TensorArray:
    """Contract pairs of axes by summing equal component indices.

    Parameters
    ----------
    tensor:
        Rectangular tensor array or object coercible to :class:`TensorArray`.
    pairs:
        Either a single axis pair, such as ``(0, 1)``, or a sequence of
        axis pairs, such as ``((0, 2), (1, 3))``.
    """
    arr = as_tensor_array(tensor)
    rank = arr.rank
    cleaned = _normalize_contract_pairs(pairs)
    used: set[int] = set()
    for a, b in cleaned:
        if a == b or a < 0 or b < 0 or a >= rank or b >= rank:
            raise ContractionError(f"Invalid contraction axis pair {(a, b)} for rank-{rank} tensor.")
        if a in used or b in used:
            raise ContractionError("An axis can occur in at most one contraction pair.")
        if arr.dimensions[a] != arr.dimensions[b]:
            raise ContractionError(f"Contracted axes {(a, b)} must have equal dimensions, got {arr.dimensions[a]} and {arr.dimensions[b]}.")
        used.update({a, b})
    free_axes = tuple(i for i in range(rank) if i not in used)
    free_shape = tuple(arr.dimensions[i] for i in free_axes)
    pair_ranges = tuple(range(arr.dimensions[a]) for a, _ in cleaned)

    def getter(free_key: Index) -> Scalar:
        base: dict[int, int] = dict(zip(free_axes, free_key))
        total = 0
        for contracted_values in product(*pair_ranges) if pair_ranges else ((),):
            key_map = dict(base)
            for (a, b), value in zip(cleaned, contracted_values):
                key_map[a] = value
                key_map[b] = value
            full_key = tuple(key_map[i] for i in range(rank))
            total += arr.component(full_key)
        return _simplify(total)

    props = dict(arr.properties)
    props["last_contract"] = cleaned
    return TensorArray(_build_nested(free_shape, getter), tuple(arr.variance[i] for i in free_axes), props)


def tensor_map_components(function, tensor: TensorArray | Any) -> TensorArray:
    """Map a scalar function over all tensor components, preserving metadata."""
    arr = as_tensor_array(tensor)
    return TensorArray(_build_nested(arr.dimensions, lambda key: function(arr.component(key))), arr.variance, arr.properties)
