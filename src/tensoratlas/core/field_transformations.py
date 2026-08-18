"""Coordinate-field transformation workflows for scalar, vector, covector, tensor, and density fields between charts."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Mapping, Sequence

from .coordinate_tools import CoordinateMap, coordinate_map_between, standard_coordinate_entry
from .manifolds import TensorKernelError
from .symbolic_arrays import TensorArray, as_tensor_array
from .symbolic_utils import as_matrix as _matrix
from .symbolic_utils import build_nested as _build_nested
from .symbolic_utils import is_zero, iter_indices, normalize_variance
from .symbolic_utils import require_sympy as _require_sympy

Scalar = Any


def _simplify(value: Scalar) -> Scalar:
    """Cheap component simplification for coordinate transformations.

    Full SymPy simplification routines can become extremely expensive on chart
    maps involving ``atan2``, square roots, and absolute values. Coordinate
    transformations are called component-by-component, so keep the default pass
    deliberately bounded. Users can always simplify returned components later.
    """
    try:
        ops = value.count_ops() if hasattr(value, "count_ops") else 0
    except Exception:
        ops = 0
    if ops > 32:
        return value
    sp = _require_sympy("scalar simplification")
    try:
        value = sp.cancel(value)
    except Exception:
        pass
    try:
        if hasattr(value, "count_ops") and value.count_ops() <= 24:
            return sp.cancel(sp.trigsimp(value))
    except Exception:
        pass
    return value


def _substitute_source_to_target(value: Scalar, cmap: CoordinateMap) -> Scalar:
    if cmap.inverse is None:
        raise TensorKernelError("Transforming a field to target coordinates requires an inverse branch.")
    return value.subs(dict(zip(cmap.source_symbols, cmap.inverse))) if hasattr(value, "subs") else value



@dataclass(frozen=True, slots=True)
class FieldTransformationResult:
    """Result of a coordinate-field transformation."""

    field_type: str
    source_coordinates: tuple[str, ...]
    target_coordinates: tuple[str, ...]
    components: Any
    variance: tuple[str | None, ...] = ()
    density_weight: int | float = 0
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)
    convention_metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def as_tensor_array(self) -> TensorArray:
        props = dict(self.metadata)
        props.setdefault("conventions", dict(self.convention_metadata))
        return TensorArray(self.components, self.variance, props)


def _factor_matrix(cmap: CoordinateMap, variance: str, *, substitute: bool = True):
    sp = _require_sympy()
    normalized = normalize_variance(variance)
    if normalized == "up":
        mat = sp.Matrix(cmap.jacobian())
    elif normalized == "down":
        inv = cmap.inverse_jacobian()
        if inv is None:
            raise TensorKernelError("Covariant field transformation requires an inverse Jacobian.")
        mat = sp.Matrix(inv).T  # rows target covariant slots, columns source covariant slots
    else:  # pragma: no cover - normalize_variance handles validation
        raise TensorKernelError("Variance entries must be up/down or contravariant/covariant.")
    if substitute and cmap.inverse is not None:
        mat = mat.applyfunc(lambda value: _substitute_source_to_target(value, cmap))
    return mat


def transform_scalar_field(expression: Scalar, cmap: CoordinateMap, *, density_weight: int | float = 0) -> Scalar:
    """Transform a scalar or scalar density from source to target coordinates."""
    value = _substitute_source_to_target(expression, cmap)
    if density_weight:
        inv_jac = cmap.inverse_jacobian()
        if inv_jac is None:
            raise TensorKernelError("Density transformation requires an inverse Jacobian.")
        det = _require_sympy().Abs(_matrix(inv_jac).det())
        det = _substitute_source_to_target(det, cmap)
        value = det ** density_weight * value
    return _simplify(value)


def transform_tensor_field(components: Any, cmap: CoordinateMap, variance: Sequence[str], *, density_weight: int | float = 0) -> FieldTransformationResult:
    """Transform tensor components from source coordinates to target coordinates.

    Contravariant slots use ``dy^a/dx^i``; covariant slots use
    ``dx^i/dy^a``.  Components are substituted through the inverse branch so the
    result is expressed in target coordinates.  Sparse/zero source components
    are skipped before the dense target array is assembled.
    """
    normalized_variance = tuple(normalize_variance(var) for var in variance)
    arr = as_tensor_array(components, variance=normalized_variance)
    if arr.rank != len(normalized_variance):
        raise TensorKernelError("Variance length must match tensor rank.")
    if any(dim != cmap.source.dimension for dim in arr.dimensions):
        raise TensorKernelError("Tensor component dimensions must match the source coordinate dimension.")
    dim_target = cmap.target.dimension
    factors = tuple(_factor_matrix(cmap, var) for var in normalized_variance)
    inv_det_factor = 1
    density_jacobian = None
    if density_weight:
        inv = cmap.inverse_jacobian()
        if inv is None:
            raise TensorKernelError("Tensor-density transformation requires an inverse Jacobian.")
        density_jacobian = _substitute_source_to_target(_require_sympy().Abs(_matrix(inv).det()), cmap)
        inv_det_factor = density_jacobian ** density_weight

    source_terms: list[tuple[tuple[int, ...], Scalar]] = []
    for source_key in iter_indices(arr.dimensions):
        value = _substitute_source_to_target(arr.component(source_key), cmap)
        if not is_zero(value):
            source_terms.append((source_key, value))

    def getter(target_key: tuple[int, ...]) -> Scalar:
        total = 0
        for source_key, source_value in source_terms:
            coeff = inv_det_factor
            for slot, (target_axis, source_axis) in enumerate(zip(target_key, source_key)):
                factor = factors[slot][target_axis, source_axis]
                if is_zero(factor):
                    coeff = 0
                    break
                coeff *= factor
            if not is_zero(coeff):
                total += coeff * source_value
        return _simplify(total)

    shape = (dim_target,) * arr.rank
    field_type = "tensor_density" if density_weight else ("tensor" if arr.rank != 1 else ("vector" if normalized_variance[0] == "up" else "covector"))
    convention_metadata = {
        "component_basis": "coordinate",
        "density_convention": "target components use |det(d source / d target)|^weight",
        "variance_convention": "contravariant dy/dx; covariant dx/dy",
        "source_chart": cmap.source.name,
        "target_chart": cmap.target.name,
    }
    metadata = {
        "coordinate_map": cmap.name,
        "jacobian_det": cmap.jacobian_determinant(),
        "density_jacobian": density_jacobian,
        "source_nonzero_components": len(source_terms),
    }
    return FieldTransformationResult(
        field_type,
        cmap.source.coordinate_names,
        cmap.target.coordinate_names,
        _build_nested(shape, getter),
        normalized_variance,
        density_weight,
        metadata,
        convention_metadata,
    )


def transform_field(field: Any, source: str | CoordinateMap, target: str | None = None, *, field_type: str = "scalar", variance: Sequence[str] | str | None = None, density_weight: int | float = 0) -> FieldTransformationResult | Scalar:
    """Transform scalar, vector, covector, tensor, or tensor-density fields.

    ``source`` may be a ``CoordinateMap`` or the name of a standard coordinate
    system.  When a name is supplied, ``target`` must be another standard name
    and a catalog transition branch is used.
    """
    if isinstance(source, CoordinateMap):
        cmap = source
    else:
        if target is None:
            raise TensorKernelError("A target coordinate-system name is required.")
        cmap = coordinate_map_between(source, target)
    normalized = field_type.strip().lower().replace("_", "-")
    if normalized in {"scalar", "scalar-density"}:
        return transform_scalar_field(field, cmap, density_weight=density_weight if normalized == "scalar-density" else density_weight)
    if normalized == "vector":
        return transform_tensor_field(field, cmap, ("up",), density_weight=density_weight)
    if normalized == "covector":
        return transform_tensor_field(field, cmap, ("down",), density_weight=density_weight)
    if normalized in {"tensor", "tensor-density"}:
        if variance is None:
            raise TensorKernelError("Tensor field transformation requires a variance sequence.")
        if isinstance(variance, str):
            raise TensorKernelError("Tensor variance must be a sequence for rank greater than one.")
        return transform_tensor_field(field, cmap, variance, density_weight=density_weight)
    raise TensorKernelError(f"Unsupported field type {field_type!r}.")


def transform_vector_field(components: Sequence[Scalar], cmap: CoordinateMap) -> FieldTransformationResult:
    return transform_tensor_field(tuple(components), cmap, ("up",))


def transform_covector_field(components: Sequence[Scalar], cmap: CoordinateMap) -> FieldTransformationResult:
    return transform_tensor_field(tuple(components), cmap, ("down",))


def transform_tensor_density(components: Any, cmap: CoordinateMap, variance: Sequence[str], weight: int | float) -> FieldTransformationResult:
    return transform_tensor_field(components, cmap, variance, density_weight=weight)
