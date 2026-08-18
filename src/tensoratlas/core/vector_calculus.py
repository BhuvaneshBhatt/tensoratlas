"""Polished vector-calculus APIs for coordinate component workflows."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Mapping, Sequence

from .manifolds import TensorKernelError
from .symbolic_utils import normalize_variance

Scalar = Any


@dataclass(frozen=True, slots=True)
class VectorCalculusResult:
    """Result wrapper carrying vector-calculus convention metadata."""

    operator: str
    components: Any
    coordinates: tuple[Scalar, ...]
    convention_metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)


def _vector_calculus_conventions(operator: str, coordinates: Sequence[Scalar], convention: str, *, variance: Any = None) -> dict[str, Any]:
    return {
        "operator": operator,
        "component_basis": "coordinate",
        "coordinates": tuple(str(coord) for coord in coordinates),
        "convention": convention,
        "variance": variance,
    }


def _require_sympy():
    try:
        import sympy as sp  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise TensorKernelError("Vector-calculus operations require SymPy.") from exc
    return sp


def _matrix(value: Any):
    sp = _require_sympy()
    return sp.Matrix(value)


def _simplify(value: Scalar) -> Scalar:
    sp = _require_sympy()
    try:
        return sp.cancel(value)
    except Exception:
        return value


def _inverse_metric(metric=None, inverse_metric=None):
    if inverse_metric is not None:
        return _matrix(inverse_metric)
    if metric is None:
        return None
    return _matrix(metric).inv()


def _sqrt_metric_det(metric=None, sqrt_metric_det=None):
    if sqrt_metric_det is not None:
        return sqrt_metric_det
    if metric is None:
        return 1
    sp = _require_sympy()
    return sp.sqrt(sp.Abs(_matrix(metric).det()))


def levi_civita_connection(metric: Sequence[Sequence[Scalar]], coordinates: Sequence[Scalar], inverse_metric=None) -> tuple[tuple[tuple[Scalar, ...], ...], ...]:
    """Return Christoffel symbols Gamma^i_{jk} for a coordinate metric."""
    sp = _require_sympy()
    coords = tuple(coordinates)
    g = _matrix(metric)
    ginv = _inverse_metric(metric, inverse_metric)
    if g.rows != len(coords) or g.cols != len(coords):
        raise TensorKernelError("Metric shape must match the coordinate dimension.")
    dim = len(coords)
    return tuple(
        tuple(
            tuple(
                _simplify(sum(ginv[i, ell] * (sp.diff(g[k, ell], coords[j]) + sp.diff(g[j, ell], coords[k]) - sp.diff(g[j, k], coords[ell])) / 2 for ell in range(dim)))
                for k in range(dim)
            )
            for j in range(dim)
        )
        for i in range(dim)
    )


def gradient(scalar: Scalar, coordinates: Sequence[Scalar], *, metric=None, inverse_metric=None, covariant: bool = False) -> tuple[Scalar, ...]:
    """Return the gradient components of a scalar.

    By default this returns contravariant components ``g^{ij} d_j f``.  Set
    ``covariant=True`` to return the one-form components ``d_i f``.
    """
    coords = tuple(coordinates)
    cov = tuple(_simplify(_require_sympy().diff(scalar, coord)) for coord in coords)
    if covariant:
        return cov
    ginv = _inverse_metric(metric, inverse_metric)
    if ginv is None:
        return cov
    return tuple(_simplify(sum(ginv[i, j] * cov[j] for j in range(len(coords)))) for i in range(len(coords)))


def divergence(vector_components: Sequence[Scalar], coordinates: Sequence[Scalar], *, metric=None, sqrt_metric_det=None) -> Scalar:
    """Return divergence of contravariant vector components."""
    sp = _require_sympy()
    coords = tuple(coordinates)
    if len(vector_components) != len(coords):
        raise TensorKernelError("Vector component length must match coordinate dimension.")
    sqrtg = _sqrt_metric_det(metric, sqrt_metric_det)
    return _simplify(sum(sp.diff(sqrtg * vector_components[i], coords[i]) for i in range(len(coords))) / sqrtg)


def curl(vector_components: Sequence[Scalar], coordinates: Sequence[Scalar], *, metric=None, inverse_metric=None, sqrt_metric_det=None, input_variance: str = "contravariant") -> tuple[Scalar, ...]:
    """Return the 3D curl as contravariant components.

    If ``input_variance`` is ``"contravariant"``, the vector is first lowered
    with the metric.  If it is ``"covariant"``, the supplied components are
    interpreted as one-form components.
    """
    sp = _require_sympy()
    coords = tuple(coordinates)
    if len(coords) == 2 and len(vector_components) == 2:
        if input_variance not in {"contravariant", "covariant"}:
            raise TensorKernelError("input_variance must be 'contravariant' or 'covariant'.")
        g2 = _matrix(metric) if metric is not None else sp.eye(2)
        if input_variance == "contravariant":
            cov2 = tuple(_simplify(sum(g2[i, j] * vector_components[j] for j in range(2))) for i in range(2))
        else:
            cov2 = tuple(vector_components)
        sqrtg2 = _sqrt_metric_det(g2, sqrt_metric_det)
        return (_simplify((sp.diff(cov2[1], coords[0]) - sp.diff(cov2[0], coords[1])) / sqrtg2),)
    if len(coords) != 3 or len(vector_components) != 3:
        raise TensorKernelError("curl requires either two coordinates/components or three coordinates/components.")
    g = _matrix(metric) if metric is not None else sp.eye(3)
    # In coordinates, (curl A)^i = |g|^{-1/2} [i j k] partial_j A_k
    # for covariant components A_k.  No extra inverse metric appears here;
    # adding one would produce orthonormal/raised components inconsistently.
    sqrtg = _sqrt_metric_det(g, sqrt_metric_det)
    if input_variance not in {"contravariant", "covariant"}:
        raise TensorKernelError("input_variance must be 'contravariant' or 'covariant'.")
    if input_variance == "contravariant":
        cov = tuple(_simplify(sum(g[i, j] * vector_components[j] for j in range(3))) for i in range(3))
    else:
        cov = tuple(vector_components)
    out = []
    for i in range(3):
        total = sum(sp.LeviCivita(i, j, k) * sp.diff(cov[k], coords[j]) for j in range(3) for k in range(3))
        out.append(_simplify(total / sqrtg))
    return tuple(out)


def hessian(scalar: Scalar, coordinates: Sequence[Scalar], *, metric=None, connection=None, convention: str = "covariant") -> tuple[tuple[Scalar, ...], ...]:
    """Return a scalar Hessian.

    ``convention="coordinate"`` returns ordinary second partials.
    ``convention="covariant"`` returns ``nabla_i nabla_j f``.
    """
    sp = _require_sympy()
    coords = tuple(coordinates)
    dim = len(coords)
    if convention not in {"coordinate", "covariant"}:
        raise TensorKernelError("Hessian convention must be 'coordinate' or 'covariant'.")
    gamma = None
    if convention == "covariant":
        gamma = connection if connection is not None else None
        if gamma is None and metric is not None:
            gamma = levi_civita_connection(metric, coords)
    grad_cov = [sp.diff(scalar, coord) for coord in coords]
    rows = []
    for i in range(dim):
        row = []
        for j in range(dim):
            value = sp.diff(grad_cov[i], coords[j])
            if gamma is not None:
                value -= sum(gamma[k][i][j] * grad_cov[k] for k in range(dim))
            row.append(_simplify(value))
        rows.append(tuple(row))
    return tuple(rows)


def scalar_laplacian(scalar: Scalar, coordinates: Sequence[Scalar], *, metric=None, inverse_metric=None, sqrt_metric_det=None) -> Scalar:
    """Return Laplace-Beltrami operator on a scalar."""
    sp = _require_sympy()
    coords = tuple(coordinates)
    ginv = _inverse_metric(metric, inverse_metric)
    if ginv is None:
        ginv = sp.eye(len(coords))
    sqrtg = _sqrt_metric_det(metric, sqrt_metric_det)
    total = 0
    for i, coord_i in enumerate(coords):
        inner = sum(ginv[i, j] * sp.diff(scalar, coords[j]) for j in range(len(coords)))
        total += sp.diff(sqrtg * inner, coord_i)
    return _simplify(total / sqrtg)


def _covariant_derivative_tensor(components: Any, variance: Sequence[str], coordinates: Sequence[Scalar], connection) -> dict[tuple[int, ...], Scalar]:
    sp = _require_sympy()
    coords = tuple(coordinates)
    dim = len(coords)
    rank = len(variance)
    data: dict[tuple[int, ...], Scalar] = {}

    def comp(key: tuple[int, ...]):
        value = components
        for item in key:
            value = value[item]
        return value

    for key in product(range(dim), repeat=rank):
        for deriv in range(dim):
            total = sp.diff(comp(key), coords[deriv])
            for slot, var in enumerate(variance):
                for mid in range(dim):
                    replaced = tuple(mid if pos == slot else item for pos, item in enumerate(key))
                    if var == "up":
                        total += connection[key[slot]][deriv][mid] * comp(replaced)
                    elif var == "down":
                        total -= connection[mid][deriv][key[slot]] * comp(replaced)
                    else:
                        raise TensorKernelError("Variance entries must be 'up' or 'down'.")
            data[key + (deriv,)] = _simplify(total)
    return data


def tensor_laplacian(components: Any, coordinates: Sequence[Scalar], variance: Sequence[str], *, metric=None, inverse_metric=None, connection=None) -> Any:
    """Return the rough/connection Laplacian of a tensor component array."""
    coords = tuple(coordinates)
    dim = len(coords)
    gamma = connection if connection is not None else None
    if gamma is None and metric is not None:
        gamma = levi_civita_connection(metric, coords)
    if gamma is None:
        gamma = tuple(tuple(tuple(0 for _ in range(dim)) for _ in range(dim)) for _ in range(dim))
    ginv = _inverse_metric(metric, inverse_metric)
    if ginv is None:
        ginv = _require_sympy().eye(dim)
    first = _covariant_derivative_tensor(components, tuple(variance), coords, gamma)

    def first_array(key):
        return first[key]

    second = _covariant_derivative_tensor(lambda_like(first_array, len(variance) + 1, dim), tuple(variance) + ("down",), coords, gamma)
    rank = len(variance)

    def build(prefix: tuple[int, ...], depth: int):
        if depth == rank:
            return _simplify(sum(ginv[a, b] * second[prefix + (a, b)] for a in range(dim) for b in range(dim)))
        return tuple(build(prefix + (i,), depth + 1) for i in range(dim))

    return build(tuple(), 0)


class lambda_like:
    """Indexable adapter around a callable on index tuples."""

    def __init__(self, function, rank: int, dimension: int):
        self.function = function
        self.rank = rank
        self.dimension = dimension

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        if len(item) == self.rank:
            return self.function(tuple(item))
        return _NestedAdapter(self, tuple(item))


class _NestedAdapter:
    def __init__(self, parent: lambda_like, prefix: tuple[int, ...]):
        self.parent = parent
        self.prefix = prefix

    def __getitem__(self, item):
        key = self.prefix + ((item,) if not isinstance(item, tuple) else item)
        if len(key) == self.parent.rank:
            return self.parent.function(key)
        return _NestedAdapter(self.parent, key)


def componentwise_laplacian(components: Any, coordinates: Sequence[Scalar], *, metric=None, inverse_metric=None, sqrt_metric_det=None) -> Any:
    """Apply the scalar Laplace-Beltrami operator to each component."""
    if isinstance(components, (list, tuple)):
        return tuple(componentwise_laplacian(item, coordinates, metric=metric, inverse_metric=inverse_metric, sqrt_metric_det=sqrt_metric_det) for item in components)
    return scalar_laplacian(components, coordinates, metric=metric, inverse_metric=inverse_metric, sqrt_metric_det=sqrt_metric_det)


def vector_laplacian(vector_components: Sequence[Scalar], coordinates: Sequence[Scalar], *, metric=None, inverse_metric=None, connection=None, variance: str = "up", convention: str = "rough") -> tuple[Scalar, ...]:
    """Return a vector Laplacian with an explicit convention.

    ``convention="rough"`` returns ``g^{ij} nabla_i nabla_j V``.
    ``convention="componentwise"`` applies the scalar Laplacian to each
    supplied component.  A full Hodge vector Laplacian is intentionally not
    aliased to either convention.
    """
    if convention not in {"rough", "connection", "componentwise"}:
        raise TensorKernelError("Vector Laplacian convention must be 'rough', 'connection', or 'componentwise'.")
    if convention == "componentwise":
        return tuple(componentwise_laplacian(tuple(vector_components), coordinates, metric=metric, inverse_metric=inverse_metric))
    variance = normalize_variance(variance)
    result = tensor_laplacian(tuple(vector_components), coordinates, (variance,), metric=metric, inverse_metric=inverse_metric, connection=connection)
    return tuple(result)


def laplacian(obj: Any, coordinates: Sequence[Scalar], *, metric=None, inverse_metric=None, sqrt_metric_det=None, connection=None, variance: Sequence[str] | str | None = None, convention: str = "laplace_beltrami") -> Any:
    """Dispatch to scalar, vector, or tensor Laplacian by variance and convention."""
    if variance is None:
        if convention not in {"laplace_beltrami", "scalar"}:
            raise TensorKernelError("Scalar Laplacian convention must be 'laplace_beltrami' or 'scalar'.")
        return scalar_laplacian(obj, coordinates, metric=metric, inverse_metric=inverse_metric, sqrt_metric_det=sqrt_metric_det)
    if isinstance(variance, str):
        vector_convention = "rough" if convention in {"laplace_beltrami", "rough", "connection"} else convention
        return vector_laplacian(obj, coordinates, metric=metric, inverse_metric=inverse_metric, connection=connection, variance=variance, convention=vector_convention)
    if convention == "componentwise":
        return componentwise_laplacian(obj, coordinates, metric=metric, inverse_metric=inverse_metric, sqrt_metric_det=sqrt_metric_det)
    if convention not in {"rough", "connection", "laplace_beltrami"}:
        raise TensorKernelError("Tensor Laplacian convention must be 'rough', 'connection', 'laplace_beltrami', or 'componentwise'.")
    return tensor_laplacian(obj, coordinates, variance, metric=metric, inverse_metric=inverse_metric, connection=connection)


def scale_factors(metric: Sequence[Sequence[Scalar]]) -> tuple[Scalar, ...]:
    """Return orthogonal-coordinate scale factors ``h_i = sqrt(|g_ii|)``."""
    sp = _require_sympy()
    g = _matrix(metric)
    if any(g[i, j] != 0 for i in range(g.rows) for j in range(g.cols) if i != j):
        raise TensorKernelError("Physical component conversion currently requires a diagonal metric.")
    return tuple(sp.sqrt(sp.Abs(g[i, i])) for i in range(g.rows))


def coordinate_to_physical_vector(vector_components: Sequence[Scalar], metric: Sequence[Sequence[Scalar]], *, variance: str = "contravariant") -> tuple[Scalar, ...]:
    """Convert coordinate vector/covector components to orthonormal physical components."""
    h = scale_factors(metric)
    if variance in {"contravariant", "up", "upper"}:
        return tuple(_simplify(h[i] * vector_components[i]) for i in range(len(h)))
    if variance in {"covariant", "down", "lower"}:
        return tuple(_simplify(vector_components[i] / h[i]) for i in range(len(h)))
    raise TensorKernelError("variance must be contravariant/up or covariant/down.")


def physical_to_coordinate_vector(physical_components: Sequence[Scalar], metric: Sequence[Sequence[Scalar]], *, variance: str = "contravariant") -> tuple[Scalar, ...]:
    """Convert orthonormal physical vector/covector components to coordinate components."""
    h = scale_factors(metric)
    if variance in {"contravariant", "up", "upper"}:
        return tuple(_simplify(physical_components[i] / h[i]) for i in range(len(h)))
    if variance in {"covariant", "down", "lower"}:
        return tuple(_simplify(physical_components[i] * h[i]) for i in range(len(h)))
    raise TensorKernelError("variance must be contravariant/up or covariant/down.")


def physical_curl(vector_physical_components: Sequence[Scalar], coordinates: Sequence[Scalar], *, metric, input_variance: str = "contravariant") -> tuple[Scalar, ...]:
    """Return curl components in an orthonormal frame for a diagonal 3D metric."""
    coord = physical_to_coordinate_vector(vector_physical_components, metric, variance=input_variance)
    result = curl(coord, coordinates, metric=metric, input_variance=input_variance)
    return coordinate_to_physical_vector(result, metric, variance="contravariant")


def hessian_result(scalar: Scalar, coordinates: Sequence[Scalar], *, metric=None, connection=None, convention: str = "covariant") -> VectorCalculusResult:
    """Return Hessian with explicit convention metadata."""
    value = hessian(scalar, coordinates, metric=metric, connection=connection, convention=convention)
    return VectorCalculusResult("hessian", value, tuple(coordinates), _vector_calculus_conventions("hessian", coordinates, convention, variance=("down", "down")))


def curl_result(vector_components: Sequence[Scalar], coordinates: Sequence[Scalar], *, metric=None, inverse_metric=None, sqrt_metric_det=None, input_variance: str = "contravariant") -> VectorCalculusResult:
    """Return curl with explicit convention metadata."""
    value = curl(vector_components, coordinates, metric=metric, inverse_metric=inverse_metric, sqrt_metric_det=sqrt_metric_det, input_variance=input_variance)
    meta = _vector_calculus_conventions("curl", coordinates, "coordinate_levi_civita_density", variance=input_variance)
    return VectorCalculusResult("curl", value, tuple(coordinates), meta)


def laplacian_result(obj: Any, coordinates: Sequence[Scalar], *, metric=None, inverse_metric=None, sqrt_metric_det=None, connection=None, variance: Sequence[str] | str | None = None, convention: str = "laplace_beltrami") -> VectorCalculusResult:
    """Return Laplacian with explicit convention metadata."""
    value = laplacian(obj, coordinates, metric=metric, inverse_metric=inverse_metric, sqrt_metric_det=sqrt_metric_det, connection=connection, variance=variance, convention=convention)
    return VectorCalculusResult("laplacian", value, tuple(coordinates), _vector_calculus_conventions("laplacian", coordinates, convention, variance=variance))
