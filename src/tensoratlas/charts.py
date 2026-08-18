from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple, List, Any

import sympy as sp

from .normal_forms import TNFMatrix, TNFTensorArray, as_tnf_matrix, as_tnf_array, tnf_matrix_to_sympy, tnf_array_to_sympy, tnf_build_array, tnf_build_matrix, tnf_column_from_entries
from .symbolic_decision import is_equal, is_zero, canonical_simplify
from .symbolic_simplification_policy import coordinate_simplify_expr, domain_policy_from_specs


MetricFunc = Callable[[Tuple[sp.Symbol, ...]], sp.Matrix]
AssumptionFunc = Callable[[Tuple[sp.Symbol, ...]], sp.Expr]
ParameterFunc = Callable[[], Dict[str, sp.Symbol]]


def _default_parameters() -> Dict[str, sp.Symbol]:
    return {"a": sp.Symbol("a", positive=True, real=True)}


def _abc_parameters() -> Dict[str, sp.Symbol]:
    return {
        "a": sp.Symbol("a", positive=True, real=True),
        "b": sp.Symbol("b", positive=True, real=True),
        "c": sp.Symbol("c", positive=True, real=True),
    }


def _bc_parameters() -> Dict[str, sp.Symbol]:
    return {
        "b": sp.Symbol("b", positive=True, real=True),
        "c": sp.Symbol("c", positive=True, real=True),
    }


def _infer_coordinate_domains(metric_name: str, chart_name: str, dimension: int, params: Optional[Dict[str, sp.Symbol]] = None) -> Dict[str, object]:
    params = params or {}
    a = params.get("a", sp.Symbol("a", positive=True, real=True))
    b = params.get("b", sp.Symbol("b", positive=True, real=True))
    c = params.get("c", sp.Symbol("c", positive=True, real=True))
    defaults = {
        ("Euclidean", "Cartesian", 2): {
            "x": {"kind": "real_line"},
            "y": {"kind": "real_line"},
        },
        ("Euclidean", "Cartesian", 3): {
            "x": {"kind": "real_line"},
            "y": {"kind": "real_line"},
            "z": {"kind": "real_line"},
        },
        ("Euclidean", "Polar", 2): {
            "r": {"kind": "half_line", "min": 0},
            "theta": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
        },
        ("Euclidean", "Cylindrical", 3): {
            "r": {"kind": "half_line", "min": 0},
            "theta": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
            "z": {"kind": "real_line"},
        },
        ("Euclidean", "Spherical", 3): {
            "r": {"kind": "half_line", "min": 0},
            "theta": {"kind": "open_interval", "min": 0, "max": sp.pi},
            "phi": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
        },
        ("Euclidean", "Elliptic", 2): {
            "mu": {"kind": "half_line", "min": 0},
            "nu": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
        },
        ("Euclidean", "Parabolic", 2): {
            "sigma": {"kind": "real_line"},
            "tau": {"kind": "half_line", "min": 0},
        },
        ("Euclidean", "Paraboloidal", 3): {
            "u": {"kind": "half_line", "min": 0},
            "v": {"kind": "half_line", "min": 0},
            "phi": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
        },
        ("Euclidean", "ProlateSpheroidal", 3): {
            "mu": {"kind": "open_interval", "min": 0, "max": sp.oo},
            "nu": {"kind": "open_interval", "min": 0, "max": sp.pi},
            "phi": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
        },
        ("Euclidean", "OblateSpheroidal", 3): {
            "mu": {"kind": "open_interval", "min": 0, "max": sp.oo},
            "nu": {"kind": "open_interval", "min": -sp.pi/2, "max": sp.pi/2},
            "phi": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
        },
        ("Euclidean", "Bispherical", 3): {
            "sigma": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
            "tau": {"kind": "real_line"},
            "phi": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
        },
        ("Euclidean", "Toroidal", 3): {
            "tau": {"kind": "half_line", "min": 0},
            "sigma": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
            "phi": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
        },
        ("Euclidean", "Bipolar", 2): {
            "sigma": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
            "tau": {"kind": "real_line"},
        },
        ("Euclidean", "ParabolicCylindrical", 3): {
            "u": {"kind": "half_line", "min": 0},
            "v": {"kind": "real_line"},
            "z": {"kind": "real_line"},
        },
        ("Euclidean", "EllipticCylindrical", 3): {
            "mu": {"kind": "half_line", "min": 0},
            "nu": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
            "z": {"kind": "real_line"},
        },
        ("Euclidean", "Conical", 3): {
            "r": {"kind": "half_line", "min": 0},
            "mu": {"kind": "open_interval", "min": b, "max": sp.oo},
            "nu": {"kind": "open_interval", "min": c, "max": b},
        },
        ("Euclidean", "Ellipsoidal", 3): {
            "lam": {"kind": "open_interval", "min": a**2, "max": sp.oo},
            "mu": {"kind": "open_interval", "min": b**2, "max": a**2},
            "nu": {"kind": "open_interval", "min": c**2, "max": b**2},
        },
    }
    return defaults.get((metric_name, chart_name, dimension), {})


def _default_chart_description(metric_name: str, chart_name: str, dimension: int) -> str:
    return f"{metric_name} {chart_name} coordinates in {dimension}D"


def _all_chart_family_properties() -> Tuple[str, ...]:
    return (
        "metric_name",
        "chart_name",
        "family_name",
        "dimension",
        "coordinates",
        "coordinate_names",
        "parameters",
        "description",
        "orthogonal_metric",
        "metric_tensor",
        "inverse_metric_tensor",
        "scale_factors",
        "sqrt_metric_det",
        "coordinate_range_assumptions",
        "coordinate_domains",
        "available_properties",
        "riemann_tensor",
        "ricci_tensor",
        "scalar_curvature",
        "einstein_tensor",
        "weyl_tensor",
        "cyclic_coordinates",
    )


def _metric_is_orthogonal(metric: Optional[object]) -> Optional[bool]:
    if metric is None:
        return None
    rows = metric.rows
    cols = metric.cols
    for i in range(rows):
        for j in range(cols):
            if i != j and not is_zero(metric[i, j]):
                return False
    return True


def _tnf_numeric_matrix(rows: int, cols: int, value_func) -> TNFMatrix:
    return tnf_build_matrix(rows, cols, lambda i, j: sp.N(value_func(i, j)))


def _tnf_numeric_column(entries) -> TNFMatrix:
    return tnf_column_from_entries(sp.N(entry) for entry in entries)


def _tnf_column_to_tuple(column: TNFMatrix) -> tuple[sp.Expr, ...]:
    return tuple(column[i] for i in range(column.rows))


def _chart_runtime_cache(chart: "CoordinateChart") -> dict:
    try:
        return object.__getattribute__(chart, '_runtime_cache')
    except AttributeError:
        cache = {}
        object.__setattr__(chart, '_runtime_cache', cache)
        return cache


def _coords_cache_key(coords):
    if coords is None:
        return None
    return tuple(sp.srepr(c) for c in coords)


def _cached_chart_value(chart: "CoordinateChart", name: str, coords, extra, compute):
    cache = _chart_runtime_cache(chart)
    key = (name, _coords_cache_key(coords), extra)
    if key not in cache:
        cache[key] = compute()
    return cache[key]


def _compute_christoffel_symbols(chart, coords, kind: str = "second"):
    if coords is None:
        coords = chart.symbols()
    metric_tnf = chart.metric_tnf(coords)
    if metric_tnf is None:
        return None
    dim = chart.dimension
    gamma1 = tnf_build_array(
        (dim, dim, dim),
        lambda idx: sp.Rational(1, 2) * (
            sp.diff(metric_tnf[idx[0], idx[2]], coords[idx[1]])
            + sp.diff(metric_tnf[idx[0], idx[1]], coords[idx[2]])
            - sp.diff(metric_tnf[idx[1], idx[2]], coords[idx[0]])
        ),
    )
    if kind == "first":
        return gamma1
    if kind != "second":
        raise ValueError("kind must be 'first' or 'second'.")
    metric_inv_nf = chart.inverse_metric_tnf(coords)
    return _cached_chart_value(chart, 'christoffel_symbols', coords, kind, lambda: tnf_build_array(
        (dim, dim, dim),
        lambda idx: sum(metric_inv_nf[idx[0], m] * gamma1[m, idx[1], idx[2]] for m in range(dim)),
    ))


def _compute_riemann_tensor(chart, coords):
    if coords is None:
        coords = chart.symbols()
    gamma = chart.christoffel_symbols(coords, kind="second")
    if gamma is None:
        return None
    dim = chart.dimension
    return _cached_chart_value(chart, 'riemann_tensor', coords, None, lambda: tnf_build_array(
        (dim, dim, dim, dim),
        lambda idx: (
            sp.diff(gamma[idx[0], idx[1], idx[3]], coords[idx[2]])
            - sp.diff(gamma[idx[0], idx[1], idx[2]], coords[idx[3]])
            + sum(
                gamma[idx[0], m, idx[2]] * gamma[m, idx[1], idx[3]]
                - gamma[idx[0], m, idx[3]] * gamma[m, idx[1], idx[2]]
                for m in range(dim)
            )
        ),
    ))


def _compute_ricci_tensor(chart, coords):
    if coords is None:
        coords = chart.symbols()
    riem = chart.riemann_tensor(coords)
    if riem is None:
        return None
    dim = chart.dimension
    return _cached_chart_value(chart, 'ricci_tensor', coords, None, lambda: tnf_build_array((dim, dim), lambda idx: sum(riem[i, idx[0], i, idx[1]] for i in range(dim))))


def _compute_scalar_curvature(chart, coords):
    if coords is None:
        coords = chart.symbols()
    ric = chart.ricci_tensor(coords)
    ginv = chart.inverse_metric(coords)
    if ric is None or ginv is None:
        return None
    return _cached_chart_value(chart, 'scalar_curvature', coords, None, lambda: canonical_simplify(sum(ginv[i, j] * ric[i, j] for i in range(chart.dimension) for j in range(chart.dimension))))



def _standard_scale_factors(metric_name: str, chart_name: str, dimension: int, coords, params):
    if metric_name != "Euclidean":
        return None
    a = params.get("a", sp.Symbol("a", positive=True, real=True))
    if chart_name == "Polar" and dimension == 2:
        r, _theta = coords
        return (sp.Integer(1), r)
    if chart_name == "Cylindrical" and dimension == 3:
        r, _theta, _z = coords
        return (sp.Integer(1), r, sp.Integer(1))
    if chart_name == "Spherical" and dimension == 3:
        r, theta, _phi = coords
        return (sp.Integer(1), r, r * sp.sin(theta))
    if chart_name == "Elliptic" and dimension == 2:
        mu, nu = coords
        h = a * sp.sqrt(sp.sinh(mu)**2 + sp.sin(nu)**2)
        return (h, h)
    if chart_name == "ProlateSpheroidal" and dimension == 3:
        mu, nu, _phi = coords
        h = a * sp.sqrt(sp.sinh(mu)**2 + sp.sin(nu)**2)
        return (h, h, a * sp.sinh(mu) * sp.sin(nu))
    if chart_name == "Bipolar" and dimension == 2:
        sigma, tau = coords
        h = a / (sp.cosh(tau) - sp.cos(sigma))
        return (h, h)
    return None

def _standard_scale_factor_magnitudes(metric_name: str, chart_name: str, dimension: int, coords, params, scale_factors):
    if metric_name == "Euclidean" and chart_name == "Polar" and dimension == 2:
        r, _theta = coords
        return (sp.Integer(1), sp.Abs(r))
    if metric_name == "Euclidean" and chart_name == "Cylindrical" and dimension == 3:
        r, _theta, _z = coords
        return (sp.Integer(1), sp.Abs(r), sp.Integer(1))
    if metric_name == "Euclidean" and chart_name == "Spherical" and dimension == 3:
        r, theta, _phi = coords
        return (sp.Integer(1), sp.Abs(r), sp.Abs(r * sp.sin(theta)))
    return tuple(sp.sqrt(h**2) for h in scale_factors)


def _standard_volume_factor(metric_name: str, chart_name: str, dimension: int, coords, params, scale_factors):
    if metric_name == "Euclidean" and chart_name == "Polar" and dimension == 2:
        r, _theta = coords
        return sp.Abs(r)
    if metric_name == "Euclidean" and chart_name == "Cylindrical" and dimension == 3:
        r, _theta, _z = coords
        return sp.Abs(r)
    if metric_name == "Euclidean" and chart_name == "Spherical" and dimension == 3:
        r, theta, _phi = coords
        return r**2 * sp.Abs(sp.sin(theta))
    return sp.sqrt(sp.prod(h**2 for h in scale_factors))

@dataclass(frozen=True)
class CoordinateChart:
    metric_name: str
    chart_name: str
    dimension: int
    coordinate_names: Tuple[str, ...]
    assumptions_func: Optional[AssumptionFunc] = None
    metric_func: Optional[MetricFunc] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    parameter_func: Optional[ParameterFunc] = _default_parameters

    def symbols(self) -> Tuple[sp.Symbol, ...]:
        normalized_names = []
        for name in self.coordinate_names:
            normalized_names.append(
                name.replace("θ", "theta").replace("φ", "phi").replace("ρ", "rho").replace("ψ", "psi")
            )
        return sp.symbols(" ".join(normalized_names), real=True)

    def parameters(self) -> Dict[str, sp.Symbol]:
        if self.parameter_func is None:
            return {}
        return self.parameter_func()


    @property
    def coords(self) -> Tuple[sp.Symbol, ...]:
        """Alias for ``symbols()`` used in notebook examples."""
        return self.symbols()

    @property
    def coordinate_symbols(self) -> Tuple[sp.Symbol, ...]:
        """Symbolic coordinate tuple for this chart."""
        return self.symbols()

    def domain_conditions(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Dict[str, object]:
        """Return coordinate-domain metadata and symbolic assumptions."""
        if coords is None:
            coords = self.symbols()
        return {
            "coordinate_domains": self.coordinate_domains(),
            "domain_assumptions": self.domain_assumptions(coords),
            "singularity_loci": self.singularity_loci(coords),
        }

    def summary(self) -> Dict[str, object]:
        """Return a compact chart summary for notebooks and debugging."""
        return {
            "metric_name": self.metric_name,
            "chart_name": self.chart_name,
            "dimension": self.dimension,
            "coordinate_names": self.coordinate_names,
            "coordinates": self.symbols(),
            "description": self.description(),
            "orthogonal_metric": self.is_orthogonal_metric(),
            "metadata": self.metadata_completeness(),
        }

    def validate(self) -> bool:
        """Validate basic chart metadata and metric dimensions."""
        if len(self.coordinate_names) != self.dimension:
            raise ValueError("coordinate_names length must match chart dimension")
        metric = self.metric()
        if metric is not None and metric.shape != (self.dimension, self.dimension):
            raise ValueError("metric shape must match chart dimension")
        return True

    def metric_tnf(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFMatrix]:
        if self.metric_func is None:
            return None
        if coords is None:
            coords = self.symbols()
        return _cached_chart_value(self, 'metric_tnf', coords, None, lambda: as_tnf_matrix(self.metric_func(coords)))

    def metric(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Matrix]:
        metric_tnf = self.metric_tnf(coords)
        return None if metric_tnf is None else tnf_matrix_to_sympy(metric_tnf)

    def metric_sympy(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Matrix]:
        return self.metric(coords)

    def inverse_metric_tnf(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFMatrix]:
        metric_tnf = self.metric_tnf(coords)
        if metric_tnf is None:
            return None
        if all(metric_tnf[i, j] == 0 for i in range(metric_tnf.rows) for j in range(metric_tnf.cols) if i != j):
            return tnf_build_matrix(metric_tnf.rows, metric_tnf.cols, lambda i, j: sp.Integer(0) if i != j else 1 / metric_tnf[i, i])
        return metric_tnf.inv()

    def inverse_metric(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Matrix]:
        metric_tnf = self.inverse_metric_tnf(coords)
        return None if metric_tnf is None else tnf_matrix_to_sympy(metric_tnf)

    def inverse_metric_sympy(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Matrix]:
        return self.inverse_metric(coords)

    def scale_factors(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[Tuple[sp.Expr, ...]]:
        if coords is None:
            coords = self.symbols()
        fast = self._fast_scale_factors(coords)
        if fast is not None:
            # Public scale factors should preserve the conventional magnitude
            # notation.  Domain-specific sign cleanup is reserved for internal
            # coordinate operations, where it avoids unnecessary Abs/sign work.
            return _cached_chart_value(self, 'scale_factors_fast', coords, None, lambda: _standard_scale_factor_magnitudes(self.metric_name, self.chart_name, self.dimension, coords, self.parameters(), fast))
        metric_tnf = self.metric_tnf(coords)
        if metric_tnf is None:
            return None
        return _cached_chart_value(self, 'scale_factors', coords, None, lambda: tuple(sp.sqrt(metric_tnf[idx, idx]) for idx in range(metric_tnf.rows)))

    def sqrt_metric_det(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if coords is None:
            coords = self.symbols()
        fast = self._fast_scale_factors(coords)
        if fast is not None:
            return _cached_chart_value(self, 'sqrt_metric_det_fast', coords, None, lambda: _standard_volume_factor(self.metric_name, self.chart_name, self.dimension, coords, self.parameters(), fast))
        metric_tnf = self.metric_tnf(coords)
        if metric_tnf is None:
            return None
        return _cached_chart_value(self, 'sqrt_metric_det', coords, None, lambda: sp.sqrt(sp.Abs(metric_tnf.det().to_sympy())))
    def volume_density(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        return self.sqrt_metric_det(coords)

    def connection_coefficients(self, coords: Optional[Tuple[sp.Symbol, ...]] = None, connection: Optional[TNFTensorArray] = None) -> Optional[TNFTensorArray]:
        if connection is not None:
            return as_tnf_array(connection)
        return self.christoffel_symbols(coords, kind="second")

    def torsion_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None, connection: Optional[TNFTensorArray] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        gamma = self.connection_coefficients(coords, connection=connection)
        if gamma is None:
            return None
        dim = self.dimension
        return tnf_build_array((dim, dim, dim), lambda idx: canonical_simplify(gamma[idx[0], idx[1], idx[2]] - gamma[idx[0], idx[2], idx[1]]))

    def nonmetricity_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None, connection: Optional[TNFTensorArray] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        metric = self.metric_tnf(coords)
        gamma = self.connection_coefficients(coords, connection=connection)
        if metric is None or gamma is None:
            return None
        dim = self.dimension
        return tnf_build_array((dim, dim, dim), lambda idx: canonical_simplify(
            sp.diff(metric[idx[1], idx[2]], coords[idx[0]])
            - sum(gamma[m, idx[1], idx[0]] * metric[m, idx[2]] for m in range(dim))
            - sum(gamma[m, idx[2], idx[0]] * metric[idx[1], m] for m in range(dim))
        ))

    def algebraic_bianchi_residual(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        riem = self.lower_riemann_tensor(coords)
        if riem is None:
            return None
        dim = self.dimension
        return tnf_build_array((dim, dim, dim, dim), lambda idx: canonical_simplify(
            riem[idx[0], idx[1], idx[2], idx[3]]
            + riem[idx[0], idx[2], idx[3], idx[1]]
            + riem[idx[0], idx[3], idx[1], idx[2]]
        ))

    def differential_bianchi_residual(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        riem = self.riemann_tensor(coords)
        gamma = self.christoffel_symbols(coords, kind="second")
        if riem is None or gamma is None:
            return None
        dim = self.dimension

        def covariant(a, b, c, d, e):
            total = sp.diff(riem[a, b, c, d], coords[e])
            for m in range(dim):
                total += gamma[a, m, e] * riem[m, b, c, d]
                total -= gamma[m, b, e] * riem[a, m, c, d]
                total -= gamma[m, c, e] * riem[a, b, m, d]
                total -= gamma[m, d, e] * riem[a, b, c, m]
            return canonical_simplify(total)

        return tnf_build_array((dim, dim, dim, dim, dim), lambda idx: canonical_simplify(
            covariant(idx[0], idx[1], idx[2], idx[3], idx[4])
            + covariant(idx[0], idx[1], idx[3], idx[4], idx[2])
            + covariant(idx[0], idx[1], idx[4], idx[2], idx[3])
        ))

    def bianchi_identity_report(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Dict[str, Optional[TNFTensorArray]]:
        if coords is None:
            coords = self.symbols()
        return {
            "algebraic_residual": self.algebraic_bianchi_residual(coords),
            "differential_residual": self.differential_bianchi_residual(coords),
        }

    def assumptions(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if self.assumptions_func is None:
            return None
        if coords is None:
            coords = self.symbols()
        return self.assumptions_func(coords)

    def coordinate_domains(self) -> Dict[str, object]:
        domains = dict(_infer_coordinate_domains(self.metric_name, self.chart_name, self.dimension, self.parameters()))
        domains.update(dict(self.metadata.get("coordinate_domains", {})))
        return domains
    def coordinate_domain_policy(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        if coords is None:
            coords = self.symbols()
        return domain_policy_from_specs(self.coordinate_names, coords, self.coordinate_domains())

    def cleanup_coordinate_expr(self, expr, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Expr:
        return coordinate_simplify_expr(expr, self.coordinate_domain_policy(coords))

    def _fast_scale_factors(self, coords: Tuple[sp.Symbol, ...]) -> Optional[Tuple[sp.Expr, ...]]:
        return _standard_scale_factors(self.metric_name, self.chart_name, self.dimension, coords, self.parameters())


    def standard_jacobian_determinant(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if coords is None:
            coords = self.symbols()
        fast = self._fast_scale_factors(coords)
        if fast is None:
            return None
        return self.cleanup_coordinate_expr(sp.prod(fast), coords)

    def standard_gradient_components(self, expr, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFMatrix]:
        if coords is None:
            coords = self.symbols()
        fast = self._fast_scale_factors(coords)
        if fast is None:
            return None
        return tnf_column_from_entries(self.cleanup_coordinate_expr(sp.diff(expr, coords[i]) / fast[i]**2, coords) for i in range(self.dimension))

    def standard_divergence(self, components, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if coords is None:
            coords = self.symbols()
        fast = self._fast_scale_factors(coords)
        if fast is None:
            return None
        comps = as_tnf_matrix(components)
        density = sp.prod(fast)
        total = sum(sp.diff(density * comps[i, 0], coords[i]) for i in range(self.dimension))
        return self.cleanup_coordinate_expr(total / density, coords)

    def standard_laplacian(self, expr, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if coords is None:
            coords = self.symbols()
        if self.metric_name == "Euclidean" and self.chart_name == "Polar" and self.dimension == 2:
            r, theta = coords
            return sp.diff(expr, (r, 2)) + sp.diff(expr, r) / r + sp.diff(expr, (theta, 2)) / r**2
        if self.metric_name == "Euclidean" and self.chart_name == "Cylindrical" and self.dimension == 3:
            r, theta, z = coords
            return sp.diff(expr, (r, 2)) + sp.diff(expr, r) / r + sp.diff(expr, (theta, 2)) / r**2 + sp.diff(expr, (z, 2))
        if self.metric_name == "Euclidean" and self.chart_name == "Spherical" and self.dimension == 3:
            r, theta, phi = coords
            return (
                sp.diff(expr, (r, 2))
                + 2 * sp.diff(expr, r) / r
                + sp.diff(expr, (theta, 2)) / r**2
                + sp.diff(expr, theta) / (r**2 * sp.tan(theta))
                + sp.diff(expr, (phi, 2)) / (r**2 * sp.sin(theta)**2)
            )
        fast = self._fast_scale_factors(coords)
        if fast is None:
            return None
        density = sp.prod(fast)
        total = sum(sp.diff(density * sp.diff(expr, coords[i]) / fast[i]**2, coords[i]) for i in range(self.dimension))
        return self.cleanup_coordinate_expr(total / density, coords)

    def domain_assumptions(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        """Return symbolic assumptions induced by the registered coordinate domains."""
        if coords is None:
            coords = self.symbols()
        domains = self.coordinate_domains()
        clauses: list[sp.Expr] = []
        for name, symbol in zip(self.coordinate_names, coords):
            spec = domains.get(name, {})
            kind = spec.get("kind")
            if kind == "half_line":
                clauses.append(sp.Ge(symbol, spec.get("min", 0)))
            elif kind == "open_interval":
                clauses.append(sp.Gt(symbol, spec["min"]))
                clauses.append(sp.Lt(symbol, spec["max"]))
            elif kind == "closed_interval":
                clauses.append(sp.Ge(symbol, spec["min"]))
                clauses.append(sp.Le(symbol, spec["max"]))
        if self.assumptions_func is not None:
            chart_assumptions = self.assumptions(coords)
            if chart_assumptions is not None:
                clauses.append(chart_assumptions)
        if not clauses:
            return None
        return sp.And(*clauses)

    def singularity_loci(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Tuple[sp.Expr, ...]:
        """Return a tuple of simple singularity conditions inferred from scale factors."""
        if coords is None:
            coords = self.symbols()
        scale_factors = self.scale_factors(coords)
        if scale_factors is None:
            return tuple()
        singularities: list[sp.Expr] = []
        for factor in scale_factors:
            zero_set = canonical_simplify(factor)
            if zero_set not in {0, 1}:
                singularities.append(sp.Eq(zero_set, 0))
        return tuple(singularities)

    def validate_point(self, point: Tuple[sp.Expr, ...]) -> bool:
        """Check whether a concrete point satisfies the registered coordinate domain constraints."""
        coords = self.symbols()
        assumptions = self.domain_assumptions(coords)
        if assumptions is None:
            return True
        substituted = self.cleanup_coordinate_expr(assumptions.subs(dict(zip(coords, point))), coords)
        return substituted != False

    def metadata_completeness(self) -> Dict[str, bool]:
        """Summarize whether the standard chart metadata hooks are available."""
        coords = self.symbols()
        return {
            "metric": self.metric(coords) is not None,
            "inverse_metric": self.inverse_metric(coords) is not None,
            "scale_factors": self.scale_factors(coords) is not None,
            "sqrt_metric_det": self.sqrt_metric_det(coords) is not None,
            "coordinate_domains": bool(self.coordinate_domains()),
            "description": bool(self.description()),
        }

    def description(self) -> str:
        return str(self.metadata.get("description", _default_chart_description(self.metric_name, self.chart_name, self.dimension)))

    def is_orthogonal_metric(self) -> Optional[bool]:
        metric_tnf = self.metric()
        return _metric_is_orthogonal(metric_tnf)

    def chart_properties(self) -> Tuple[str, ...]:
        keys = list(_all_chart_family_properties())
        extra = [k for k in self.metadata.keys() if k not in keys]
        return tuple(keys + extra)


    def christoffel_symbols(self, coords: Optional[Tuple[sp.Symbol, ...]] = None, kind: str = "second") -> Optional[TNFTensorArray]:
        return _compute_christoffel_symbols(self, coords, kind)

    def christoffel_symbols_sympy(self, coords: Optional[Tuple[sp.Symbol, ...]] = None, kind: str = "second") -> Optional[sp.MutableDenseNDimArray]:
        gamma = self.christoffel_symbols(coords, kind=kind)
        return None if gamma is None else gamma.to_sympy()

    def riemann_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        return _compute_riemann_tensor(self, coords)

    def riemann_tensor_sympy(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.MutableDenseNDimArray]:
        tensor = self.riemann_tensor(coords)
        return None if tensor is None else tensor.to_sympy()

    def ricci_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        return _compute_ricci_tensor(self, coords)

    def ricci_tensor_sympy(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.MutableDenseNDimArray]:
        tensor = self.ricci_tensor(coords)
        return None if tensor is None else tensor.to_sympy()

    def scalar_curvature(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        return _compute_scalar_curvature(self, coords)

    def geodesic_equations(self, functions: Optional[Tuple[sp.Function, ...]] = None, parameter: Optional[sp.Symbol] = None):
        if parameter is None:
            parameter = sp.Symbol('lambda', real=True)
        coords = self.symbols()
        if functions is None:
            functions = tuple(sp.Function(str(c))(parameter) for c in coords)
        gamma = self.christoffel_symbols(coords)
        if gamma is None:
            raise ValueError("Chart does not define a metric.")
        subs = dict(zip(coords, functions))
        eqs = []
        for i in range(self.dimension):
            total = sp.diff(functions[i], parameter, 2)
            for j in range(self.dimension):
                for k in range(self.dimension):
                    total += canonical_simplify(gamma[i, j, k].subs(subs)) * sp.diff(functions[j], parameter) * sp.diff(functions[k], parameter)
            eqs.append(sp.Eq(canonical_simplify(total), 0))
        return tuple(eqs)

    def geodesic_lagrangian(self, functions: Optional[Tuple[sp.Function, ...]] = None, parameter: Optional[sp.Symbol] = None) -> sp.Expr:
        if parameter is None:
            parameter = sp.Symbol('lambda', real=True)
        coords = self.symbols()
        metric = self.metric(coords)
        if metric is None:
            raise ValueError("Chart does not define a metric.")
        if functions is None:
            functions = tuple(sp.Function(str(c))(parameter) for c in coords)
        subs = dict(zip(coords, functions))
        velocities = [sp.diff(func, parameter) for func in functions]
        total = 0
        for i in range(self.dimension):
            for j in range(self.dimension):
                total += metric[i, j].subs(subs) * velocities[i] * velocities[j]
        return self.cleanup_coordinate_expr(sp.Rational(1, 2) * total, coords)

    def cyclic_coordinates(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Tuple[int, ...]:
        if coords is None:
            coords = self.symbols()
        metric = self.metric(coords)
        if metric is None:
            return tuple()
        cyclic = []
        for idx, coord in enumerate(coords):
            if all(is_zero(sp.diff(metric[i, j], coord)) for i in range(self.dimension) for j in range(self.dimension)):
                cyclic.append(idx)
        return tuple(cyclic)

    def coordinate_killing_vectors(self) -> Tuple[Tuple[sp.Expr, ...], ...]:
        out = []
        for idx in self.cyclic_coordinates():
            comps = [sp.Integer(0)] * self.dimension
            comps[idx] = sp.Integer(1)
            out.append(tuple(comps))
        return tuple(out)

    def geodesic_first_integrals(self, functions: Optional[Tuple[sp.Function, ...]] = None, parameter: Optional[sp.Symbol] = None):
        if parameter is None:
            parameter = sp.Symbol('lambda', real=True)
        coords = self.symbols()
        metric = self.metric(coords)
        if metric is None:
            raise ValueError("Chart does not define a metric.")
        if functions is None:
            functions = tuple(sp.Function(str(c))(parameter) for c in coords)
        subs = dict(zip(coords, functions))
        velocities = [sp.diff(func, parameter) for func in functions]
        integrals = {}
        for idx in self.cyclic_coordinates(coords):
            momentum = 0
            for j in range(self.dimension):
                momentum += metric[idx, j].subs(subs) * velocities[j]
            integrals[coords[idx]] = self.cleanup_coordinate_expr(momentum, coords)
        return integrals

    def metric_lie_derivative(self, vector_components, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        metric = self.metric(coords)
        if metric is None:
            return None
        if len(vector_components) != self.dimension:
            raise ValueError("vector_components must match chart dimension.")
        return tnf_build_array(
            (self.dimension, self.dimension),
            lambda idx: sum(
                vector_components[k] * sp.diff(metric[idx[0], idx[1]], coords[k])
                + metric[k, idx[1]] * sp.diff(vector_components[k], coords[idx[0]])
                + metric[idx[0], k] * sp.diff(vector_components[k], coords[idx[1]])
                for k in range(self.dimension)
            ),
        )

    def is_killing_vector(self, vector_components, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> bool:
        lied = self.metric_lie_derivative(vector_components, coords)
        if lied is None:
            return False
        return all(is_zero(lied[i, j]) for i in range(self.dimension) for j in range(self.dimension))

    def lower_riemann_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        metric = self.metric(coords)
        riem = self.riemann_tensor(coords)
        if metric is None or riem is None:
            return None
        return tnf_build_array(
            (self.dimension, self.dimension, self.dimension, self.dimension),
            lambda idx: sum(metric[idx[0], m] * riem[m, idx[1], idx[2], idx[3]] for m in range(self.dimension)),
        )

    def einstein_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        metric = self.metric(coords)
        ric = self.ricci_tensor(coords)
        scalar = self.scalar_curvature(coords)
        if metric is None or ric is None or scalar is None:
            return None
        return _cached_chart_value(self, 'einstein_tensor', coords, None, lambda: tnf_build_array((self.dimension, self.dimension), lambda idx: ric[idx] - sp.Rational(1, 2) * scalar * metric[idx]))

    def schouten_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        n = self.dimension
        metric = self.metric(coords)
        ric = self.ricci_tensor(coords)
        scalar = self.scalar_curvature(coords)
        if metric is None or ric is None or scalar is None:
            return None
        if n < 3:
            return tnf_build_array((n, n), lambda idx: sp.Integer(0))
        return tnf_build_array((n, n), lambda idx: (ric[idx] - scalar * metric[idx] / (2 * (n - 1))) / (n - 2))

    def weyl_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        n = self.dimension
        if n < 4:
            return tnf_build_array((n, n, n, n), lambda idx: sp.Integer(0))
        metric = self.metric(coords)
        riem = self.lower_riemann_tensor(coords)
        schouten = self.schouten_tensor(coords)
        if metric is None or riem is None or schouten is None:
            return None
        return tnf_build_array(
            (n, n, n, n),
            lambda idx: riem[idx]
            - (
                metric[idx[0], idx[2]] * schouten[idx[1], idx[3]]
                - metric[idx[0], idx[3]] * schouten[idx[1], idx[2]]
                - metric[idx[1], idx[2]] * schouten[idx[0], idx[3]]
                + metric[idx[1], idx[3]] * schouten[idx[0], idx[2]]
            ),
        )

    def curvature_decomposition(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Dict[str, object]:
        if coords is None:
            coords = self.symbols()
        return {
            "riemann_mixed": self.riemann_tensor(coords),
            "riemann_lowered": self.lower_riemann_tensor(coords),
            "ricci": self.ricci_tensor(coords),
            "scalar_curvature": self.scalar_curvature(coords),
            "einstein_tensor": self.einstein_tensor(coords),
            "weyl_tensor": self.weyl_tensor(coords),
            "cyclic_coordinates": self.cyclic_coordinates(coords),
            "einstein": self.einstein_tensor(coords),
            "schouten": self.schouten_tensor(coords),
            "weyl": self.weyl_tensor(coords),
        }

    def kulkarni_nomizu_product(self, left: TNFTensorArray, right: TNFTensorArray) -> TNFTensorArray:
        """Return the Kulkarni–Nomizu product of two symmetric rank-2 tensors."""
        n = self.dimension
        return tnf_build_array(
            (n, n, n, n),
            lambda idx: (
                left[idx[0], idx[2]] * right[idx[1], idx[3]]
                - left[idx[0], idx[3]] * right[idx[1], idx[2]]
                - left[idx[1], idx[2]] * right[idx[0], idx[3]]
                + left[idx[1], idx[3]] * right[idx[0], idx[2]]
            ),
        )

    def ricci_decomposition(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[Dict[str, TNFTensorArray]]:
        if coords is None:
            coords = self.symbols()
        metric = self.metric(coords)
        weyl = self.weyl_tensor(coords)
        schouten = self.schouten_tensor(coords)
        riem = self.lower_riemann_tensor(coords)
        if metric is None or weyl is None or schouten is None or riem is None:
            return None
        schouten_part = self.kulkarni_nomizu_product(metric, schouten)
        return {
            "riemann": riem,
            "weyl": weyl,
            "schouten_part": schouten_part,
        }

    def ricci_square(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if coords is None:
            coords = self.symbols()
        ric = self.ricci_tensor(coords)
        ginv = self.inverse_metric(coords)
        if ric is None or ginv is None:
            return None
        total = sp.Integer(0)
        n = self.dimension
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        total += ginv[i, k] * ginv[j, l] * ric[i, j] * ric[k, l]
        return canonical_simplify(total)

    def kretschmann_scalar(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if coords is None:
            coords = self.symbols()
        riem = self.lower_riemann_tensor(coords)
        ginv = self.inverse_metric(coords)
        if riem is None or ginv is None:
            return None
        n = self.dimension
        total = sp.Integer(0)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        for e in range(n):
                            for f in range(n):
                                for g in range(n):
                                    for h in range(n):
                                        total += ginv[a, e] * ginv[b, f] * ginv[c, g] * ginv[d, h] * riem[a, b, c, d] * riem[e, f, g, h]
        return canonical_simplify(total)

    def weyl_square(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if coords is None:
            coords = self.symbols()
        weyl = self.weyl_tensor(coords)
        ginv = self.inverse_metric(coords)
        if weyl is None or ginv is None:
            return None
        n = self.dimension
        total = sp.Integer(0)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        for e in range(n):
                            for f in range(n):
                                for g in range(n):
                                    for h in range(n):
                                        total += ginv[a, e] * ginv[b, f] * ginv[c, g] * ginv[d, h] * weyl[a, b, c, d] * weyl[e, f, g, h]
        return canonical_simplify(total)

    def sectional_curvature(self, u, v, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if coords is None:
            coords = self.symbols()
        riem = self.lower_riemann_tensor(coords)
        metric = self.metric(coords)
        if riem is None or metric is None:
            return None
        if len(u) != self.dimension or len(v) != self.dimension:
            raise ValueError("u and v must match chart dimension.")
        num = sp.Integer(0)
        den_uu = sp.Integer(0)
        den_vv = sp.Integer(0)
        den_uv = sp.Integer(0)
        n = self.dimension
        for a in range(n):
            for b in range(n):
                den_uu += metric[a, b] * u[a] * u[b]
                den_vv += metric[a, b] * v[a] * v[b]
                den_uv += metric[a, b] * u[a] * v[b]
                for c in range(n):
                    for d in range(n):
                        num += riem[a, b, c, d] * u[a] * v[b] * u[c] * v[d]
        return canonical_simplify(num / (den_uu * den_vv - den_uv**2))

    def curvature_invariants(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Dict[str, Optional[sp.Expr]]:
        if coords is None:
            coords = self.symbols()
        if self.metric_name in {"Euclidean", "Minkowski"}:
            return {
                "scalar_curvature": sp.Integer(0),
                "ricci_square": sp.Integer(0),
                "kretschmann_scalar": sp.Integer(0),
                "weyl_square": sp.Integer(0),
            }
        return {
            "scalar_curvature": self.scalar_curvature(coords),
            "ricci_square": self.ricci_square(coords),
            "kretschmann_scalar": self.kretschmann_scalar(coords),
            "weyl_square": self.weyl_square(coords),
        }
    def orthonormal_frame_matrix_tnf(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
        if coords is None:
            coords = self.symbols()
        if not self.is_orthogonal(coords):
            raise ValueError("Orthonormal frame support currently requires an orthogonal chart.")
        scale_factors = self.scale_factors(coords)
        return tnf_build_matrix(self.dimension, self.dimension, lambda i, j: canonical_simplify(1 / scale_factors[i], final=False) if i == j else sp.Integer(0))

    def orthonormal_frame_matrix(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
        return tnf_matrix_to_sympy(self.orthonormal_frame_matrix_tnf(coords))

    def orthonormal_coframe_matrix_tnf(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
        if coords is None:
            coords = self.symbols()
        if not self.is_orthogonal(coords):
            raise ValueError("Orthonormal frame support currently requires an orthogonal chart.")
        scale_factors = self.scale_factors(coords)
        return tnf_build_matrix(self.dimension, self.dimension, lambda i, j: canonical_simplify(scale_factors[i], final=False) if i == j else sp.Integer(0))

    def orthonormal_coframe_matrix(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
        return tnf_matrix_to_sympy(self.orthonormal_coframe_matrix_tnf(coords))

    def is_orthogonal_metric(self) -> Optional[bool]:
        metric_tnf = self.metric()
        return _metric_is_orthogonal(metric_tnf)

    def chart_properties(self) -> Tuple[str, ...]:
        keys = list(_all_chart_family_properties())
        extra = [k for k in self.metadata.keys() if k not in keys]
        return tuple(keys + extra)


    def christoffel_symbols(self, coords: Optional[Tuple[sp.Symbol, ...]] = None, kind: str = "second") -> Optional[TNFTensorArray]:
        return _compute_christoffel_symbols(self, coords, kind)

    def christoffel_symbols_sympy(self, coords: Optional[Tuple[sp.Symbol, ...]] = None, kind: str = "second") -> Optional[sp.MutableDenseNDimArray]:
        gamma = self.christoffel_symbols(coords, kind=kind)
        return None if gamma is None else gamma.to_sympy()

    def riemann_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        return _compute_riemann_tensor(self, coords)

    def riemann_tensor_sympy(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.MutableDenseNDimArray]:
        tensor = self.riemann_tensor(coords)
        return None if tensor is None else tensor.to_sympy()

    def ricci_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        return _compute_ricci_tensor(self, coords)

    def ricci_tensor_sympy(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.MutableDenseNDimArray]:
        tensor = self.ricci_tensor(coords)
        return None if tensor is None else tensor.to_sympy()

    def scalar_curvature(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        return _compute_scalar_curvature(self, coords)

    def geodesic_equations(self, functions: Optional[Tuple[sp.Function, ...]] = None, parameter: Optional[sp.Symbol] = None):
        if parameter is None:
            parameter = sp.Symbol('lambda', real=True)
        coords = self.symbols()
        if functions is None:
            functions = tuple(sp.Function(str(c))(parameter) for c in coords)
        gamma = self.christoffel_symbols(coords)
        if gamma is None:
            raise ValueError("Chart does not define a metric.")
        subs = dict(zip(coords, functions))
        eqs = []
        for i in range(self.dimension):
            total = sp.diff(functions[i], parameter, 2)
            for j in range(self.dimension):
                for k in range(self.dimension):
                    total += canonical_simplify(gamma[i, j, k].subs(subs)) * sp.diff(functions[j], parameter) * sp.diff(functions[k], parameter)
            eqs.append(sp.Eq(canonical_simplify(total), 0))
        return tuple(eqs)

    def geodesic_lagrangian(self, functions: Optional[Tuple[sp.Function, ...]] = None, parameter: Optional[sp.Symbol] = None) -> sp.Expr:
        if parameter is None:
            parameter = sp.Symbol('lambda', real=True)
        coords = self.symbols()
        metric = self.metric(coords)
        if metric is None:
            raise ValueError("Chart does not define a metric.")
        if functions is None:
            functions = tuple(sp.Function(str(c))(parameter) for c in coords)
        subs = dict(zip(coords, functions))
        velocities = [sp.diff(func, parameter) for func in functions]
        total = 0
        for i in range(self.dimension):
            for j in range(self.dimension):
                total += metric[i, j].subs(subs) * velocities[i] * velocities[j]
        return self.cleanup_coordinate_expr(sp.Rational(1, 2) * total, coords)

    def cyclic_coordinates(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Tuple[int, ...]:
        if coords is None:
            coords = self.symbols()
        metric = self.metric(coords)
        if metric is None:
            return tuple()
        cyclic = []
        for idx, coord in enumerate(coords):
            if all(is_zero(sp.diff(metric[i, j], coord)) for i in range(self.dimension) for j in range(self.dimension)):
                cyclic.append(idx)
        return tuple(cyclic)

    def coordinate_killing_vectors(self) -> Tuple[Tuple[sp.Expr, ...], ...]:
        out = []
        for idx in self.cyclic_coordinates():
            comps = [sp.Integer(0)] * self.dimension
            comps[idx] = sp.Integer(1)
            out.append(tuple(comps))
        return tuple(out)

    def geodesic_first_integrals(self, functions: Optional[Tuple[sp.Function, ...]] = None, parameter: Optional[sp.Symbol] = None):
        if parameter is None:
            parameter = sp.Symbol('lambda', real=True)
        coords = self.symbols()
        metric = self.metric(coords)
        if metric is None:
            raise ValueError("Chart does not define a metric.")
        if functions is None:
            functions = tuple(sp.Function(str(c))(parameter) for c in coords)
        subs = dict(zip(coords, functions))
        velocities = [sp.diff(func, parameter) for func in functions]
        integrals = {}
        for idx in self.cyclic_coordinates(coords):
            momentum = 0
            for j in range(self.dimension):
                momentum += metric[idx, j].subs(subs) * velocities[j]
            integrals[coords[idx]] = self.cleanup_coordinate_expr(momentum, coords)
        return integrals

    def metric_lie_derivative(self, vector_components, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        metric = self.metric(coords)
        if metric is None:
            return None
        if len(vector_components) != self.dimension:
            raise ValueError("vector_components must match chart dimension.")
        return tnf_build_array(
            (self.dimension, self.dimension),
            lambda idx: sum(
                vector_components[k] * sp.diff(metric[idx[0], idx[1]], coords[k])
                + metric[k, idx[1]] * sp.diff(vector_components[k], coords[idx[0]])
                + metric[idx[0], k] * sp.diff(vector_components[k], coords[idx[1]])
                for k in range(self.dimension)
            ),
        )

    def is_killing_vector(self, vector_components, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> bool:
        lied = self.metric_lie_derivative(vector_components, coords)
        if lied is None:
            return False
        return all(is_zero(lied[i, j]) for i in range(self.dimension) for j in range(self.dimension))

    def lower_riemann_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        metric = self.metric(coords)
        riem = self.riemann_tensor(coords)
        if metric is None or riem is None:
            return None
        return tnf_build_array(
            (self.dimension, self.dimension, self.dimension, self.dimension),
            lambda idx: sum(metric[idx[0], m] * riem[m, idx[1], idx[2], idx[3]] for m in range(self.dimension)),
        )

    def einstein_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        metric = self.metric(coords)
        ric = self.ricci_tensor(coords)
        scalar = self.scalar_curvature(coords)
        if metric is None or ric is None or scalar is None:
            return None
        return _cached_chart_value(self, 'einstein_tensor', coords, None, lambda: tnf_build_array((self.dimension, self.dimension), lambda idx: ric[idx] - sp.Rational(1, 2) * scalar * metric[idx]))

    def schouten_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        n = self.dimension
        metric = self.metric(coords)
        ric = self.ricci_tensor(coords)
        scalar = self.scalar_curvature(coords)
        if metric is None or ric is None or scalar is None:
            return None
        if n < 3:
            return tnf_build_array((n, n), lambda idx: sp.Integer(0))
        return tnf_build_array((n, n), lambda idx: (ric[idx] - scalar * metric[idx] / (2 * (n - 1))) / (n - 2))

    def weyl_tensor(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[TNFTensorArray]:
        if coords is None:
            coords = self.symbols()
        n = self.dimension
        if n < 4:
            return tnf_build_array((n, n, n, n), lambda idx: sp.Integer(0))
        metric = self.metric(coords)
        riem = self.lower_riemann_tensor(coords)
        schouten = self.schouten_tensor(coords)
        if metric is None or riem is None or schouten is None:
            return None
        return tnf_build_array(
            (n, n, n, n),
            lambda idx: riem[idx]
            - (
                metric[idx[0], idx[2]] * schouten[idx[1], idx[3]]
                - metric[idx[0], idx[3]] * schouten[idx[1], idx[2]]
                - metric[idx[1], idx[2]] * schouten[idx[0], idx[3]]
                + metric[idx[1], idx[3]] * schouten[idx[0], idx[2]]
            ),
        )

    def curvature_decomposition(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Dict[str, object]:
        if coords is None:
            coords = self.symbols()
        return {
            "riemann_mixed": self.riemann_tensor(coords),
            "riemann_lowered": self.lower_riemann_tensor(coords),
            "ricci": self.ricci_tensor(coords),
            "scalar_curvature": self.scalar_curvature(coords),
            "einstein_tensor": self.einstein_tensor(coords),
            "weyl_tensor": self.weyl_tensor(coords),
            "cyclic_coordinates": self.cyclic_coordinates(coords),
            "einstein": self.einstein_tensor(coords),
            "schouten": self.schouten_tensor(coords),
            "weyl": self.weyl_tensor(coords),
        }

    def kulkarni_nomizu_product(self, left: TNFTensorArray, right: TNFTensorArray) -> TNFTensorArray:
        """Return the Kulkarni–Nomizu product of two symmetric rank-2 tensors."""
        n = self.dimension
        return tnf_build_array(
            (n, n, n, n),
            lambda idx: (
                left[idx[0], idx[2]] * right[idx[1], idx[3]]
                - left[idx[0], idx[3]] * right[idx[1], idx[2]]
                - left[idx[1], idx[2]] * right[idx[0], idx[3]]
                + left[idx[1], idx[3]] * right[idx[0], idx[2]]
            ),
        )

    def ricci_decomposition(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[Dict[str, TNFTensorArray]]:
        if coords is None:
            coords = self.symbols()
        metric = self.metric(coords)
        weyl = self.weyl_tensor(coords)
        schouten = self.schouten_tensor(coords)
        riem = self.lower_riemann_tensor(coords)
        if metric is None or weyl is None or schouten is None or riem is None:
            return None
        schouten_part = self.kulkarni_nomizu_product(metric, schouten)
        return {
            "riemann": riem,
            "weyl": weyl,
            "schouten_part": schouten_part,
        }

    def ricci_square(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if coords is None:
            coords = self.symbols()
        ric = self.ricci_tensor(coords)
        ginv = self.inverse_metric(coords)
        if ric is None or ginv is None:
            return None
        total = sp.Integer(0)
        n = self.dimension
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        total += ginv[i, k] * ginv[j, l] * ric[i, j] * ric[k, l]
        return canonical_simplify(total)

    def kretschmann_scalar(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if coords is None:
            coords = self.symbols()
        riem = self.lower_riemann_tensor(coords)
        ginv = self.inverse_metric(coords)
        if riem is None or ginv is None:
            return None
        n = self.dimension
        total = sp.Integer(0)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        for e in range(n):
                            for f in range(n):
                                for g in range(n):
                                    for h in range(n):
                                        total += ginv[a, e] * ginv[b, f] * ginv[c, g] * ginv[d, h] * riem[a, b, c, d] * riem[e, f, g, h]
        return canonical_simplify(total)

    def weyl_square(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if coords is None:
            coords = self.symbols()
        weyl = self.weyl_tensor(coords)
        ginv = self.inverse_metric(coords)
        if weyl is None or ginv is None:
            return None
        n = self.dimension
        total = sp.Integer(0)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        for e in range(n):
                            for f in range(n):
                                for g in range(n):
                                    for h in range(n):
                                        total += ginv[a, e] * ginv[b, f] * ginv[c, g] * ginv[d, h] * weyl[a, b, c, d] * weyl[e, f, g, h]
        return canonical_simplify(total)

    def sectional_curvature(self, u, v, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Optional[sp.Expr]:
        if coords is None:
            coords = self.symbols()
        riem = self.lower_riemann_tensor(coords)
        metric = self.metric(coords)
        if riem is None or metric is None:
            return None
        if len(u) != self.dimension or len(v) != self.dimension:
            raise ValueError("u and v must match chart dimension.")
        num = sp.Integer(0)
        den_uu = sp.Integer(0)
        den_vv = sp.Integer(0)
        den_uv = sp.Integer(0)
        n = self.dimension
        for a in range(n):
            for b in range(n):
                den_uu += metric[a, b] * u[a] * u[b]
                den_vv += metric[a, b] * v[a] * v[b]
                den_uv += metric[a, b] * u[a] * v[b]
                for c in range(n):
                    for d in range(n):
                        num += riem[a, b, c, d] * u[a] * v[b] * u[c] * v[d]
        return canonical_simplify(num / (den_uu * den_vv - den_uv**2))

    def curvature_invariants(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> Dict[str, Optional[sp.Expr]]:
        if coords is None:
            coords = self.symbols()
        if self.metric_name in {"Euclidean", "Minkowski"}:
            return {
                "scalar_curvature": sp.Integer(0),
                "ricci_square": sp.Integer(0),
                "kretschmann_scalar": sp.Integer(0),
                "weyl_square": sp.Integer(0),
            }
        return {
            "scalar_curvature": self.scalar_curvature(coords),
            "ricci_square": self.ricci_square(coords),
            "kretschmann_scalar": self.kretschmann_scalar(coords),
            "weyl_square": self.weyl_square(coords),
        }
    def orthonormal_frame_matrix_tnf(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
        if coords is None:
            coords = self.symbols()
        if not self.is_orthogonal(coords):
            raise ValueError("Orthonormal frame support currently requires an orthogonal chart.")
        scale_factors = self.scale_factors(coords)
        return tnf_build_matrix(self.dimension, self.dimension, lambda i, j: canonical_simplify(1 / scale_factors[i], final=False) if i == j else sp.Integer(0))

    def orthonormal_frame_matrix(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
        return tnf_matrix_to_sympy(self.orthonormal_frame_matrix_tnf(coords))

    def orthonormal_coframe_matrix_tnf(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> TNFMatrix:
        if coords is None:
            coords = self.symbols()
        if not self.is_orthogonal(coords):
            raise ValueError("Orthonormal frame support currently requires an orthogonal chart.")
        scale_factors = self.scale_factors(coords)
        return tnf_build_matrix(self.dimension, self.dimension, lambda i, j: canonical_simplify(scale_factors[i], final=False) if i == j else sp.Integer(0))

    def orthonormal_coframe_matrix(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> sp.Matrix:
        return tnf_matrix_to_sympy(self.orthonormal_coframe_matrix_tnf(coords))

    def is_orthogonal(self, coords: Optional[Tuple[sp.Symbol, ...]] = None) -> bool:
        return bool(_metric_is_orthogonal(self.metric(coords)))


    def killing_equations(self, components=None, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        """Return the linear first-order PDE system L_X g = 0 for a Killing vector field X.

        If components is None, create unknown scalar functions X0(coords), ..., X{n-1}(coords).
        """
        if coords is None:
            coords = self.symbols()
        if components is None:
            components = tuple(sp.Function(f"X{idx}")(*coords) for idx in range(self.dimension))
        if len(components) != self.dimension:
            raise ValueError("components must have one entry per coordinate.")
        lied = self.metric_lie_derivative(components, coords)
        eqs = []
        for i in range(self.dimension):
            for j in range(i, self.dimension):
                eqs.append(sp.Eq(self.cleanup_coordinate_expr(lied[i, j], coords), 0))
        return tuple(eqs)

    def solve_killing_vectors_affine(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        """Solve the Killing equations in the affine ansatz X^i = a^i + B^i_j x^j.

        This captures translations and linear rotational/boost-type symmetries in flat metrics.
        """
        if coords is None:
            coords = self.symbols()
        a = sp.symbols(f'a0:{self.dimension}', real=True)
        B = sp.Matrix(self.dimension, self.dimension, lambda i, j: sp.Symbol(f'B{i}{j}', real=True))
        comps = tuple(self.cleanup_coordinate_expr(a[i] + sum(B[i, j] * coords[j] for j in range(self.dimension)), coords) for i in range(self.dimension))
        eqs = [eq.lhs for eq in self.killing_equations(comps, coords)]
        scalar_eqs = []
        for expr in eqs:
            expr = sp.expand(expr)
            poly = sp.Poly(expr, *coords, domain='EX')
            scalar_eqs.extend(poly.coeffs())
        unknowns = list(a) + list(B)
        sol = sp.solve([sp.Eq(self.cleanup_coordinate_expr(e, coords), 0) for e in scalar_eqs], unknowns, dict=True)
        families = []
        for s in sol:
            families.append(tuple(self.cleanup_coordinate_expr(c.subs(s), coords) for c in comps))
        return tuple(families)

    def geodesic_rhs(self, state, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        """Return the first-order RHS for geodesic flow.

        state = (q_0,...,q_{n-1}, v_0,...,v_{n-1})
        returns (v_0,...,v_{n-1}, a_0,...,a_{n-1}) with a^i = -Gamma^i_{jk} v^j v^k.
        """
        if coords is None:
            coords = self.symbols()
        if len(state) != 2 * self.dimension:
            raise ValueError("state must have length 2*dimension.")
        q = tuple(state[:self.dimension])
        v = tuple(state[self.dimension:])
        gamma = self.christoffel_symbols(coords)
        if gamma is None:
            raise ValueError("Chart does not define a metric.")
        subs = dict(zip(coords, q))
        acc = []
        for i in range(self.dimension):
            total = 0
            for j in range(self.dimension):
                for k in range(self.dimension):
                    total += gamma[i, j, k].subs(subs) * v[j] * v[k]
            acc.append(self.cleanup_coordinate_expr(-total, coords))
        return tuple(v) + tuple(acc)

    def integrate_geodesic(self, initial_position, initial_velocity, parameter_values):
        """Numerically integrate the geodesic equation with RK4.

        parameter_values should be a monotone iterable of parameter samples.
        """
        pts = list(parameter_values)
        if len(initial_position) != self.dimension or len(initial_velocity) != self.dimension:
            raise ValueError("Initial position/velocity must match the chart dimension.")
        if len(pts) < 2:
            raise ValueError("Need at least two parameter values.")
        coords = self.symbols()
        state = [sp.N(v) for v in tuple(initial_position) + tuple(initial_velocity)]
        out = [(pts[0], tuple(state[:self.dimension]), tuple(state[self.dimension:]))]

        def f(st):
            return [sp.N(x) for x in self.geodesic_rhs(st, coords)]

        for t0, t1 in zip(pts[:-1], pts[1:]):
            h = sp.N(t1 - t0)
            k1 = f(state)
            k2 = f([state[i] + h * k1[i] / 2 for i in range(len(state))])
            k3 = f([state[i] + h * k2[i] / 2 for i in range(len(state))])
            k4 = f([state[i] + h * k3[i] for i in range(len(state))])
            state = [sp.N(state[i] + h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6) for i in range(len(state))]
            out.append((t1, tuple(state[:self.dimension]), tuple(state[self.dimension:])))
        return tuple(out)

    def compile_geodesic_rhs(self, coords: Optional[Tuple[sp.Symbol, ...]] = None):
        """Return a fast numeric RHS callable for geodesic flow using lambdified Christoffel symbols."""
        if coords is None:
            coords = self.symbols()
        gamma = self.christoffel_symbols(coords)
        if gamma is None:
            raise ValueError("Chart does not define a metric.")
        gamma_funcs = [[[sp.lambdify(coords, gamma[i, j, k], 'math') for k in range(self.dimension)] for j in range(self.dimension)] for i in range(self.dimension)]
        dim = self.dimension
        def rhs_numeric(state):
            q = tuple(float(state[i]) for i in range(dim))
            v = tuple(float(state[dim + i]) for i in range(dim))
            acc = []
            for i in range(dim):
                total = 0.0
                for j in range(dim):
                    for k in range(dim):
                        total += gamma_funcs[i][j][k](*q) * v[j] * v[k]
                acc.append(-total)
            return tuple(v) + tuple(acc)
        return rhs_numeric

    def integrate_geodesic_uniform(self, initial_position, initial_velocity, t0, t1, steps: int = 100):
        if steps < 1:
            raise ValueError("steps must be positive.")
        rhs = self.compile_geodesic_rhs()
        h = float(sp.N((t1 - t0) / steps))
        state = [float(sp.N(v)) for v in tuple(initial_position) + tuple(initial_velocity)]
        out = [(float(sp.N(t0)), tuple(state[:self.dimension]), tuple(state[self.dimension:]))]
        for k in range(steps):
            t = float(sp.N(t0 + k * h))
            k1 = rhs(state)
            k2 = rhs([state[i] + h * k1[i] / 2.0 for i in range(len(state))])
            k3 = rhs([state[i] + h * k2[i] / 2.0 for i in range(len(state))])
            k4 = rhs([state[i] + h * k3[i] for i in range(len(state))])
            state = [state[i] + h * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0 for i in range(len(state))]
            out.append((t + h, tuple(state[:self.dimension]), tuple(state[self.dimension:])))
        return tuple(out)

    def sample_geodesic_positions(self, initial_position, initial_velocity, t0, t1, steps: int = 100):
        return tuple((t, q) for (t, q, _v) in self.integrate_geodesic_uniform(initial_position, initial_velocity, t0, t1, steps=steps))

    def integrate_geodesic_adaptive(self, initial_position, initial_velocity, t0, t1, *, dt: float | None = None, tol: float = 1e-6, max_steps: int = 10000):
        rhs = self.compile_geodesic_rhs()
        if dt is None:
            span = abs(float(sp.N(t1 - t0)))
            dt = span / 50.0 if span else 0.1
        t = float(sp.N(t0))
        target = float(sp.N(t1))
        sign = 1.0 if target >= t else -1.0
        h = abs(float(dt)) * sign
        state = [float(sp.N(v)) for v in tuple(initial_position) + tuple(initial_velocity)]
        out = [(t, tuple(state[:self.dimension]), tuple(state[self.dimension:]))]

        def rk4_step(y, step):
            k1 = rhs(y)
            k2 = rhs([y[i] + step * k1[i] / 2.0 for i in range(len(y))])
            k3 = rhs([y[i] + step * k2[i] / 2.0 for i in range(len(y))])
            k4 = rhs([y[i] + step * k3[i] for i in range(len(y))])
            return [y[i] + step * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0 for i in range(len(y))]

        steps = 0
        while (target - t) * sign > 1e-15 and steps < max_steps:
            if abs(h) > abs(target - t):
                h = target - t
            big = rk4_step(state, h)
            half = rk4_step(state, h / 2.0)
            half = rk4_step(half, h / 2.0)
            err = max(abs(big[i] - half[i]) for i in range(len(state)))
            if err <= tol or abs(h) <= 1e-12:
                t += h
                state = half
                out.append((t, tuple(state[:self.dimension]), tuple(state[self.dimension:])))
                if err < tol / 8.0:
                    h *= 2.0
            else:
                h /= 2.0
            steps += 1
        return tuple(out)

    def compile_parallel_transport_rhs(self, position_func, velocity_func):
        coords = self.symbols()
        gamma = self.christoffel_symbols(coords)
        if gamma is None:
            raise ValueError("Chart does not define a metric.")
        gamma_funcs = [[[sp.lambdify(coords, gamma[i, j, k], 'math') for k in range(self.dimension)] for j in range(self.dimension)] for i in range(self.dimension)]

        def rhs(t, V):
            q = [float(sp.N(x)) for x in position_func(t)]
            dq = [float(sp.N(x)) for x in velocity_func(t)]
            out = []
            for i in range(self.dimension):
                total = 0.0
                for j in range(self.dimension):
                    for k in range(self.dimension):
                        total += gamma_funcs[i][j][k](*q) * dq[j] * float(V[k])
                out.append(-total)
            return out

        return rhs

    def integrate_parallel_transport_adaptive(self, position_func, velocity_func, initial_vector, t0, t1, *, dt: float | None = None, tol: float = 1e-6, max_steps: int = 10000):
        rhs = self.compile_parallel_transport_rhs(position_func, velocity_func)
        if dt is None:
            span = abs(float(sp.N(t1 - t0)))
            dt = span / 50.0 if span else 0.1
        t = float(sp.N(t0))
        target = float(sp.N(t1))
        sign = 1.0 if target >= t else -1.0
        h = abs(float(dt)) * sign
        state = [float(sp.N(v)) for v in initial_vector]
        out = [(t, tuple(state))]

        def rk4_step(y, time, step):
            k1 = rhs(time, y)
            k2 = rhs(time + step / 2.0, [y[i] + step * k1[i] / 2.0 for i in range(len(y))])
            k3 = rhs(time + step / 2.0, [y[i] + step * k2[i] / 2.0 for i in range(len(y))])
            k4 = rhs(time + step, [y[i] + step * k3[i] for i in range(len(y))])
            return [y[i] + step * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) / 6.0 for i in range(len(y))]

        steps = 0
        while (target - t) * sign > 1e-15 and steps < max_steps:
            if abs(h) > abs(target - t):
                h = target - t
            big = rk4_step(state, t, h)
            half = rk4_step(state, t, h / 2.0)
            half = rk4_step(half, t + h / 2.0, h / 2.0)
            err = max(abs(big[i] - half[i]) for i in range(len(state)))
            if err <= tol or abs(h) <= 1e-12:
                t += h
                state = half
                out.append((t, tuple(state)))
                if err < tol / 8.0:
                    h *= 2.0
            else:
                h /= 2.0
            steps += 1
        return tuple(out)

    def parallel_transport_equations(self, curve_functions, vector_functions=None, parameter: Optional[sp.Symbol] = None):
        if parameter is None:
            parameter = sp.Symbol('lambda', real=True)
        if len(curve_functions) != self.dimension:
            raise ValueError("curve_functions must match chart dimension.")
        coords = self.symbols()
        gamma = self.christoffel_symbols(coords)
        if gamma is None:
            raise ValueError("Chart does not define a metric.")
        if vector_functions is None:
            vector_functions = tuple(sp.Function(f'V{idx}')(parameter) for idx in range(self.dimension))
        subs = dict(zip(coords, curve_functions))
        dq = [sp.diff(qi, parameter) for qi in curve_functions]
        eqs = []
        for i in range(self.dimension):
            total = sp.diff(vector_functions[i], parameter)
            for j in range(self.dimension):
                for k in range(self.dimension):
                    total += gamma[i, j, k].subs(subs) * dq[j] * vector_functions[k]
            eqs.append(sp.Eq(canonical_simplify(total), 0))
        return tuple(eqs)

    def integrate_parallel_transport(self, path_samples, initial_vector):
        """Numerically integrate parallel transport along a sampled path q(t).

        path_samples is an iterable of (t, q_tuple).
        """
        samples = list(path_samples)
        if len(samples) < 2:
            raise ValueError("Need at least two path samples.")
        if len(initial_vector) != self.dimension:
            raise ValueError("initial_vector must match chart dimension.")
        coords = self.symbols()
        gamma = self.christoffel_symbols(coords)
        if gamma is None:
            raise ValueError("Chart does not define a metric.")
        gamma_funcs = [[[sp.lambdify(coords, gamma[i, j, k], 'math') for k in range(self.dimension)] for j in range(self.dimension)] for i in range(self.dimension)]
        V = [float(sp.N(v)) for v in initial_vector]
        out = [(samples[0][0], tuple(V))]

        def transport_matrix(q, dq):
            return sp.Matrix(self.dimension, self.dimension, lambda i, k: sum(gamma_funcs[i][j][k](*q) * dq[j] for j in range(self.dimension)))

        for (t0, q0), (t1, q1) in zip(samples[:-1], samples[1:]):
            h = float(sp.N(t1 - t0))
            q0n = [float(sp.N(x)) for x in q0]
            q1n = [float(sp.N(x)) for x in q1]
            dq = [(q1n[i] - q0n[i]) / h for i in range(self.dimension)]
            qmid = [(a + b) / 2.0 for a, b in zip(q0n, q1n)]
            A0 = transport_matrix(q0n, dq)
            Amid = transport_matrix(qmid, dq)
            A1 = transport_matrix(q1n, dq)
            vec = sp.Matrix(V)
            k1 = -A0 * vec
            k2 = -Amid * (vec + h * k1 / 2)
            k3 = -Amid * (vec + h * k2 / 2)
            k4 = -A1 * (vec + h * k3)
            vec = sp.Matrix([float(sp.N(vec[i] + h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6)) for i in range(self.dimension)])
            V = list(vec)
            out.append((t1, tuple(V)))
        return tuple(out)

    def data(self, include_adv_geom: bool = False) -> Dict[str, object]:
        coords = self.symbols()
        payload = {
            "metric_name": self.metric_name,
            "chart_name": self.chart_name,
            "family_name": f"{self.metric_name}:{self.chart_name}",
            "dimension": self.dimension,
            "coordinates": coords,
            "coordinate_names": self.coordinate_names,
            "parameters": self.parameters(),
            "description": self.description(),
            "orthogonal_metric": self.is_orthogonal_metric(),
            "metric_tensor": self.metric(coords),
            "inverse_metric_tensor": self.inverse_metric(coords),
            "scale_factors": self.scale_factors(coords),
            "sqrt_metric_det": self.sqrt_metric_det(coords),
            "coordinate_range_assumptions": self.assumptions(coords),
            "coordinate_domains": self.coordinate_domains(),
            "available_properties": self.chart_properties(),
            "cyclic_coordinates": self.cyclic_coordinates(coords),
            **self.metadata,
        }
        if include_adv_geom:
            payload.update({
                "riemann_tensor": self.riemann_tensor(coords),
                "ricci_tensor": self.ricci_tensor(coords),
                "scalar_curvature": self.scalar_curvature(coords),
                "einstein_tensor": self.einstein_tensor(coords),
                "weyl_tensor": self.weyl_tensor(coords),
            })
        return payload


def _euclidean_cartesian_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    return sp.eye(len(coords))


def _euclidean_all_real(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    return sp.And(*[sp.Q.real(c) for c in coords])


def _polar_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    r, _theta = coords
    return sp.diag(1, r**2)


def _polar_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    r, theta = coords
    return sp.And(sp.Q.real(r), sp.Q.real(theta), sp.StrictGreaterThan(r, 0), sp.StrictLessThan(-sp.pi, theta), sp.LessThan(theta, sp.pi))


def _cyl_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    r, _theta, _z = coords
    return sp.diag(1, r**2, 1)


def _cyl_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    r, theta, z = coords
    return sp.And(sp.Q.real(r), sp.Q.real(theta), sp.Q.real(z), sp.StrictGreaterThan(r, 0), sp.StrictLessThan(-sp.pi, theta), sp.LessThan(theta, sp.pi))


def _spherical_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    r, theta, _phi = coords
    return sp.diag(1, r**2, r**2 * sp.sin(theta)**2)


def _spherical_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    r, theta, phi = coords
    return sp.And(sp.Q.real(r), sp.Q.real(theta), sp.Q.real(phi), sp.StrictGreaterThan(r, 0), sp.StrictGreaterThan(theta, 0), sp.StrictLessThan(theta, sp.pi), sp.StrictLessThan(-sp.pi, phi), sp.LessThan(phi, sp.pi))


def _elliptic_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    mu, nu = coords
    a = _default_parameters()["a"]
    h2 = a**2 * (sp.sinh(mu)**2 + sp.sin(nu)**2)
    return sp.diag(h2, h2)


def _elliptic_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    mu, nu = coords
    return sp.And(sp.Q.real(mu), sp.Q.real(nu), sp.GreaterThan(mu, 0), sp.StrictLessThan(-sp.pi, nu), sp.LessThan(nu, sp.pi))


def _parabolic_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    sigma, tau = coords
    h2 = sigma**2 + tau**2
    return sp.diag(h2, h2)


def _parabolic_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    sigma, tau = coords
    return sp.And(sp.Q.real(sigma), sp.Q.real(tau), sp.GreaterThan(tau, 0))


def _paraboloidal_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    u, v, _phi = coords
    return sp.diag(u**2 + v**2, u**2 + v**2, u**2 * v**2)


def _paraboloidal_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    u, v, phi = coords
    return sp.And(sp.Q.real(u), sp.Q.real(v), sp.Q.real(phi), sp.GreaterThan(u, 0), sp.GreaterThan(v, 0), sp.StrictLessThan(-sp.pi, phi), sp.LessThan(phi, sp.pi))


def _prolate_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    mu, nu, _phi = coords
    a = _default_parameters()["a"]
    common = a**2 * (sp.sinh(mu)**2 + sp.sin(nu)**2)
    return sp.diag(common, common, a**2 * sp.sinh(mu)**2 * sp.sin(nu)**2)


def _prolate_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    mu, nu, phi = coords
    return sp.And(sp.Q.real(mu), sp.Q.real(nu), sp.Q.real(phi), sp.GreaterThan(mu, 0), sp.GreaterThan(nu, 0), sp.StrictLessThan(nu, sp.pi), sp.StrictLessThan(-sp.pi, phi), sp.LessThan(phi, sp.pi))


def _oblate_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    mu, nu, _phi = coords
    a = _default_parameters()["a"]
    common = a**2 * (sp.sinh(mu)**2 + sp.sin(nu)**2)
    return sp.diag(common, common, a**2 * sp.cosh(mu)**2 * sp.cos(nu)**2)


def _oblate_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    mu, nu, phi = coords
    return sp.And(sp.Q.real(mu), sp.Q.real(nu), sp.Q.real(phi), sp.GreaterThan(mu, 0), sp.LessThan(-sp.pi/2, nu), sp.LessThan(nu, sp.pi/2), sp.StrictLessThan(-sp.pi, phi), sp.LessThan(phi, sp.pi))


def _bispherical_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    sigma, tau, _phi = coords
    a = _default_parameters()["a"]
    denom = (sp.cosh(tau) - sp.cos(sigma))
    common = a**2 / denom**2
    return sp.diag(common, common, common * sp.sin(sigma)**2)


def _bispherical_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    sigma, tau, phi = coords
    return sp.And(sp.Q.real(sigma), sp.Q.real(tau), sp.Q.real(phi), sp.GreaterThan(sigma, 0), sp.StrictLessThan(sigma, sp.pi), sp.StrictLessThan(-sp.pi, phi), sp.LessThan(phi, sp.pi))


def _toroidal_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    tau, sigma, _phi = coords
    a = _default_parameters()["a"]
    denom = (sp.cosh(tau) - sp.cos(sigma))
    common = a**2 / denom**2
    return sp.diag(common, common, common * sp.sinh(tau)**2)


def _toroidal_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    tau, sigma, phi = coords
    return sp.And(sp.Q.real(tau), sp.Q.real(sigma), sp.Q.real(phi), sp.GreaterThan(tau, 0), sp.StrictLessThan(-sp.pi, sigma), sp.LessThan(sigma, sp.pi), sp.StrictLessThan(-sp.pi, phi), sp.LessThan(phi, sp.pi))


def _bipolar_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    sigma, tau = coords
    a = _default_parameters()["a"]
    denom = sp.cosh(tau) - sp.cos(sigma)
    common = a**2 / denom**2
    return sp.diag(common, common)


def _bipolar_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    sigma, tau = coords
    return sp.And(
        sp.Q.real(sigma),
        sp.Q.real(tau),
        sp.StrictLessThan(-sp.pi, sigma),
        sp.LessThan(sigma, sp.pi),
    )


def _parabolic_cylindrical_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    u, v, _z = coords
    common = u**2 + v**2
    return sp.diag(common, common, 1)


def _parabolic_cylindrical_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    u, v, z = coords
    return sp.And(sp.Q.real(u), sp.Q.real(v), sp.Q.real(z))


def _elliptic_cylindrical_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    mu, nu, _z = coords
    a = _default_parameters()["a"]
    common = a**2 * (sp.sinh(mu)**2 + sp.sin(nu)**2)
    return sp.diag(common, common, 1)


def _elliptic_cylindrical_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    mu, nu, z = coords
    return sp.And(
        sp.Q.real(mu), sp.Q.real(nu), sp.Q.real(z),
        sp.GreaterThan(mu, 0),
        sp.StrictLessThan(-sp.pi, nu),
        sp.LessThan(nu, sp.pi),
    )


def _ellipsoidal_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    lam, mu, nu = coords
    params = _abc_parameters()
    A = params["a"]**2
    B = params["b"]**2
    C = params["c"]**2
    hlam2 = (lam - mu) * (lam - nu) / (4 * (lam - A) * (lam - B) * (lam - C))
    hmu2 = (mu - lam) * (mu - nu) / (4 * (mu - A) * (mu - B) * (mu - C))
    hnu2 = (nu - lam) * (nu - mu) / (4 * (nu - A) * (nu - B) * (nu - C))
    return sp.diag(canonical_simplify(hlam2, final=True), canonical_simplify(hmu2, final=True), canonical_simplify(hnu2, final=True))


def _ellipsoidal_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    lam, mu, nu = coords
    params = _abc_parameters()
    A = params["a"]**2
    B = params["b"]**2
    C = params["c"]**2
    return sp.And(
        sp.Q.real(lam), sp.Q.real(mu), sp.Q.real(nu),
        sp.GreaterThan(lam, A),
        sp.GreaterThan(mu, B),
        sp.LessThan(mu, A),
        sp.GreaterThan(nu, C),
        sp.LessThan(nu, B),
    )


def _conical_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    r, mu, nu = coords
    params = _bc_parameters()
    B = params["b"]**2
    C = params["c"]**2
    hmu2 = r**2 * (mu**2 - nu**2) / ((mu**2 - B) * (mu**2 - C))
    hnu2 = r**2 * (mu**2 - nu**2) / ((B - nu**2) * (nu**2 - C))
    return sp.diag(1, canonical_simplify(hmu2, final=True), canonical_simplify(hnu2, final=True))


def _conical_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    r, mu, nu = coords
    params = _bc_parameters()
    b = params["b"]
    c = params["c"]
    return sp.And(
        sp.Q.real(r), sp.Q.real(mu), sp.Q.real(nu),
        sp.StrictGreaterThan(r, 0),
        sp.StrictGreaterThan(mu, b),
        sp.StrictGreaterThan(nu, c),
        sp.StrictLessThan(nu, b),
    )




def _minkowski_cartesian_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    _t, _x, _y, _z = coords
    return sp.diag(-1, 1, 1, 1)


def _minkowski_spherical_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    _t, r, theta, _phi = coords
    return sp.diag(-1, 1, r**2, r**2 * sp.sin(theta)**2)


def _minkowski_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    return sp.And(*(sp.Q.real(c) for c in coords))


def _rindler_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    rho, _eta, _y, _z = coords
    return sp.diag(1, -rho**2, 1, 1)


def _rindler_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    rho, eta, y, z = coords
    return sp.And(sp.Q.real(rho), sp.Q.real(eta), sp.Q.real(y), sp.Q.real(z), sp.GreaterThan(rho, 0))


def _frw_parameter_symbols() -> Dict[str, sp.Symbol]:
    return {"k": sp.Symbol("k", real=True)}


def _frw_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    t, r, theta, _phi = coords
    k = _frw_parameter_symbols()["k"]
    a = sp.Function("a")
    scale = a(t)
    radial = scale**2 / (1 - k * r**2)
    return sp.diag(-1, radial, scale**2 * r**2, scale**2 * r**2 * sp.sin(theta)**2)


def _frw_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    t, r, theta, phi = coords
    return sp.And(sp.Q.real(t), sp.Q.real(r), sp.Q.real(theta), sp.Q.real(phi), sp.GreaterThan(r, 0), sp.StrictLessThan(0, theta), sp.LessThan(theta, sp.pi))




def _minkowski_cylindrical_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    _t, rho, _phi, _z = coords
    return sp.diag(-1, 1, rho**2, 1)


def _lightcone_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    _u, _v, _x, _y = coords
    return sp.Matrix([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _lightcone_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    return sp.And(*(sp.Q.real(c) for c in coords))


def _eddington_finkelstein_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    v, r, theta, _phi = coords
    m = sp.Symbol('M', positive=True, real=True)
    f = 1 - 2*m/r
    return sp.Matrix([[-f, 1, 0, 0], [1, 0, 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sp.sin(theta)**2]])


def _schwarzschild_like_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    t, r, theta, phi = coords
    return sp.And(sp.Q.real(t), sp.Q.real(r), sp.Q.real(theta), sp.Q.real(phi), sp.GreaterThan(r, 0), sp.StrictLessThan(0, theta), sp.LessThan(theta, sp.pi))


def _kruskal_szekeres_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    u, v, theta, _phi = coords
    m = sp.Symbol('M', positive=True, real=True)
    r = sp.Function('r')(u, v)
    pref = 32 * m**3 * sp.exp(-r/(2*m)) / r
    return sp.Matrix([[-pref, 0, 0, 0], [0, pref, 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sp.sin(theta)**2]])


def _de_sitter_static_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    _t, r, theta, _phi = coords
    L = sp.Symbol('L', positive=True, real=True)
    f = 1 - r**2/L**2
    return sp.diag(-f, 1/f, r**2, r**2*sp.sin(theta)**2)



def _hyperbolic_geodesic_polar_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    r, theta = coords
    return sp.diag(1, sp.sinh(r)**2)


def _hyperboloid_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    chi, theta = coords
    return sp.diag(1, sp.sinh(chi)**2)


def _de_sitter_flat_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    t, x, y, z = coords
    H = sp.Symbol('H', positive=True, real=True)
    a = sp.exp(H * t)
    return sp.diag(-1, a**2, a**2, a**2)


def _ads_poincare_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    t, x, y, z = coords
    L = sp.Symbol('L', positive=True, real=True)
    factor = L**2 / z**2
    return sp.diag(-factor, factor, factor, factor)
def _anti_de_sitter_global_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    _t, rho, theta, _phi = coords
    L = sp.Symbol('L', positive=True, real=True)
    return sp.diag(-(1 + rho**2/L**2), 1/(1 + rho**2/L**2), rho**2, rho**2*sp.sin(theta)**2)


def _hyperbolic_polar_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    r, _theta = coords
    return sp.diag(1, sp.sinh(r)**2)


def _hyperbolic_polar_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    r, theta = coords
    return sp.And(sp.Q.real(r), sp.Q.real(theta), sp.GreaterThan(r, 0), sp.StrictLessThan(-sp.pi, theta), sp.LessThan(theta, sp.pi))

def _poincare_disk_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    u, v = coords
    factor = 4 / (1 - u**2 - v**2)**2
    return sp.diag(factor, factor)


def _poincare_disk_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    u, v = coords
    return sp.And(sp.Q.real(u), sp.Q.real(v), sp.StrictLessThan(u**2 + v**2, 1))


def _poincare_half_plane_metric(coords: Tuple[sp.Symbol, ...]) -> sp.Matrix:
    _x, y = coords
    factor = 1 / y**2
    return sp.diag(factor, factor)


def _poincare_half_plane_assumptions(coords: Tuple[sp.Symbol, ...]) -> sp.Expr:
    x, y = coords
    return sp.And(sp.Q.real(x), sp.Q.real(y), sp.GreaterThan(y, 0))

_REGISTRY: Dict[Tuple[str, str, int], CoordinateChart] = {}


def register_chart(chart: CoordinateChart) -> None:
    _REGISTRY[(chart.metric_name, chart.chart_name, chart.dimension)] = chart


def get_chart(metric_name: str, chart_name: str, dimension: int) -> CoordinateChart:
    key = (metric_name, chart_name, dimension)
    if key not in _REGISTRY:
        raise KeyError(f"No registered chart for {key}")
    return _REGISTRY[key]


def list_charts() -> Iterable[Tuple[str, str, int]]:
    return sorted(_REGISTRY.keys())


def list_charts_with_orthogonal_metric() -> Iterable[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    for key in sorted(_REGISTRY.keys()):
        chart = _REGISTRY[key]
        if chart.is_orthogonal_metric():
            out.append(key)
    return out


def list_charts_with_property(property_name: str) -> Iterable[Tuple[str, str, int]]:
    out: List[Tuple[str, str, int]] = []
    for key in sorted(_REGISTRY.keys()):
        chart = _REGISTRY[key]
        if property_name in chart.chart_properties():
            out.append(key)
    return out


def chart_property_names(chart: CoordinateChart) -> Tuple[str, ...]:
    """Return the property names exposed by a coordinate chart."""
    return chart.chart_properties()


def list_chart_family_properties(metric_name: Optional[str] = None, chart_name: Optional[str] = None) -> Dict[Tuple[str, str], Tuple[str, ...]] | Tuple[str, ...]:
    family_map: Dict[Tuple[str, str], set] = {}
    for (_metric_name, _chart_name, _dimension), chart in _REGISTRY.items():
        if metric_name is not None and _metric_name != metric_name:
            continue
        if chart_name is not None and _chart_name != chart_name:
            continue
        family_map.setdefault((_metric_name, _chart_name), set()).update(chart.chart_properties())
    if metric_name is not None and chart_name is not None:
        return tuple(sorted(family_map.get((metric_name, chart_name), set())))
    return {k: tuple(sorted(v)) for k, v in sorted(family_map.items())}


register_chart(CoordinateChart("Euclidean", "Cartesian", 2, ("x", "y"), _euclidean_all_real, _euclidean_cartesian_metric, parameter_func=None))
register_chart(CoordinateChart("Euclidean", "Cartesian", 3, ("x", "y", "z"), _euclidean_all_real, _euclidean_cartesian_metric, parameter_func=None))
register_chart(CoordinateChart("Euclidean", "Polar", 2, ("r", "theta"), _polar_assumptions, _polar_metric, parameter_func=None))
register_chart(CoordinateChart("Euclidean", "Cylindrical", 3, ("r", "theta", "z"), _cyl_assumptions, _cyl_metric, parameter_func=None))
register_chart(CoordinateChart("Euclidean", "Spherical", 3, ("r", "theta", "phi"), _spherical_assumptions, _spherical_metric, parameter_func=None))
register_chart(CoordinateChart("Euclidean", "Elliptic", 2, ("mu", "nu"), _elliptic_assumptions, _elliptic_metric))
register_chart(CoordinateChart("Euclidean", "Parabolic", 2, ("sigma", "tau"), _parabolic_assumptions, _parabolic_metric, metadata={"family_name": "Parabolic", "description": "Planar parabolic coordinates."}, parameter_func=None))
register_chart(CoordinateChart("Euclidean", "Paraboloidal", 3, ("u", "v", "phi"), _paraboloidal_assumptions, _paraboloidal_metric, metadata={"note": "Circular paraboloidal coordinates."}, parameter_func=None))
register_chart(CoordinateChart("Euclidean", "ProlateSpheroidal", 3, ("mu", "nu", "phi"), _prolate_assumptions, _prolate_metric))
register_chart(CoordinateChart("Euclidean", "OblateSpheroidal", 3, ("mu", "nu", "phi"), _oblate_assumptions, _oblate_metric))
register_chart(CoordinateChart("Euclidean", "Bispherical", 3, ("sigma", "tau", "phi"), _bispherical_assumptions, _bispherical_metric))
register_chart(CoordinateChart("Euclidean", "Toroidal", 3, ("tau", "sigma", "phi"), _toroidal_assumptions, _toroidal_metric))

register_chart(CoordinateChart("Euclidean", "Bipolar", 2, ("sigma", "tau"), _bipolar_assumptions, _bipolar_metric, metadata={
    "coordinate_domains": {
        "sigma": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
        "tau": {"kind": "real_line"},
    },
    "description": "2D bipolar coordinates with focal parameter a > 0.",
}))
register_chart(CoordinateChart("Euclidean", "ParabolicCylindrical", 3, ("u", "v", "z"), _parabolic_cylindrical_assumptions, _parabolic_cylindrical_metric, metadata={
    "note": "Parabolic cylindrical coordinates.",
    "coordinate_domains": {
        "u": {"kind": "half_line", "min": 0},
        "v": {"kind": "real_line"},
        "z": {"kind": "real_line"},
    },
}, parameter_func=None))
register_chart(CoordinateChart("Euclidean", "EllipticCylindrical", 3, ("mu", "nu", "z"), _elliptic_cylindrical_assumptions, _elliptic_cylindrical_metric, metadata={
    "coordinate_domains": {
        "mu": {"kind": "half_line", "min": 0},
        "nu": {"kind": "open_interval", "min": -sp.pi, "max": sp.pi},
        "z": {"kind": "real_line"},
    },
    "description": "3D elliptic cylindrical coordinates with semifocal distance a > 0.",
}))
register_chart(CoordinateChart("Euclidean", "Conical", 3, ("r", "mu", "nu"), _conical_assumptions, _conical_metric, metadata={
    "note": "One standard orthogonal conical coordinate parameterization.",
    "coordinate_domains": {
        "r": {"kind": "half_line", "min": 0},
        "mu": {"kind": "open_interval", "min": sp.Symbol("b", positive=True, real=True), "max": sp.oo},
        "nu": {"kind": "open_interval", "min": sp.Symbol("c", positive=True, real=True), "max": sp.Symbol("b", positive=True, real=True)},
    },
    "description": "Conical coordinates with parameters b > c > 0 and principal-octant branch conventions.",
}, parameter_func=_bc_parameters))
register_chart(CoordinateChart("Euclidean", "Ellipsoidal", 3, ("lam", "mu", "nu"), _ellipsoidal_assumptions, _ellipsoidal_metric, metadata={
    "note": "Confocal ellipsoidal coordinates in a principal-octant branch.",
    "coordinate_domains": {
        "lam": {"kind": "open_interval", "min": sp.Symbol("a", positive=True, real=True)**2, "max": sp.oo},
        "mu": {"kind": "open_interval", "min": sp.Symbol("b", positive=True, real=True)**2, "max": sp.Symbol("a", positive=True, real=True)**2},
        "nu": {"kind": "open_interval", "min": sp.Symbol("c", positive=True, real=True)**2, "max": sp.Symbol("b", positive=True, real=True)**2},
    },
    "description": "Confocal ellipsoidal coordinates with a > b > c > 0 in a principal-octant branch.",
}, parameter_func=_abc_parameters))
register_chart(CoordinateChart(
    "Euclidean",
    "Hyperspherical",
    3,
    ("r", "theta", "phi"),
    _spherical_assumptions,
    _spherical_metric,
    metadata={"note": "Metadata-only placeholder for the broader hyperspherical coordinate family."},
    parameter_func=None,
))

register_chart(CoordinateChart("Minkowski", "Cartesian", 4, ("t", "x", "y", "z"), _minkowski_assumptions, _minkowski_cartesian_metric, parameter_func=None, metadata={"signature": "-+++"}))
register_chart(CoordinateChart("Minkowski", "Spherical", 4, ("t", "r", "theta", "phi"), _frw_assumptions, _minkowski_spherical_metric, parameter_func=None, metadata={"signature": "-+++"}))
register_chart(CoordinateChart("Minkowski", "Cylindrical", 4, ("t", "rho", "phi", "z"), _rindler_assumptions, _minkowski_cylindrical_metric, parameter_func=None, metadata={"signature": "-+++"}))
register_chart(CoordinateChart("Minkowski", "LightCone", 4, ("u", "v", "x", "y"), _lightcone_assumptions, _lightcone_metric, parameter_func=None, metadata={"signature": "-+++", "description": "Null/light-cone coordinates with ds^2 = -2 du dv + dx^2 + dy^2."}))
register_chart(CoordinateChart("Rindler", "Standard", 4, ("rho", "eta", "y", "z"), _rindler_assumptions, _rindler_metric, parameter_func=None, metadata={"signature": "-+++", "description": "Rindler wedge coordinates."}))
register_chart(CoordinateChart("FRW", "Standard", 4, ("t", "r", "theta", "phi"), _frw_assumptions, _frw_metric, parameter_func=_frw_parameter_symbols, metadata={"signature": "-+++", "description": "Friedmann-Robertson-Walker metric with symbolic scale factor a(t)."}))
register_chart(CoordinateChart("Schwarzschild", "EddingtonFinkelstein", 4, ("v", "r", "theta", "phi"), _schwarzschild_like_assumptions, _eddington_finkelstein_metric, parameter_func=None, metadata={"signature": "-+++", "description": "Ingoing Eddington-Finkelstein coordinates."}))
register_chart(CoordinateChart("Schwarzschild", "KruskalSzekeres", 4, ("U", "V", "theta", "phi"), _minkowski_assumptions, _kruskal_szekeres_metric, parameter_func=None, metadata={"signature": "-+++", "description": "Kruskal-Szekeres chart with implicit Schwarzschild radius r(U,V)."}))
register_chart(CoordinateChart("deSitter", "Static", 4, ("t", "r", "theta", "phi"), _schwarzschild_like_assumptions, _de_sitter_static_metric, parameter_func=None, metadata={"signature": "-+++"}))
register_chart(CoordinateChart("antiDeSitter", "Global", 4, ("t", "rho", "theta", "phi"), _schwarzschild_like_assumptions, _anti_de_sitter_global_metric, parameter_func=None, metadata={"signature": "-+++"}))
register_chart(CoordinateChart("Hyperbolic", "GeodesicPolar", 2, ("r", "theta"), _polar_assumptions, _hyperbolic_geodesic_polar_metric, parameter_func=None, metadata={"signature": "++", "description": "Geodesic polar coordinates on the hyperbolic plane."}))
register_chart(CoordinateChart("Hyperbolic", "HyperboloidPolar", 2, ("chi", "theta"), _polar_assumptions, _hyperboloid_metric, parameter_func=None, metadata={"signature": "++", "description": "Hyperboloid-model polar chart on the hyperbolic plane."}))
register_chart(CoordinateChart("deSitter", "FlatSlicing", 4, ("t", "x", "y", "z"), _minkowski_assumptions, _de_sitter_flat_metric, parameter_func=None, metadata={"signature": "-+++"}))
register_chart(CoordinateChart("antiDeSitter", "Poincare", 4, ("t", "x", "y", "z"), _minkowski_assumptions, _ads_poincare_metric, parameter_func=None, metadata={"signature": "-+++"}))
register_chart(CoordinateChart("Hyperbolic", "PoincareDisk", 2, ("u", "v"), _poincare_disk_assumptions, _poincare_disk_metric, parameter_func=None, metadata={"description": "Poincare disk model of the hyperbolic plane."}))
register_chart(CoordinateChart("Hyperbolic", "PoincareHalfPlane", 2, ("x", "y"), _poincare_half_plane_assumptions, _poincare_half_plane_metric, parameter_func=None, metadata={"description": "Poincare upper half-plane model of the hyperbolic plane."}))
register_chart(CoordinateChart("Hyperbolic", "Polar", 2, ("r", "theta"), _hyperbolic_polar_assumptions, _hyperbolic_polar_metric, parameter_func=None, metadata={"description": "Geodesic polar coordinates on the hyperbolic plane."}))
