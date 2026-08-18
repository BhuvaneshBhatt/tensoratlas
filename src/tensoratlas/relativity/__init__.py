"""Relativity utilities: metric catalog, curvature tensors, and geodesics."""

from .curvature import (
    CurvatureComputer,
    christoffel_component,
    christoffel_symbols,
    inverse_metric_component,
    metric_component,
    einstein_component,
    einstein_tensor,
    ricci_component,
    ricci_tensor,
    riemann_component,
    riemann_tensor,
    scalar_curvature,
)
from .geodesics import geodesic_equation, geodesic_equations, geodesic_rhs
from .inspection import (
    nonzero_christoffel,
    nonzero_components,
    nonzero_einstein,
    nonzero_ricci,
    nonzero_riemann,
    sparse_nonzero_einstein,
    sparse_nonzero_ricci,
    sparse_nonzero_riemann,
)
from .metrics import MetricModel, flrw_metric, minkowski_metric, schwarzschild_metric, two_sphere_metric
from .plotting import geodesic_plot_2d

__all__ = [
    "MetricModel",
    "CurvatureComputer",
    "minkowski_metric",
    "two_sphere_metric",
    "schwarzschild_metric",
    "flrw_metric",
    "metric_component",
    "inverse_metric_component",
    "christoffel_symbols",
    "christoffel_component",
    "riemann_tensor",
    "riemann_component",
    "ricci_tensor",
    "ricci_component",
    "scalar_curvature",
    "einstein_tensor",
    "einstein_component",
    "geodesic_equations",
    "geodesic_equation",
    "geodesic_rhs",
    "nonzero_components",
    "nonzero_christoffel",
    "nonzero_riemann",
    "nonzero_ricci",
    "nonzero_einstein",
    "sparse_nonzero_riemann",
    "sparse_nonzero_ricci",
    "sparse_nonzero_einstein",
    "geodesic_plot_2d",
]
