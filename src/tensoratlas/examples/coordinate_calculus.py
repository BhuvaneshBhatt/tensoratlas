"""Coordinate-calculus tutorial examples."""

from __future__ import annotations

import sympy as sp

from tensoratlas.core import coordinate_gradient, coordinate_divergence, coordinate_laplacian, standard_coordinate_system_data


def coordinate_calculus_workflow() -> dict[str, object]:
    """Compute standard polar-coordinate scalar/vector-calculus quantities."""
    r, theta = sp.symbols("r theta", positive=True)
    data = standard_coordinate_system_data("polar")
    metric = data["metric"]
    scalar = r**2 * sp.sin(theta)
    vector = (r, sp.sin(theta))
    return {
        "coordinates": (r, theta),
        "metric": metric,
        "scalar": scalar,
        "gradient": coordinate_gradient(scalar, (r, theta), metric=metric),
        "divergence": sp.simplify(coordinate_divergence(vector, (r, theta), metric=metric)),
        "laplacian": sp.simplify(coordinate_laplacian(scalar, (r, theta), metric=metric)),
    }
