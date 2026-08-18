"""Relativity tutorial examples."""

from __future__ import annotations

import sympy as sp

from tensoratlas.relativity import (
    christoffel_component,
    einstein_component,
    flrw_metric,
    ricci_component,
    riemann_component,
    scalar_curvature,
    schwarzschild_metric,
    two_sphere_metric,
)


def two_sphere_workflow() -> dict[str, object]:
    """Return a compact curvature and geodesic workflow for the radius-R sphere."""
    sphere = two_sphere_metric()
    theta, phi = sphere.coordinates
    radius = sphere.parameters[0]
    return {
        "metric": sphere.metric,
        "scalar_curvature": sp.simplify(scalar_curvature(sphere) - 2 / radius**2),
        "gamma_theta_phiphi": christoffel_component(sphere, 0, 1, 1),
        "riemann_theta_phi_theta_phi": riemann_component(sphere, 0, 1, 0, 1),
        "coordinates": (theta, phi),
    }


def schwarzschild_workflow() -> dict[str, object]:
    """Return selected Schwarzschild vacuum checks without computing every display item."""
    model = schwarzschild_metric()
    return {
        "metric": model.metric,
        "ricci_tt": sp.simplify(ricci_component(model, 0, 0)),
        "ricci_rr": sp.simplify(ricci_component(model, 1, 1)),
        "sample_christoffel": christoffel_component(model, 1, 0, 0),
    }


def flrw_workflow() -> dict[str, object]:
    """Return selected FLRW Einstein-tensor components."""
    model = flrw_metric()
    return {
        "metric": model.metric,
        "einstein_tt": einstein_component(model, 0, 0),
        "einstein_rr": einstein_component(model, 1, 1),
    }
