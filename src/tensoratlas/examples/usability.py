"""User-facing examples for summaries, validation, selected components, and display helpers."""

from __future__ import annotations

import sympy as sp

from tensoratlas.core import catalog_transition_map, tensor_contract, tensor_product
from tensoratlas.display import display_nonzero_components
from tensoratlas.geometric_algebra import GeometricAlgebra
from tensoratlas.relativity import (
    christoffel_component,
    geodesic_equation,
    nonzero_christoffel,
    scalar_curvature,
    two_sphere_metric,
)
from tensoratlas.tensor_valued_forms import TensorValuedForm


def coordinate_map_summary_example() -> dict[str, object]:
    cmap = catalog_transition_map("cartesian2", "polar")
    return {"summary": cmap.summary(), "domains": cmap.domain_conditions(), "valid": cmap.validate()}


def tensor_array_usability_example() -> dict[str, object]:
    a = ((1, 2), (3, 4))
    b = ((0, 5), (6, 7))
    product = tensor_product(a, b)
    contracted = tensor_contract(product, (1, 2))
    return {"summary": contracted.summary(), "components": contracted.components, "sympy": contracted.to_sympy_array()}


def relativity_usability_example() -> dict[str, object]:
    model = two_sphere_metric()
    theta_component = christoffel_component(model, 0, 1, 1)
    return {
        "metric_summary": model.summary(),
        "selected_christoffel": theta_component,
        "scalar_curvature": scalar_curvature(model),
        "one_geodesic_equation": geodesic_equation(model, 0),
        "nonzero_christoffel": nonzero_christoffel(model),
    }


def forms_and_ga_usability_example() -> dict[str, object]:
    omega = TensorValuedForm(1, (1, -1), {(0, 1): sp.Symbol("omega01")}, label="omega")
    ga = GeometricAlgebra.euclidean(2)
    e1, e2 = ga.basis_vectors()
    bivector = e1.wedge(e2)
    return {
        "form_summary": omega.summary(),
        "form_valid": omega.validate(),
        "ga_summary": ga.summary(),
        "bivector_summary": bivector.summary(),
        "display": display_nonzero_components([[0, sp.Symbol("x")], [0, 2]], name="A"),
    }


def usability_workflow_examples() -> dict[str, object]:
    """Return all usability examples as a single dictionary."""
    return {
        "coordinate_map": coordinate_map_summary_example(),
        "tensor_array": tensor_array_usability_example(),
        "relativity": relativity_usability_example(),
        "forms_and_ga": forms_and_ga_usability_example(),
    }
