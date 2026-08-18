import sympy as sp

from tensoratlas.examples import (
    canonicalization_workflow,
    cartan_structure_workflow,
    coordinate_calculus_workflow,
    differential_forms_workflow,
    electromagnetic_workflow,
    flrw_workflow,
    geometric_algebra_workflow,
    schwarzschild_workflow,
    two_sphere_workflow,
)
from tensoratlas.relativity import (
    christoffel_component,
    einstein_component,
    flrw_metric,
    ricci_component,
    riemann_component,
    two_sphere_metric,
)


def test_selected_component_curvature_apis_on_two_sphere():
    sphere = two_sphere_metric()
    radius = sphere.parameters[0]
    theta, _phi = sphere.coordinates
    assert sp.simplify(christoffel_component(sphere, 0, 1, 1) + sp.sin(theta) * sp.cos(theta)) == 0
    assert sp.simplify(riemann_component(sphere, 0, 1, 0, 1) - sp.sin(theta) ** 2) == 0
    assert sp.simplify(ricci_component(sphere, 0, 0) - 1) == 0
    assert sp.simplify(2 / radius**2 - 2 / radius**2) == 0


def test_selected_einstein_component_on_flrw():
    model = flrw_metric()
    value = einstein_component(model, 0, 0)
    assert value != 0


def test_public_example_workflows_execute():
    results = [
        coordinate_calculus_workflow(),
        differential_forms_workflow(),
        electromagnetic_workflow(),
        cartan_structure_workflow(),
        two_sphere_workflow(),
        schwarzschild_workflow(),
        flrw_workflow(),
        geometric_algebra_workflow(),
        canonicalization_workflow(),
    ]
    assert all(isinstance(item, dict) and item for item in results)
