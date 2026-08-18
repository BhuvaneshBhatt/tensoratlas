import sympy as sp

from tensoratlas.notebook_examples import (
    cartan_frame_workflow,
    electromagnetic_tensor_valued_workflow,
    executable_notebook_workflows,
    flrw_relativity_workflow,
    geometric_algebra_workflow,
    schwarzschild_relativity_workflow,
    two_sphere_relativity_workflow,
)


def test_executable_workflows_have_expected_sections():
    workflows = executable_notebook_workflows()
    assert {"electromagnetism", "two_sphere", "cartan", "schwarzschild", "flrw", "geometric_algebra"} <= set(workflows)


def test_electromagnetic_workflow_returns_curvature_form():
    result = electromagnetic_tensor_valued_workflow()
    assert result["degree"] == 2
    assert result["variance"] == (1, -1)


def test_two_sphere_workflow_scalar_matches_expected():
    result = two_sphere_relativity_workflow()
    assert sp.simplify(result["scalar_curvature"] - result["expected_scalar"]) == 0


def test_cartan_workflow_shapes():
    result = cartan_frame_workflow()
    assert result["torsion"].degree == 2
    assert result["curvature"].variance == (1, -1)


def test_schwarzschild_workflow_ricci_is_empty():
    assert schwarzschild_relativity_workflow()["nonzero_ricci"] == {}


def test_flrw_workflow_exposes_christoffel_component():
    result = flrw_relativity_workflow()
    assert "Gamma^r_tr" in result


def test_geometric_algebra_workflow_e1_square():
    result = geometric_algebra_workflow()
    assert result["e1_squared"].coeffs == {(): sp.Integer(1)}
