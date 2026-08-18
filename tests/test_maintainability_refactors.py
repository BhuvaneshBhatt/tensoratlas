import pytest
import sympy as sp

import tensoratlas
from tensoratlas.geometric_algebra import GeometricAlgebra, rotate, project_vector_onto_vector
from tensoratlas.relativity import CurvatureComputer, two_sphere_metric
from tensoratlas.validation import ValidationReport
from tensoratlas.core.symbolic_arrays import tensor_contract, tensor_product


def test_examples_are_not_root_exports():
    assert "usability_workflow_examples" not in tensoratlas.__all__
    assert "two_sphere_relativity_workflow" not in tensoratlas.__all__


def test_geometric_algebra_accepts_basis_names_and_split_modules():
    ga = GeometricAlgebra.euclidean(2)
    e1, _e2 = ga.basis_vectors()
    assert ga.blade("e1", "e2") == ga.blade(0, 1)
    assert ga.basis_product("e1", "e1").scalar_part() == 1
    assert rotate(e1, ga.scalar(1)) == e1


def test_projection_rejects_null_or_zero_direction():
    ga = GeometricAlgebra.euclidean(2)
    e1, _e2 = ga.basis_vectors()
    with pytest.raises(ZeroDivisionError):
        project_vector_onto_vector(e1, ga.zero())


def test_curvature_computer_reuses_selected_component_workflow():
    model = two_sphere_metric()
    curv = CurvatureComputer(model, simplify=True)
    assert sp.simplify(curv.ricci(0, 0) - 1) == 0
    assert sp.simplify(curv.scalar() - model.scalar_curvature()) == 0


def test_validation_report_raise_as_uses_requested_exception():
    with pytest.raises(RuntimeError):
        ValidationReport(False, ("bad",)).raise_as(RuntimeError)


def test_tensor_contract_error_message_constant_is_user_facing():
    product = tensor_product(((1, 2), (3, 4)), ((0, 5), (6, 7)))
    with pytest.raises(Exception, match="tensor_contract expected a pair"):
        tensor_contract(product, "not a pair")
