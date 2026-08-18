import pytest
import sympy as sp

from tensoratlas.core import TensorArray, catalog_transition_map, tensor_contract, tensor_product
from tensoratlas.display import display_nonzero_components, to_latex
from tensoratlas.errors import ContractionError, FormDegreeError, UnsupportedGeometryError
from tensoratlas.examples.usability import usability_workflow_examples
from tensoratlas.geometric_algebra import GeometricAlgebra
from tensoratlas.relativity import (
    christoffel_component,
    geodesic_equation,
    inverse_metric_component,
    metric_component,
    nonzero_christoffel,
    two_sphere_metric,
)
from tensoratlas.tensor_valued_forms import TensorValuedForm


def test_friendly_contraction_error_and_method_alias():
    arr = TensorArray.from_sympy_array(sp.Array([[[1], [2]], [[3], [4]]]))
    assert arr.shape == (2, 2, 1)
    with pytest.raises(ContractionError, match="expected a pair"):
        tensor_contract(arr, (0, 1, 2))
    product = tensor_product(((1, 2), (3, 4)), ((0, 5), (6, 7)))
    assert product.contract(1, 2).components == ((12, 19), (24, 43))


def test_coordinate_map_summary_domain_and_validation():
    cmap = catalog_transition_map("cartesian2", "polar")
    summary = cmap.summary()
    assert summary["source_coordinates"] == ("x", "y")
    assert summary["target_coordinates"] == ("r", "theta")
    assert cmap.domain_conditions()["singularities"]
    assert cmap.validate() is True


def test_relativity_selected_components_methods_and_nonzero_helpers():
    model = two_sphere_metric()
    theta, phi = model.coordinates
    assert model.validate() is True
    assert metric_component(model, 0, 0) == model.parameters[0] ** 2
    assert inverse_metric_component(model, 0, 0) == model.parameters[0] ** -2
    assert christoffel_component(model, 0, 1, 1) == -sp.sin(theta) * sp.cos(theta)
    assert model.christoffel_component(0, 1, 1) == -sp.sin(theta) * sp.cos(theta)
    assert geodesic_equation(model, 0).has(sp.Function(str(theta))(sp.Symbol("lambda")))
    assert nonzero_christoffel(model)


def test_tensor_valued_form_summary_methods_and_errors():
    omega = TensorValuedForm(1, (1, -1), {(0, 1): sp.Symbol("omega01")}, label="omega")
    assert omega.summary()["rank"] == 2
    assert omega.validate() is True
    assert omega.exterior_derivative().degree == 2
    assert omega.wedge(omega).degree == 2
    with pytest.raises(FormDegreeError):
        TensorValuedForm(1, (1, -1), {(0,): 1})


def test_geometric_algebra_summary_display_and_unsupported_metric():
    ga = GeometricAlgebra.euclidean(2)
    e1, e2 = ga.basis_vectors()
    biv = e1.wedge(e2)
    assert ga.summary()["dimension"] == 2
    assert biv.summary()["grades"] == [2]
    assert to_latex(sp.Symbol("x")) == "x"
    assert display_nonzero_components([[0, sp.Symbol("x")]], name="A") == {"A(0, 1)": sp.Symbol("x")}
    with pytest.raises(UnsupportedGeometryError):
        GeometricAlgebra(("e1", "e2"), sp.Matrix([[1, sp.Symbol("a")], [sp.Symbol("a"), 1]]))


def test_public_usability_examples_execute():
    examples = usability_workflow_examples()
    assert set(examples) == {"coordinate_map", "tensor_array", "relativity", "forms_and_ga"}
