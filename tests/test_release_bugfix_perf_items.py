import sympy as sp
import pytest

from tensoratlas.core.coordinate_map_data import (
    catalog_transition_map,
    coordinate_map_data,
    standard_coordinate_system_data,
)
from tensoratlas.errors import UnsupportedGeometryError
from tensoratlas.geometric_algebra import GeometricAlgebra, project_vector_onto_vector
from tensoratlas.relativity import christoffel_component, inverse_metric_component, two_sphere_metric
from tensoratlas.tensor_valued_forms import TensorValuedForm
from tensoratlas.tensor_valued_forms.valued import exterior_derivative_tvform


def test_geometric_algebra_natural_blade_and_basis_product_forms():
    ga = GeometricAlgebra.euclidean(2)
    assert ga.blade(0, 1).coeffs == ga.blade([0, 1]).coeffs
    assert ga.basis_product(0, 0).coeffs == {(): sp.Integer(1)}


def test_geometric_algebra_rejects_overbroad_inverse_and_projection():
    ga = GeometricAlgebra.euclidean(2)
    e1, e2 = ga.basis_vectors()
    with pytest.raises(UnsupportedGeometryError):
        (e1 + ga.blade(0, 1)).inverse()
    with pytest.raises(UnsupportedGeometryError):
        project_vector_onto_vector(e1 + ga.blade(0, 1), e2)


def test_formal_tensor_valued_derivative_is_structured_function():
    x = sp.symbols("x")
    form = TensorValuedForm(1, (1,), {(0,): x})
    derived = exterior_derivative_tvform(form)
    value = derived.components[(0,)]
    assert value == sp.Function("d")(x)


def test_metric_inverse_simplification_can_be_disabled():
    model = two_sphere_metric()
    assert inverse_metric_component(model, 0, 0, simplify=False) == model.inverse_metric[0, 0]
    theta = model.coordinates[0]
    assert christoffel_component(model, 0, 1, 1, simplify=False) == -sp.sin(theta) * sp.cos(theta)


def test_coordinate_metadata_can_skip_expensive_fields():
    cmap = catalog_transition_map("cartesian2", "polar")
    data = coordinate_map_data(cmap, include_inverse_jacobian=False, include_inverse_branches=False).as_dict()
    assert data["inverse_jacobian"] is None
    assert data["inverse_branches"] == ()
    polar = standard_coordinate_system_data("polar", include_inverse_metric=False, include_transform_properties=False)
    assert polar["inverse_metric"] is None
    assert polar["transform_properties"] is None
