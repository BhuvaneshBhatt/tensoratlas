import sympy as sp

from tensoratlas.core import (
    ComponentTensor,
    CoordinateSystem,
    DifferentialForm,
    Manifold,
    TensorHead,
    exterior_derivative_form,
    metric_component_tensor,
    wedge_forms,
)


def test_wedge_and_exterior_derivative_on_coordinate_forms():
    manifold = Manifold("M", 2)
    coords = CoordinateSystem("cart", manifold, ("x", "y"))
    basis = coords.coordinate_basis()
    x, y = sp.symbols("x y")
    alpha = DifferentialForm(basis, 1, {(0,): y, (1,): -x})
    d_alpha = exterior_derivative_form(alpha, coordinates=(x, y))
    assert sp.simplify(d_alpha.component(0, 1) + 2) == 0
    dx = DifferentialForm(basis, 1, {(0,): 1})
    dy = DifferentialForm(basis, 1, {(1,): 1})
    area = wedge_forms(dx, dy)
    assert area.component(0, 1) == 1
    assert area.component(1, 0) == -1


def test_interior_product_and_hodge_star():
    manifold = Manifold("M", 2)
    coords = CoordinateSystem("cart", manifold, ("x", "y"))
    basis = coords.coordinate_basis()
    metric = metric_component_tensor("g", basis, ((1, 0), (0, 1)))
    dx = DifferentialForm(basis, 1, {(0,): 1})
    star_dx = dx.hodge_star(metric)
    assert star_dx.degree == 1
    assert sp.simplify(star_dx.component(1) - 1) == 0
    vector_head = TensorHead("X", (basis.index_type,), variance=("up",))
    vector = ComponentTensor(vector_head, basis, {(0,): 2, (1,): 3}, variance=("up",))
    two_form = DifferentialForm(basis, 2, {(0, 1): 5})
    contraction = two_form.interior_product(vector)
    assert sp.simplify(contraction.component(1) - 10) == 0
    assert sp.simplify(contraction.component(0) + 15) == 0


def test_exterior_derivative_squared_is_zero_for_one_form():
    manifold = Manifold("M", 3)
    coords = CoordinateSystem("cart", manifold, ("x", "y", "z"))
    basis = coords.coordinate_basis()
    x, y, z = sp.symbols("x y z")
    alpha = DifferentialForm(basis, 1, {(0,): x*y, (1,): z*x, (2,): y*z})
    dd_alpha = alpha.exterior_derivative(coordinates=(x, y, z)).exterior_derivative(coordinates=(x, y, z))
    assert all(sp.simplify(value) == 0 for value in dd_alpha.components.values())
