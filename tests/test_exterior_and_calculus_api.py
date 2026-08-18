import sympy as sp

from tensoratlas import (
    ScalarField,
    TensorField,
    VectorField,
    coordinate_chart,
    curl,
    divergence,
    exterior_derivative,
    gradient,
    hodge_star,
    laplacian,
    wedge,
)


def test_calculus_api_matches_field_methods():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    x, y, z = cart.symbols()
    scalar = ScalarField(cart, x**2 + y**2 + z**2)
    grad = gradient(scalar)
    assert grad.components == sp.Matrix([[2 * x], [2 * y], [2 * z]])
    vector = VectorField(cart, sp.Matrix([[y], [-x], [z]]))
    assert sp.simplify(divergence(vector) - 1) == 0
    c = curl(vector)
    assert c.components == sp.Matrix([[0], [0], [-2]])
    assert sp.simplify(laplacian(scalar) - 6) == 0


def test_exterior_derivative_wedge_and_hodge_star_on_cartesian_forms():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    one_form = TensorField(cart, sp.MutableDenseNDimArray([y, -x]), "l")
    d_form = exterior_derivative(one_form)
    assert sp.simplify(d_form.components[0, 1] + 2) == 0
    dx = TensorField(cart, sp.MutableDenseNDimArray([1, 0]), "l")
    dy = TensorField(cart, sp.MutableDenseNDimArray([0, 1]), "l")
    area = wedge(dx, dy)
    assert sp.simplify(area.components[0, 1] - 1) == 0
    star_dx = hodge_star(dx)
    assert sp.simplify(star_dx.components[0] - 0) == 0
    assert sp.simplify(star_dx.components[1] - 1) == 0
