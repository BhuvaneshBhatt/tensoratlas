import sympy as sp

from tensoratlas.core import (
    ComponentTensor,
    CoordinateSystem,
    Manifold,
    TensorHead,
    basis_one_form,
    lie_derivative,
    metric_component_tensor,
    volume_form,
)


def _cartesian_basis(dim=2):
    manifold = Manifold("M", dim)
    names = tuple("xyz"[:dim])
    coords = CoordinateSystem("cart", manifold, names)
    return coords.coordinate_basis(), sp.symbols(" ".join(names))


def test_form_addition_scaling_and_basis_covectors():
    basis, _ = _cartesian_basis(2)
    dx = basis_one_form(basis, 0)
    dy = basis_one_form(basis, 1)
    alpha = 2 * dx + 3 * dy
    assert alpha.component(0) == 2
    assert alpha.component(1) == 3
    assert (alpha - dx).component(0) == 1
    assert volume_form(basis).component(0, 1) == 1


def test_cartan_lie_derivative_on_one_form():
    basis, (x, y) = _cartesian_basis(2)
    vector_head = TensorHead("X", (basis.index_type,), variance=("up",))
    vector = ComponentTensor(vector_head, basis, {(0,): x, (1,): y}, variance=("up",))
    dx = basis_one_form(basis, 0)
    result = lie_derivative(dx, vector, coordinates=(x, y))
    assert sp.simplify(result.component(0) - 1) == 0
    assert sp.simplify(result.component(1)) == 0


def test_hodge_star_squared_on_euclidean_one_form():
    basis, _ = _cartesian_basis(2)
    metric = metric_component_tensor("g", basis, ((1, 0), (0, 1)))
    dx = basis_one_form(basis, 0)
    back = dx.hodge_star(metric).hodge_star(metric)
    assert sp.simplify(back.component(0) + 1) == 0
    assert sp.simplify(back.component(1)) == 0
