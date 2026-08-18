import sympy as sp

from tensoratlas import (
    TensorObject,
    basis_transformation_matrix,
    coordinate_chart,
    coordinate_map,
    cotangent_basis,
    explain_zero_decision,
    indices,
    indexed,
    metric_tensor,
    possibly_zero,
    pull_back,
    push_forward,
    reconstruction_diagnostics,
    tangent_basis,
    to_indexed_tensor_form,
)
from tensoratlas.tensor_algebra import identity_tensor, tensor_product


def test_explain_zero_decision_surface():
    x = sp.Symbol("x")
    info = explain_zero_decision(sp.sin(x))
    assert info["kind"] in {"uncertain", "nonzero", "zero"}
    assert info["expression"] == "sin(x)"


def test_reconstruction_diagnostics_counts_free_and_contracted_names():
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    metric = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g")
    i, j, k, l = indices("i_ j_ k_ l_")
    expr = tensor_product(indexed(metric, i, j), indexed(metric, k, l))
    nf = to_indexed_tensor_form(expr)
    report = reconstruction_diagnostics(nf.terms[0])
    assert report.factor_count >= 1
    assert report.free_name_count >= 4
    assert report.assignment_size == 4


def test_push_forward_and_pull_back_accept_mixed_variance_tensorobject():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    mapping = coordinate_map(cart, polar)
    delta = TensorObject.from_tensor_field(identity_tensor(cart, "ul"), name="I")
    pushed = push_forward(mapping, delta)
    pulled = pull_back(mapping, delta)
    assert pushed.chart == polar
    assert pulled.chart == polar
    assert pushed.variance_spec == "ul"
    assert pulled.variance_spec == "ul"


def test_cross_chart_basis_transformation_coordinate_bases():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    coordinate_map(cart, polar)
    r, theta = polar.symbols()
    mat = basis_transformation_matrix(tangent_basis(cart), tangent_basis(polar))
    assert sp.simplify(mat[0, 0] - sp.cos(theta)) == 0
    assert sp.simplify(mat[1, 0] + sp.sin(theta) / r) == 0


def test_covariant_cross_chart_basis_transformation_coordinate_bases():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    coordinate_map(cart, polar)
    r, theta = polar.symbols()
    mat = basis_transformation_matrix(cotangent_basis(cart), cotangent_basis(polar))
    assert sp.simplify(mat[0, 0] - sp.cos(theta)) == 0
    assert sp.simplify(mat[1, 0] + r * sp.sin(theta)) == 0
