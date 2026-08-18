import sympy as sp

from tensoratlas import (
    coordinate_chart,
    coordinate_map,
    transform_coordinates,
    TensorField,
    TensorObject,
    diagonal_tensor,
    tensor_contract,
    tensor_dimensions,
    tensor_q,
    tensor_rank,
    tensor_symmetry,
    tensor_trace,
    tensor_array,
    tensor_element,
    tensor_graph,
    tensor_product,
    indexed,
    indices,
)
from tensoratlas.normal_forms import tnf_build_array


def test_tensor_dimensions_rank_and_q_for_tensorfield_and_scalar():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    arr = tnf_build_array((3, 3), lambda idx: sp.Symbol(f"a{idx[0]}{idx[1]}"))
    tensor = TensorField(chart, arr, "ll")
    scalar = sp.Symbol("f")
    from tensoratlas import ScalarField
    scalar_field = ScalarField(chart, scalar)
    assert tensor_q(tensor) is True
    assert tensor_dimensions(tensor) == (3, 3)
    assert tensor_rank(tensor) == 2
    assert tensor_q(scalar_field) is True
    assert tensor_dimensions(scalar_field) == ()
    assert tensor_rank(scalar_field) == 0


def test_tensor_contract_and_trace_match_rank2_identity_case():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    tensor = diagonal_tensor(chart, [1, 2, 3], variance_spec="ul")
    traced = tensor_trace(tensor)
    contracted = tensor_contract(tensor, [(0, 1)])
    assert sp.simplify(traced.expr - 6) == 0
    assert sp.simplify(contracted.expr - 6) == 0


def test_tensor_symmetry_reports_tensorobject_metadata():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    tensor = diagonal_tensor(chart, [1, 2, 3], variance_spec="ll")
    obj = TensorObject.from_tensor_field(tensor, symmetry_metadata={"symmetric": ((0, 1),)})
    assert tensor_symmetry(obj) == {"symmetric": ((0, 1),)}


def test_tensor_symmetry_detects_rank2_tensorfield_symmetry():
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    arr = tnf_build_array((2, 2), lambda idx: sp.Integer(1) if idx in ((0, 1), (1, 0)) else sp.Integer(0))
    tensor = TensorField(chart, arr, "ll")
    assert tensor_symmetry(tensor) == {"symmetric": ((0, 1),)}


def test_new_coordinate_charts_are_registered():
    assert coordinate_chart("Minkowski", "Cylindrical", 4).dimension == 4
    assert coordinate_chart("Minkowski", "LightCone", 4).dimension == 4
    assert coordinate_chart("Schwarzschild", "EddingtonFinkelstein", 4).dimension == 4
    assert coordinate_chart("Schwarzschild", "KruskalSzekeres", 4).dimension == 4
    assert coordinate_chart("deSitter", "Static", 4).dimension == 4
    assert coordinate_chart("antiDeSitter", "Global", 4).dimension == 4
    assert coordinate_chart("Hyperbolic", "Polar", 2).dimension == 2


def test_minkowski_cylindrical_cartesian_roundtrip_formulae():
    cyl = coordinate_chart("Minkowski", "Cylindrical", 4)
    cart = coordinate_chart("Minkowski", "Cartesian", 4)
    rho, phi = sp.symbols("rho phi", positive=True, real=True)
    point = (sp.Symbol("t", real=True), rho, phi, sp.Symbol("z", real=True))
    mapped = transform_coordinates(cyl, cart, point)
    assert mapped[1] == rho * sp.cos(phi)
    assert mapped[2] == rho * sp.sin(phi)
    back = coordinate_map(cart, cyl).transform_point(tuple(mapped))
    assert sp.simplify(back[1] - rho) == 0


def test_lightcone_cartesian_transformation_formulae():
    lc = coordinate_chart("Minkowski", "LightCone", 4)
    cart = coordinate_chart("Minkowski", "Cartesian", 4)
    u, v, x, y = sp.symbols("u v x y", real=True)
    mapped = transform_coordinates(lc, cart, (u, v, x, y))
    assert sp.simplify(mapped[0] - (u + v) / 2) == 0
    assert sp.simplify(mapped[1] - (v - u) / 2) == 0


def test_tensor_array_and_element_for_rank2_tensorfield():
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    arr = tnf_build_array((2, 2), lambda idx: sp.Integer(10 * idx[0] + idx[1]))
    tensor = TensorField(chart, arr, "ll")
    comp = tensor_array(tensor)
    assert comp.shape == (2, 2)
    assert tensor_element(tensor, (1, 0)) == 10


def test_tensor_array_and_element_for_scalar_field():
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    from tensoratlas import ScalarField
    scalar = ScalarField(chart, sp.Symbol('f'))
    arr = tensor_array(scalar)
    assert arr.shape == ()
    assert tensor_element(scalar, ()) == sp.Symbol('f')


def test_tensor_graph_for_indexed_tensor_product_records_contraction_edge():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    A = diagonal_tensor(chart, [1, 2, 3], variance_spec="ul")
    B = diagonal_tensor(chart, [4, 5, 6], variance_spec="ul")
    i, j_down, j_up, k = indices('i^ j_ j^ k_')
    expr = tensor_product(indexed(A, i, j_down), indexed(B, j_up, k))
    graph = tensor_graph(expr)
    assert len(graph['nodes']) == 2
    assert len(graph['edges']) == 1
    assert graph['edges'][0]['source'] == 0
    assert graph['edges'][0]['target'] == 1


def test_tensor_graph_for_tensorobject_exposes_slot_nodes():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    tensor = TensorObject.from_tensor_field(diagonal_tensor(chart, [1, 2, 3], variance_spec='ul'))
    graph = tensor_graph(tensor)
    assert len(graph['nodes']) == 2
    assert {node['variance'] for node in graph['nodes']} == {'u', 'l'}
