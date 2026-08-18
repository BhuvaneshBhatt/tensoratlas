import sympy as sp

from tensoratlas import (
    coordinate_chart,
    diagonal_tensor,
    tensor_from_components,
    hydrostatic_part,
    deviatoric_part,
    principal_invariants,
    principal_values,
    traction_vector,
    to_mandel,
    load_coordinate_catalog,
)


def test_geometry_new_charts_are_in_catalog_and_registry():
    expected = {
        ("Hyperbolic", "GeodesicPolar", 2),
        ("Hyperbolic", "HyperboloidPolar", 2),
        ("deSitter", "FlatSlicing", 4),
        ("antiDeSitter", "Poincare", 4),
    }
    for metric, chart, dim in expected:
        assert coordinate_chart(metric, chart, dim).dimension == dim
    catalog = load_coordinate_catalog()
    charts = {(item['metric'], item['chart'], item['dimension']) for item in catalog['charts']}
    assert expected.issubset(charts)


def test_curvature_invariants_vanish_for_flat_chart():
    chart = coordinate_chart("Minkowski", "Cartesian", 4)
    inv = chart.curvature_invariants()
    assert sp.simplify(inv['scalar_curvature']) == 0
    assert sp.simplify(inv['ricci_square']) == 0
    assert sp.simplify(inv['kretschmann_scalar']) == 0
    assert sp.simplify(inv['weyl_square']) == 0


def test_sectional_curvature_of_euclidean_plane_is_zero():
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    K = chart.sectional_curvature((1, 0), (0, 1))
    assert sp.simplify(K) == 0


def test_uniform_geodesic_in_cartesian_chart_is_linear_motion():
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    out = chart.integrate_geodesic_uniform((0.0, 0.0), (1.0, -2.0), 0.0, 1.0, steps=10)
    t1, q1, v1 = out[-1]
    assert abs(t1 - 1.0) < 1e-12
    assert abs(q1[0] - 1.0) < 1e-9
    assert abs(q1[1] + 2.0) < 1e-9
    assert abs(v1[0] - 1.0) < 1e-9
    assert abs(v1[1] + 2.0) < 1e-9


def test_hydrostatic_plus_deviatoric_reconstructs_tensor_and_principal_tools_work():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    tensor = diagonal_tensor(chart, [2, 3, 5], variance_spec='ll')
    obj = tensor_from_components(chart, tensor.components, 'll')
    hyd = hydrostatic_part(obj)
    dev = deviatoric_part(obj)
    total = hyd.components.to_sympy() + dev.components.to_sympy()
    assert total == obj.components.to_sympy()
    invs = principal_invariants(obj)
    assert sp.simplify(invs['I1'] - 10) == 0
    assert sp.simplify(invs['I2'] - 31) == 0
    assert sp.simplify(invs['I3'] - 30) == 0
    vals = principal_values(obj)
    assert set(vals) == {2, 3, 5}


def test_traction_and_mandel_helpers():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    tensor = diagonal_tensor(chart, [2, 3, 4], variance_spec='ll')
    tr = traction_vector(tensor_from_components(chart, tensor.components, 'll'), (1, 2, 3))
    assert tuple(tr.components.to_sympy()) == (2, 6, 12)
    mandel = to_mandel(tensor_from_components(chart, tensor.components, 'll')).to_sympy()
    assert tuple(mandel)[:3] == (2, 3, 4)
