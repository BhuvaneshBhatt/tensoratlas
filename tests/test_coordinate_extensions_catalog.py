
import sympy as sp

from tensoratlas import coordinate_chart, coordinate_map, list_charts


def test_new_charts_registered():
    expected = {
        ("Euclidean", "Bipolar", 2),
        ("Euclidean", "ParabolicCylindrical", 3),
        ("Euclidean", "EllipticCylindrical", 3),
        ("Euclidean", "Conical", 3),
        ("Euclidean", "Ellipsoidal", 3),
    }
    charts = set(list_charts())
    assert expected.issubset(charts)


def test_bipolar_chart_and_map_data():
    chart = coordinate_chart("Euclidean", "Bipolar", 2)
    g = chart.metric()
    assert g.shape == (2, 2)
    assert sp.simplify(g[0, 1]) == 0
    m = coordinate_map(chart, coordinate_chart("Euclidean", "Cartesian", 2))
    assert m.jacobian().shape == (2, 2)


def test_parabolic_cylindrical_and_elliptic_cylindrical_metrics():
    pc = coordinate_chart("Euclidean", "ParabolicCylindrical", 3)
    u, v, z = pc.symbols()
    gpc = pc.metric()
    assert gpc == sp.diag(u**2 + v**2, u**2 + v**2, 1)

    ec = coordinate_chart("Euclidean", "EllipticCylindrical", 3)
    gec = ec.metric()
    assert gec.shape == (3, 3)
    assert sp.simplify(gec[2, 2] - 1) == 0

    m = coordinate_map(pc, coordinate_chart("Euclidean", "Cartesian", 3))
    assert m.jacobian().shape == (3, 3)


def test_conical_and_ellipsoidal_forward_maps():
    con = coordinate_chart("Euclidean", "Conical", 3)
    ell = coordinate_chart("Euclidean", "Ellipsoidal", 3)
    cart3 = coordinate_chart("Euclidean", "Cartesian", 3)

    con_map = coordinate_map(con, cart3)
    ell_map = coordinate_map(ell, cart3)

    assert con.metric().shape == (3, 3)
    assert ell.metric().shape == (3, 3)
    assert con_map.jacobian().shape == (3, 3)
    assert ell_map.jacobian().shape == (3, 3)
