
import sympy as sp

from tensoratlas import (
    coordinate_chart,
    coordinate_map,
    chart_property_names,
    mapping_property_names,
)

def test_conical_and_ellipsoidal_inverse_maps_available():
    cart3 = coordinate_chart("Euclidean", "Cartesian", 3)
    con = coordinate_chart("Euclidean", "Conical", 3)
    ell = coordinate_chart("Euclidean", "Ellipsoidal", 3)

    con_map = coordinate_map(cart3, con)
    ell_map = coordinate_map(cart3, ell)

    assert con_map.inverse_exprs_func is not None
    assert ell_map.inverse_exprs_func is not None

def test_queryable_domain_metadata_present():
    con = coordinate_chart("Euclidean", "Conical", 3)
    ell = coordinate_chart("Euclidean", "Ellipsoidal", 3)

    con_domains = con.coordinate_domains()
    ell_domains = ell.coordinate_domains()

    assert "r" in con_domains and con_domains["r"]["kind"] == "half_line"
    assert "lam" in ell_domains and ell_domains["lam"]["kind"] == "open_interval"
    assert "coordinate_domains" in con.chart_properties()
    assert "coordinate_domains" in ell.chart_properties()

def test_property_helpers():
    chart = coordinate_chart("Euclidean", "Bipolar", 2)
    mp = coordinate_map(chart, coordinate_chart("Euclidean", "Cartesian", 2))

    cprops = chart_property_names(chart)
    mprops = mapping_property_names(mp)

    assert "coordinate_domains" in cprops
    assert "jacobian_determinant" in mprops
    assert "inverse_available" in mprops
