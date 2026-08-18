
import sympy as sp

from tensoratlas import (
    coordinate_chart,
    coordinate_map,
    list_charts_with_orthogonal_metric,
    list_charts_with_symbolic_inverse_available,
    list_charts_with_property,
    list_chart_family_properties,
)

def test_registry_queries():
    orth = set(list_charts_with_orthogonal_metric())
    assert ("Euclidean", "Polar", 2) in orth
    assert ("Euclidean", "Ellipsoidal", 3) in orth

    with_inv = set(list_charts_with_symbolic_inverse_available())
    assert ("Euclidean", "Conical", 3) in with_inv
    assert ("Euclidean", "Ellipsoidal", 3) in with_inv

    with_domains = set(list_charts_with_property("coordinate_domains"))
    assert ("Euclidean", "Cartesian", 3) in with_domains
    assert ("Euclidean", "Toroidal", 3) in with_domains

def test_chart_family_properties_and_defaults():
    props = set(list_chart_family_properties("Euclidean", "Cartesian"))
    assert "dimension" in props
    assert "coordinates" in props
    assert "metric_tensor" in props
    assert "scale_factors" in props
    assert "coordinate_domains" in props
    assert "orthogonal_metric" in props

    chart = coordinate_chart("Euclidean", "Toroidal", 3)
    data = chart.data()
    assert data["dimension"] == 3
    assert len(data["coordinates"]) == 3
    assert "tau" in data["coordinate_domains"]
    assert data["metric_tensor"].shape == (3, 3)
    assert len(data["scale_factors"]) == 3
    assert data["orthogonal_metric"] is True

def test_branch_aware_inverse_simplification_present():
    cart3 = coordinate_chart("Euclidean", "Cartesian", 3)
    con = coordinate_chart("Euclidean", "Conical", 3)
    ell = coordinate_chart("Euclidean", "Ellipsoidal", 3)

    con_map = coordinate_map(cart3, con)
    ell_map = coordinate_map(cart3, ell)

    cdata = con_map.data()
    edata = ell_map.data()

    assert cdata["inverse_available"] is True
    assert edata["inverse_available"] is True
    assert cdata["symbolic_inverse_kind"] in {"explicit", "root_based"}
    assert edata["symbolic_inverse_kind"] in {"explicit", "root_based"}
    assert len(cdata["simplified_inverse_mapping_exprs"]) == 3
    assert len(edata["simplified_inverse_mapping_exprs"]) == 3
