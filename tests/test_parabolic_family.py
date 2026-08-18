from __future__ import annotations

import sympy as sp

from tensoratlas import coordinate_chart, coordinate_map, transform_coordinates


def test_parabolic_chart_metric_and_scale_factors():
    chart = coordinate_chart("Euclidean", "Parabolic", 2)
    sigma, tau = chart.symbols()
    metric = chart.metric((sigma, tau))
    assert metric == sp.diag(sigma**2 + tau**2, sigma**2 + tau**2)
    assert chart.is_orthogonal((sigma, tau)) is True
    assert chart.coordinate_domains()["tau"]["min"] == 0


def test_parabolic_to_cartesian_map_formula():
    parab = coordinate_chart("Euclidean", "Parabolic", 2)
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    sigma, tau = parab.symbols()
    mapping = coordinate_map(parab, cart)
    assert mapping.mapping_exprs((sigma, tau)) == ((sigma**2 - tau**2) / 2, sigma * tau)


def test_parabolic_principal_branch_roundtrip_point():
    parab = coordinate_chart("Euclidean", "Parabolic", 2)
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    pt = (sp.Integer(3), sp.Integer(2))
    xy = transform_coordinates(parab, cart, pt)
    recovered = transform_coordinates(cart, parab, xy)
    assert sp.simplify(recovered[0] - pt[0]) == 0
    assert sp.simplify(recovered[1] - pt[1]) == 0
