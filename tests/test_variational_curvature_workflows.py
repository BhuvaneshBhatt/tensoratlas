from tensoratlas.core import (
    Manifold,
    TensorHead,
    christoffel_variation,
    covariant_derivative,
    einstein_hilbert_metric_variation_density,
    expand_metric_perturbation_to_order,
    ricci_variation,
    scalar_curvature_variation,
)


def _setup():
    manifold = Manifold("M", 4)
    tangent = manifold.index_type("TM")
    a, b, c = tangent.indices("a b c")
    return tangent, a, b, c


def test_christoffel_variation_has_three_metric_derivative_terms():
    tangent, a, b, c = _setup()
    ginv = TensorHead.inverse_metric("ginv", tangent)
    dg = TensorHead.metric("dg", tangent)
    D = covariant_derivative("D", tangent)

    expr = christoffel_variation(dg, ginv, D, a, -b, -c)

    text = repr(expr)
    assert text.count("Ddg(") == 3
    assert "ginv(^a" in text


def test_ricci_variation_uses_connection_divergence_difference():
    tangent, _, b, c = _setup()
    Gamma = TensorHead("dGamma", (tangent, tangent, tangent), variance=("up", "down", "down"))
    D = covariant_derivative("D", tangent)

    expr = ricci_variation(Gamma, D, -b, -c)

    text = repr(expr)
    assert "DdGamma(" in text
    assert text.startswith("-1*")


def test_scalar_curvature_variation_contains_ricci_and_delta_ricci_terms():
    tangent, a, b, _ = _setup()
    ginv = TensorHead.inverse_metric("ginv", tangent)
    dg = TensorHead.metric("dg", tangent)
    Ric = TensorHead.ricci("Ric", tangent)
    dRic = TensorHead.ricci("dRic", tangent)

    expr = scalar_curvature_variation(ginv, Ric, dRic, dg, a, b)

    text = repr(expr)
    assert "dRic(" in text
    assert "Ric(" in text
    assert "dg(" in text


def test_einstein_hilbert_bulk_variation_uses_einstein_tensor():
    tangent, a, b, _ = _setup()
    sqrtg = TensorHead.scalar("sqrtg")
    ginv = TensorHead.inverse_metric("ginv", tangent)
    dg = TensorHead.metric("dg", tangent)
    G = TensorHead.einstein("G", tangent)

    expr = einstein_hilbert_metric_variation_density(sqrtg, ginv, G, dg, a, b)

    text = repr(expr)
    assert text.startswith("-1*")
    assert "G(" in text
    assert "sqrtg()" in text


def test_metric_perturbation_orders_are_exposed_as_dictionary():
    tangent, a, b, c = _setup()
    d = tangent.index("d")
    g = TensorHead.metric("g", tangent)
    h = TensorHead.metric("h", tangent)
    expr = g(-a, -b) * g(-c, -d)

    expanded = expand_metric_perturbation_to_order(expr, g, h, max_order=2)

    assert set(expanded) == {0, 1, 2}
    assert all(expanded[order].terms for order in expanded)
