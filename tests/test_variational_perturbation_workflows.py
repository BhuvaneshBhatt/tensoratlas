from fractions import Fraction

from tensoratlas.core import (
    Manifold,
    TensorHead,
    covariant_derivative,
    euler_lagrange_expression,
    integrate_by_parts_once,
    inverse_metric_variation,
    metric_density_variation,
    perturbative_expand,
    PerturbationRule,
    variation_head,
    vary,
)


def _setup():
    manifold = Manifold("M", 4)
    tangent = manifold.index_type("T")
    a, b, c = tangent.indices("a b c")
    return tangent, a, b, c


def test_variation_uses_product_rule_only_for_requested_heads():
    tangent, a, b, _ = _setup()
    phi = TensorHead("phi", ())
    V = TensorHead("V", (tangent,), variance=("up",))
    dphi = variation_head(phi)
    expr = phi() * V(a)

    result = vary(expr, {phi: dphi})

    assert repr(result) == "V(^a)*dphi()"


def test_inverse_metric_variation_formula_is_structural():
    tangent, a, b, _, = _setup()
    g = TensorHead.metric("g", tangent)
    ginv = TensorHead.inverse_metric("ginv", tangent)
    dg = variation_head(g)
    factor = next(iter(ginv(a, b).terms)).factors[0]

    result = inverse_metric_variation(factor, g, dg)

    text = repr(result)
    assert text.startswith("-")
    assert "ginv(^a,^" in text
    assert "ginv(^b,^" in text
    assert "dg(_" in text


def test_metric_density_variation_formula():
    tangent, *_ = _setup()
    sqrtg = TensorHead.scalar("sqrtg")
    ginv = TensorHead.inverse_metric("ginv", tangent)
    dg = TensorHead.metric("dg", tangent)

    result = metric_density_variation(sqrtg, ginv, dg, tangent)

    assert result.terms[0].coefficient == Fraction(1, 2)
    assert "sqrtg()" in repr(result)
    assert "ginv(^d1,^d2)" in repr(result)
    assert "dg(_d1,_d2)" in repr(result)


def test_integration_by_parts_moves_derivative_from_variation():
    tangent, a, *_ = _setup()
    phi = TensorHead("phi", ())
    dphi = variation_head(phi)
    V = TensorHead("V", (tangent,), variance=("up",))
    D = covariant_derivative("D", tangent)
    term = V(a) * D.derivative_factor(next(iter(dphi().terms)).factors[0], -a)

    result = integrate_by_parts_once(term, D, dphi)

    assert repr(result).startswith("-")
    assert "DV(_d1,^d1)" in repr(result)
    assert "dphi()" in repr(result)


def test_euler_lagrange_coefficient_for_algebraic_term():
    tangent, *_ = _setup()
    phi = TensorHead("phi", ())
    dphi = variation_head(phi)
    D = covariant_derivative("D", tangent)
    lagrangian = phi() * phi()

    result = euler_lagrange_expression(lagrangian, phi, D, variation=dphi)

    assert repr(result.varied_expression) == "2*dphi()*phi()"
    assert repr(result.coefficient) == "2*phi()"


def test_fixed_order_perturbative_expansion():
    tangent, a, b, c = _setup()
    d = tangent.index("d")
    g = TensorHead.metric("g", tangent)
    h = TensorHead.metric("h", tangent)
    expr = g(-a, -b) * g(-c, -d)

    expanded = perturbative_expand(expr, [PerturbationRule(g, h)], max_order=2)

    assert expanded.order(0).terms
    assert expanded.order(1).terms
    assert expanded.order(2).terms
