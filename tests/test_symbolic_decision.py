import sympy as sp

from tensoratlas import equal_decision, is_equal, is_zero, zero_decision


def test_zero_decision_exact_trig_identity():
    x = sp.symbols('x')
    assert is_zero(sp.sin(x)**2 + sp.cos(x)**2 - 1)


def test_equal_decision_exact_rational():
    x = sp.symbols('x', nonzero=True)
    assert is_equal((x**2 - 1)/(x - 1), x + 1)


def test_zero_decision_uncertain_numeric_near_zero():
    x = sp.symbols('x')
    d = zero_decision(sp.sin(x), mode='safe')
    assert d.kind.value in {'nonzero', 'uncertain'}
    d2 = equal_decision(sp.sin(x), 0, mode='strict')
    assert d2.kind.value in {'uncertain', 'nonzero'}



def test_zero_decision_basic_root_canonicalize_radicals():
    expr = sp.Add(sp.sqrt(2), sp.sqrt(8), -3 * sp.sqrt(2), evaluate=False)
    d = zero_decision(expr, mode='safe')
    assert d.kind.value == 'zero'
    assert d.method in {'basic-root-canonicalize', 'simplify', 'light-simplify', 'is_zero'}


def test_zero_decision_basic_root_canonicalize_special_angle():
    expr = sp.Add(sp.exp(sp.I * sp.pi / 3), -sp.Rational(1, 2) - sp.sqrt(3) * sp.I / 2, evaluate=False)
    d = zero_decision(expr, mode='safe')
    assert d.kind.value == 'zero'


def test_zero_decision_rejects_transcendental_as_plain_algebraic():
    d = zero_decision(sp.pi, mode="safe")
    assert d.kind.value == "nonzero"


def test_zero_decision_uses_basic_root_canonicalization_for_radicals():
    expr = (sp.sqrt(2) + sp.sqrt(3))**2 - (5 + 2 * sp.sqrt(6))
    d = zero_decision(expr, mode="safe")
    assert d.kind.value == "zero"
