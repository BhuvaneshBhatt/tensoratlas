from tensoratlas.core import (
    CliffordAlgebra,
    GaugeAlgebra,
    GradedProduct,
    GradedSymbol,
    Manifold,
    SpinorIndexFamily,
    TensorHead,
    graded_commutator,
)


def test_odd_commuting_symbols_pick_up_koszul_sign():
    psi = GradedSymbol("psi", parity=1, commutative=True)
    chi = GradedSymbol("chi", parity=1, commutative=True)

    product = GradedProduct.one() * psi * chi
    reversed_product = GradedProduct.one() * chi * psi

    assert repr(product) == "-chi*psi"
    assert repr(reversed_product) == "chi*psi"


def test_graded_commutator_returns_super_lie_difference_terms():
    psi = GradedProduct.one() * GradedSymbol("psi", parity=1, commutative=False)
    chi = GradedProduct.one() * GradedSymbol("chi", parity=1, commutative=False)

    left, right = graded_commutator(psi, chi)

    assert repr(left) == "psi*chi"
    assert repr(right) == "chi*psi"


def test_clifford_anticommutator_builds_metric_delta_expression():
    manifold = Manifold("M", 4)
    tangent = manifold.index_type("TM")
    spinors = SpinorIndexFamily.create(manifold, "S", dimension=4)
    a, b = tangent.indices("a b")
    A, B = spinors.indices("A B")
    ginv = TensorHead.inverse_metric("g", tangent)
    clifford = CliffordAlgebra(tangent, ginv)

    expr = clifford.anticommutator(a, b, -A, B)
    text = repr(expr)

    assert text.startswith("2*")
    assert "g(^a,^b)" in text
    assert "delta_S(^B,_A)" in text


def test_gamma_factor_is_noncommutative_and_typed():
    manifold = Manifold("M", 4)
    tangent = manifold.index_type("TM")
    spinors = SpinorIndexFamily.create(manifold, "S")
    a = tangent.index("a")
    A, B = spinors.indices("A B")
    clifford = CliffordAlgebra(tangent, TensorHead.inverse_metric("g", tangent))

    expr = clifford.gamma(a, -A, B)
    factor = expr.terms[0].factors[0]

    assert factor.head.commutative is False
    assert factor.head.index_types == (tangent, spinors.index_type, spinors.index_type)


def test_gauge_algebra_declares_structure_constants_and_metric():
    manifold = Manifold("M", 4)
    algebra = GaugeAlgebra.create(manifold, "suN", dimension=8)
    a, b, c = algebra.adjoint_indices("a b c")

    expr = algebra.bracket(-a, -b, c)
    text = repr(expr)

    assert "f_suN(^c,_a,_b)" in text
    assert algebra.killing_form.role == "metric"
