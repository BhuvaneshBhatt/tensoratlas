from fractions import Fraction

import pytest

from tensoratlas.core import Manifold, TensorHead, TensorKernelError, covariant_derivative, partial_derivative


def setup_geometry():
    M = Manifold("M", 4)
    TM = M.index_type("TM")
    a, b, c = TM.indices("a b c")
    V = TensorHead("V", (TM,), variance=("up",))
    W = TensorHead("W", (TM,), variance=("down",))
    g = TensorHead.metric("g", TM)
    CD = covariant_derivative("D", TM)
    return TM, (a, b, c), V, W, g, CD


def test_covariant_derivative_is_linear_and_uses_leibniz_rule():
    TM, (a, b, c), V, W, g, CD = setup_geometry()
    out = CD.apply(V(c) * W(-c), -a)
    text = repr(out)
    assert "DV(_a,^d1)*W(_d1)" in text
    assert "DW(_a,_d1)*V(^d1)" in text


def test_metric_compatibility_removes_metric_derivative():
    TM, (a, b, c), V, W, g, CD = setup_geometry()
    assert CD.apply(g(-b, -c), -a).is_zero


def test_partial_derivative_does_not_assume_metric_compatibility():
    TM, (a, b, c), V, W, g, CD = setup_geometry()
    PD = partial_derivative("P", TM)
    out = PD.apply(g(-b, -c), -a)
    assert repr(out) == "Pg(_a,_b,_c)"


def test_curvature_commutator_on_vector():
    TM, (a, b, c), V, W, g, CD = setup_geometry()
    vector_factor = V(c).terms[0].factors[0]
    out = CD.commutator_on_factor(vector_factor, -a, -b)
    assert repr(out) == "R_D(^c,_d1,_a,_b)*V(^d1)"


def test_curvature_commutator_on_covector_has_negative_curvature_action():
    TM, (a, b, c), V, W, g, CD = setup_geometry()
    covector_factor = W(-c).terms[0].factors[0]
    out = CD.commutator_on_factor(covector_factor, -a, -b)
    assert repr(out) == "-1*R_D(^d1,_c,_a,_b)*W(_d1)"


def test_torsionful_commutator_adds_transport_term():
    M = Manifold("M", 4)
    TM = M.index_type("TM")
    a, b, c = TM.indices("a b c")
    V = TensorHead("V", (TM,), variance=("up",))
    CD = covariant_derivative("D", TM, torsion=True)
    out = CD.commutator_on_factor(V(c).terms[0].factors[0], -a, -b)
    text = repr(out)
    assert "R_D(^c,_d1,_a,_b)*V(^d1)" in text
    assert "DV(_d1,^c)*T_D(^d1,_a,_b)" in text


def test_derivative_rejects_wrong_index_type_and_variance():
    M = Manifold("M", 4)
    N = Manifold("N", 3)
    TM = M.index_type("TM")
    TN = N.index_type("TN")
    a = TM.index("a", variance="down")
    x = TN.index("x", variance="down")
    V = TensorHead("V", (TM,), variance=("up",))
    v = TM.index("v")
    CD = covariant_derivative("D", TM)
    with pytest.raises(TensorKernelError):
        CD.apply(V(v), x)
    with pytest.raises(TensorKernelError):
        CD.apply(V(v), a.flipped())
