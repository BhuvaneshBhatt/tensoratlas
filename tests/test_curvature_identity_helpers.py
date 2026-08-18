from tensoratlas.core import (
    Manifold,
    TensorHead,
    apply_first_bianchi,
    curvature_heads,
    first_bianchi_identity,
    is_first_bianchi_sum,
)


def test_standard_curvature_heads_have_expected_variance_and_symmetry():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a, b, c = vector.indices("a b c")
    heads = curvature_heads(vector)

    assert repr(heads["riemann_mixed"](a, -b, -c, -c)) == "0"
    assert repr(heads["ricci"](-b, -a)) == "Ric(_a,_b)"
    assert repr(heads["einstein"](-b, -a)) == "G(_a,_b)"
    assert repr(heads["scalar"]()) == "Scal()"


def test_first_bianchi_identity_is_recognized_and_reduced():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a, b, c, d = vector.indices("a b c d")
    curvature = TensorHead.curvature("R", vector)

    expr = first_bianchi_identity(curvature, a, -b, -c, -d)

    assert is_first_bianchi_sum(expr, curvature)
    assert apply_first_bianchi(expr, curvature).is_zero


def test_non_bianchi_curvature_expression_is_preserved():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a, b, c, d = vector.indices("a b c d")
    curvature = TensorHead.curvature("R", vector)

    expr = curvature(a, -b, -c, -d) + curvature(a, -c, -b, -d)

    assert not is_first_bianchi_sum(expr, curvature)
    assert repr(apply_first_bianchi(expr, curvature)) == repr(expr)
