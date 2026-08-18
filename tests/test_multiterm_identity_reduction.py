from tensoratlas.core import (
    Manifold,
    TensorHead,
    first_bianchi_reduction_system,
    identity_from_expression,
    reduce_first_bianchi,
    reduce_multiterm,
)


def test_linear_identity_reduces_oriented_pivot():
    manifold = Manifold("M", 4)
    index_type = manifold.index_type("TM")
    a, b = index_type.indices("a b", variance="down")
    S = TensorHead("S", (index_type, index_type), symmetry="symmetric")
    T = TensorHead("T", (index_type, index_type))
    identity = identity_from_expression(S(a, b) + T(a, b), name="S_plus_T")
    reduced = reduce_multiterm(S(a, b), (identity,))
    assert repr(reduced) == "-1*T(_a,_b)"


def test_first_bianchi_reducer_annihilates_cyclic_sum():
    manifold = Manifold("M", 4)
    index_type = manifold.index_type("TM")
    a, b, c, d = index_type.indices("a b c d")
    R = TensorHead.curvature("R", index_type)
    expr = R(a, -b, -c, -d) + R(a, -c, -d, -b) + R(a, -d, -b, -c)
    reduced = reduce_first_bianchi(expr, R, a, -b, -c, -d)
    assert reduced.is_zero


def test_first_bianchi_reducer_rewrites_one_orbit_representative():
    manifold = Manifold("M", 4)
    index_type = manifold.index_type("TM")
    a, b, c, d = index_type.indices("a b c d")
    R = TensorHead.curvature("R", index_type)
    system = first_bianchi_reduction_system(R, a, -b, -c, -d)
    reduced = system.reduce(R(a, -d, -b, -c))
    assert repr(reduced) == "-1*R(^a,_b,_c,_d) + R(^a,_c,_b,_d)"
