from tensoratlas.core import (
    IdentityBasis,
    Manifold,
    TensorHead,
    generate_renaming_orbit,
    identity_from_expression,
    reduce_multiterm,
    term_coefficient_map,
)


def test_identity_basis_reduces_oriented_relation():
    manifold = Manifold("M", 3)
    tangent = manifold.index_type("T")
    a = tangent.index("a")
    S = TensorHead("S", (tangent,), variance=("up",))
    T = TensorHead("T", (tangent,), variance=("up",))
    identity = identity_from_expression(S(a) + T(a), name="swap")
    basis = IdentityBasis((identity,))

    reduced = basis.reduce(S(a))

    assert repr(reduced) == "-1*T(^a)"


def test_term_coefficient_map_uses_canonical_terms():
    manifold = Manifold("M", 3)
    tangent = manifold.index_type("T")
    a = tangent.index("a")
    V = TensorHead("V", (tangent,), variance=("up",))

    cmap = term_coefficient_map(2 * V(a) - V(a))

    assert list(cmap.values()) == [1]


def test_generate_renaming_orbit_produces_reducible_copy():
    manifold = Manifold("M", 3)
    tangent = manifold.index_type("T")
    a, b = tangent.indices("a b")
    c, d = tangent.indices("c d")
    S = TensorHead("S", (tangent, tangent), symmetry="symmetric", variance=("up", "up"))
    T = TensorHead("T", (tangent, tangent), symmetry="symmetric", variance=("up", "up"))
    identity = identity_from_expression(S(a, b) + T(a, b), name="swap")

    orbit = generate_renaming_orbit(identity, [(c, d)], name_prefix="swap")
    reduced = reduce_multiterm(S(c, d), orbit)

    assert repr(reduced) == "-1*T(^c,^d)"
