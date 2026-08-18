from tensoratlas.core import (
    Manifold,
    TensorHead,
    antisymmetrized_identity,
    cyclic_identity,
    dimension_dependent_antisymmetry_identity,
    identity_orbit,
    reduce_multiterm,
    reduce_multiterm_with_trace,
    schouten_identity,
)


def test_cyclic_identity_orients_curvature_representative():
    manifold = Manifold("M", 4)
    index_type = manifold.index_type("TM")
    a, b, c, d = index_type.indices("a b c d")
    R = TensorHead.curvature("R", index_type)
    identity = cyclic_identity(R, (a, -b, -c, -d), (1, 2, 3))
    reduced = reduce_multiterm(R(a, -d, -b, -c), (identity,))
    assert repr(reduced) == "-1*R(^a,_b,_c,_d) + R(^a,_c,_b,_d)"


def test_total_antisymmetry_identity_reduces_pivot_with_trace():
    manifold = Manifold("M", 3)
    index_type = manifold.index_type("TM")
    a, b, c = index_type.indices("a b c")
    T = TensorHead("T", (index_type, index_type, index_type))
    identity = antisymmetrized_identity(T, (a, b, c), (0, 1, 2))
    result = reduce_multiterm_with_trace(T(c, b, a), (identity,))
    assert result.steps
    assert "T(^a,^b,^c)" in repr(result.expression)
    assert "T(^c,^b,^a)" not in repr(result.expression)


def test_dimension_dependent_identity_requires_too_many_slots():
    manifold = Manifold("M", 2)
    index_type = manifold.index_type("TM")
    a, b, c = index_type.indices("a b c")
    T = TensorHead("T", (index_type, index_type, index_type))
    identity = dimension_dependent_antisymmetry_identity(T, (a, b, c), (0, 1, 2))
    reduced = reduce_multiterm(T(c, b, a), (identity,))
    assert "T(^c,^b,^a)" not in repr(reduced)


def test_schouten_identity_aliases_dimension_dependent_reduction():
    manifold = Manifold("M", 2)
    index_type = manifold.index_type("TM")
    a, b, c = index_type.indices("a b c")
    T = TensorHead("T", (index_type, index_type, index_type))
    identity = schouten_identity(T, (a, b, c), (0, 1, 2))
    reduced = reduce_multiterm(T(c, b, a), (identity,))
    assert "T(^c,^b,^a)" not in repr(reduced)


def test_identity_orbit_generates_renamed_relations():
    manifold = Manifold("M", 4)
    index_type = manifold.index_type("TM")
    a, b, c = index_type.indices("a b c")
    x, y, z = index_type.indices("x y z")
    T = TensorHead("T", (index_type, index_type, index_type))
    identity = antisymmetrized_identity(T, (a, b, c), (0, 1, 2))
    renamed, = identity_orbit(identity, ({a: x, b: y, c: z},))
    reduced = reduce_multiterm(T(z, y, x), (renamed,))
    assert "T(^z,^y,^x)" not in repr(reduced)
