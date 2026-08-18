from tensoratlas.core import (
    Manifold,
    TensorHead,
    identity_closure,
    reduce_multiterm,
    young_antisymmetry_identity,
    young_row_symmetry_identity,
)


def test_young_column_antisymmetry_reduces_reversed_column():
    manifold = Manifold("M", 3)
    tangent = manifold.index_type("TM")
    a, b, c = tangent.indices("a b c")
    T = TensorHead("T", (tangent, tangent, tangent))
    identity = young_antisymmetry_identity(T, (a, b, c), (0, 1, 2))

    reduced = reduce_multiterm(T(c, b, a), (identity,))

    assert "T(^c,^b,^a)" not in repr(reduced)


def test_young_row_symmetry_reduces_adjacent_swap():
    manifold = Manifold("M", 3)
    tangent = manifold.index_type("TM")
    a, b, c = tangent.indices("a b c")
    T = TensorHead("T", (tangent, tangent, tangent))
    identity = young_row_symmetry_identity(T, (a, b, c), (0, 1))

    reduced = reduce_multiterm(T(b, a, c), (identity,))

    assert "T(^b,^a,^c)" not in repr(reduced)


def test_identity_closure_keeps_base_identity():
    manifold = Manifold("M", 3)
    tangent = manifold.index_type("TM")
    a, b, c = tangent.indices("a b c")
    T = TensorHead("T", (tangent, tangent, tangent))
    identity = young_antisymmetry_identity(T, (a, b, c), (0, 1, 2))

    basis = identity_closure((identity,))

    assert basis.identities
