from tensoratlas.core import Manifold, TensorHead, TensorKernelError


def test_manifold_indices_and_symmetric_canonicalization():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a, b = vector.index("a"), vector.index("b")
    symmetric = TensorHead("S", (vector, vector), symmetry="symmetric")

    assert repr(symmetric(b, a)) == "S(^a,^b)"


def test_antisymmetric_pair_cancels():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a, b = vector.index("a"), vector.index("b")
    antisymmetric = TensorHead("A", (vector, vector), symmetry="antisymmetric")

    assert repr(antisymmetric(a, b) + antisymmetric(b, a)) == "0"


def test_index_type_validation():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    other = manifold.index_type("E")
    a = vector.index("a")
    i = other.index("i")
    tensor = TensorHead("T", (vector, vector))

    try:
        tensor(a, i)
    except TensorKernelError as exc:
        assert "expected" in str(exc)
    else:
        raise AssertionError("mismatched index families should fail")


def test_dummy_renaming_gives_alpha_stable_product():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a = vector.index("a")
    b = vector.index("b")
    vector_head = TensorHead("V", (vector,))

    left = vector_head(a) * vector_head(-a)
    right = vector_head(b) * vector_head(-b)
    assert repr(left) == repr(right)


def test_explicit_variance_pattern_validation():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a = vector.index("a")
    covector = TensorHead("omega", (vector,), variance=("down",))

    try:
        covector(a)
    except TensorKernelError as exc:
        assert "expects down" in str(exc)
    else:
        raise AssertionError("explicit slot variance should be enforced")

    assert repr(covector(-a)) == "omega(_a)"


def test_free_index_signature_preserves_index_type_identity():
    manifold = Manifold("M", 4)
    tangent = manifold.index_type("TM")
    internal = manifold.index_type("E")
    a = tangent.index("a")
    internal_a = internal.index("a")
    vector_head = TensorHead("V", (tangent,))
    internal_head = TensorHead("W", (internal,))

    try:
        vector_head(a) + internal_head(internal_a)
    except TensorKernelError as exc:
        assert "free-index" in str(exc)
    else:
        raise AssertionError("same index name in different index families should not add")


def test_riemann_monoterm_pair_symmetries():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a, b, c, d = vector.indices("a b c d")
    riemann = TensorHead.riemann("R", vector)

    assert repr(riemann(b, a, c, d)) == "-1*R(^a,^b,^c,^d)"
    assert repr(riemann(c, d, a, b)) == "R(^a,^b,^c,^d)"
    assert repr(riemann(a, a, c, d)) == "0"
