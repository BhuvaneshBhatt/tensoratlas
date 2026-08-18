from tensoratlas.core import Manifold, TensorHead, TensorKernelError


def test_delta_contracts_vector_index():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a, b = vector.index("a"), vector.index("b")
    delta = TensorHead.delta("delta", vector)
    vector_head = TensorHead("V", (vector,))

    assert repr(delta(a, -b) * vector_head(b)) == "V(^a)"
    assert repr(delta(a, -b) * vector_head(-a)) == "V(_b)"


def test_metric_lowers_and_inverse_metric_raises_vector_index():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a, b = vector.index("a"), vector.index("b")
    metric = TensorHead.metric("g", vector)
    inverse_metric = TensorHead.inverse_metric("ginv", vector)
    vector_head = TensorHead("V", (vector,))

    assert repr(metric(-a, -b) * vector_head(b)) == "V(_a)"
    assert repr(inverse_metric(a, b) * vector_head(-b)) == "V(^a)"


def test_metric_inverse_contracts_to_delta_then_applies():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a, b, c = vector.index("a"), vector.index("b"), vector.index("c")
    metric = TensorHead.metric("g", vector)
    inverse_metric = TensorHead.inverse_metric("ginv", vector)
    vector_head = TensorHead("V", (vector,))

    assert repr(metric(-a, -b) * inverse_metric(b, c)) == "delta_TM(^c,_a)"
    assert repr(metric(-a, -b) * inverse_metric(b, c) * vector_head(a)) == "V(^c)"


def test_addition_rejects_incompatible_free_indices():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a = vector.index("a")
    vector_head = TensorHead("V", (vector,))

    try:
        vector_head(a) + vector_head(-a)
    except TensorKernelError as exc:
        assert "free-index" in str(exc)
    else:
        raise AssertionError("Adding tensors with different free-index variance should fail")


def test_noncommutative_factor_order_is_preserved():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a = vector.index("a")
    op_a = TensorHead("A", (vector,), commutative=False)
    op_b = TensorHead("B", (vector,), commutative=False)

    assert repr(op_a(a) * op_b(-a)) == "A(^d1)*B(_d1)"
    assert repr(op_b(-a) * op_a(a)) == "B(_d1)*A(^d1)"


def test_delta_trace_uses_integer_dimension():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a = vector.index("a")
    delta = TensorHead.delta("delta", vector)

    assert repr(delta(a, -a)) == "4"


def test_metric_can_lower_index_inside_rank_two_tensor():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a, b, c = vector.indices("a b c")
    metric = TensorHead.metric("g", vector)
    tensor = TensorHead("T", (vector, vector))

    assert repr(metric(-a, -b) * tensor(b, c)) == "T(_a,^c)"


def test_repeated_invalid_indices_are_rejected():
    manifold = Manifold("M", 4)
    vector = manifold.index_type("TM")
    a = vector.index("a")
    tensor = TensorHead("T", (vector, vector))

    try:
        tensor(a, a)
    except TensorKernelError as exc:
        assert "Invalid repeated index" in str(exc)
    else:
        raise AssertionError("repeated same-variance abstract index should fail")
