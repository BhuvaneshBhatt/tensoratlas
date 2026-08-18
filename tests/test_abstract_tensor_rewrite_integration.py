import sympy as sp

from tensoratlas.abstract_tensor import (
    AbstractTensorExpr,
    Index,
    IndexType,
    Metric,
    TensorHead,
    contract_metric,
    lower_index,
    raise_index,
    trace_abstract,
)


def test_plain_abstract_index_wrappers_construct_sympy_objects():
    lor = IndexType("Lor", dummy_name="L")
    i = Index("i", lor, "u")
    j = Index("j", lor, "l")
    g = Metric(lor)
    A = TensorHead("A", [lor])

    expr = g(i, j) * A(-j)
    assert isinstance(expr, AbstractTensorExpr)
    assert "A" in str(expr.to_sympy())
    assert "metric" in str(expr.to_sympy())


def test_raise_index_on_covector_uses_metric_contraction():
    lor = IndexType("Lor", dummy_name="L")
    i_down = Index("i", lor, "l")
    A = TensorHead("A", [lor])

    raised = raise_index(A(i_down), i_down)
    assert str(raised.to_sympy()) == "A(i)"


def test_lower_index_on_vector_uses_metric_contraction():
    lor = IndexType("Lor", dummy_name="L")
    i_up = Index("i", lor, "u")
    A = TensorHead("A", [lor])

    lowered = lower_index(A(i_up), i_up)
    assert str(lowered.to_sympy()) == "A(-i)"


def test_contract_metric_eliminates_simple_metric_factor():
    lor = IndexType("Lor", dummy_name="L")
    i = Index("i", lor, "u")
    j = Index("j", lor, "u")
    g = Metric(lor)
    A = TensorHead("A", [lor])

    expr = g(i, -j) * A(j)
    reduced = contract_metric(expr)
    assert str(reduced.to_sympy()) == "A(i)"


def test_trace_abstract_contracts_one_upper_and_one_lower_index():
    lor = IndexType("Lor", dummy_name="L")
    i = Index("i", lor, "u")
    j = Index("j", lor, "l")
    T = TensorHead("T", [lor, lor])

    traced = trace_abstract(T(i, j), i, j)
    assert traced.to_sympy().get_free_indices() == []
    assert "T(" in str(traced.to_sympy())
