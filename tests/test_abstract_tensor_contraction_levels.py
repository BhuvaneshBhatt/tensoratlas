import pytest
import sympy as sp

from tensoratlas import (
    AbstractTensorCanonicalizationError,
    Index,
    IndexType,
    Metric,
    TensorHead,
    coordinate_chart,
    cotangent_basis,
    indexed,
    indexed_to_abstract,
    abstract_to_indexed,
    indices,
)
from tensoratlas.abstract_tensor import (
    AbstractContractionPlan,
    build_contraction_plan,
    delta_reduce,
    metric_simplify,
    simplify_abstract,
)
from tensoratlas.tensor_core import TensorObject


def test_build_contraction_plan_detects_metric_step():
    lor = IndexType("Lor", dummy_name="L")
    i = Index("i", lor, "u")
    j = Index("j", lor, "u")
    g = Metric(lor)
    A = TensorHead("A", [lor])
    expr = g(i, -j) * A(j).to_sympy()
    plan = build_contraction_plan(expr)
    assert isinstance(plan, AbstractContractionPlan)
    assert plan.steps
    assert plan.metric_heads


def test_delta_reduce_eliminates_kronecker_delta_factor():
    lor = IndexType("Lor", dummy_name="L")
    i = Index("i", lor, "u")
    j = Index("j", lor, "u")
    delta = lor.to_sympy().delta
    A = TensorHead("A", [lor])
    expr = delta(i.to_sympy(), -j.to_sympy()) * A(j).to_sympy()
    reduced = delta_reduce(expr)
    assert str(reduced.to_sympy()) == "A(i)"


def test_simplify_abstract_metric_mode_matches_metric_reduction():
    lor = IndexType("Lor", dummy_name="L")
    i = Index("i", lor, "u")
    j = Index("j", lor, "u")
    g = Metric(lor)
    A = TensorHead("A", [lor])
    expr = g(i, -j) * A(j).to_sympy()
    assert str(metric_simplify(expr).to_sympy()) == str(simplify_abstract(expr, mode="metric").to_sympy())


def test_validate_contractions_rejects_mixed_index_types_for_same_name():
    a = IndexType("A", dummy_name="A")
    b = IndexType("B", dummy_name="B")
    T = TensorHead("T", [a, b])
    with pytest.raises(AbstractTensorCanonicalizationError):
        build_contraction_plan(T(Index("i", a, "u"), Index("i", b, "l")))


def test_abstract_bridge_handles_scalar_scaled_leaf_roundtrip():
    cart = coordinate_chart('Euclidean', 'Cartesian', 2)
    arr = sp.MutableDenseNDimArray.zeros(2, 2)
    arr[0, 0] = 1
    arr[1, 1] = 2
    T = TensorObject(cart, arr, 'll', (cotangent_basis(cart), cotangent_basis(cart)), name='T')
    i, j = indices('i_ j_')
    leaf = indexed(T, i, j)
    abstract = indexed_to_abstract(leaf)
    scaled = 3 * abstract
    roundtrip = abstract_to_indexed(scaled, tensor_registry={'T': T})
    assert 'T' in str(roundtrip)
