
import sympy as sp

from tensoratlas import (
    coordinate_chart,
    TensorObject,
    TensorIndex,
    indexed,
    indices,
    indexed_equal,
    indexed_signature,
    normalize_indexed_expression,
    IndexedNormalizationConfig,
    stronger_indexed_equal,
)
from tensoratlas.tensor_algebra import kronecker_delta_tensor


def test_central_pipeline_dummy_index_equivalence():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    delta = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i, j = indices("i^ j_")
    a, b = indices("a^ b_")
    expr1 = indexed(delta, i, j)
    expr2 = indexed(delta, a, b)
    assert indexed_equal(expr1, expr2)
    assert indexed_signature(expr1) == indexed_signature(expr2)


def test_pipeline_tensor_product_normalization():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    delta = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i, j, k = indices("i^ j_ k_")
    expr = indexed(delta, i, j) * indexed(delta, j.dual(), k)
    norm = normalize_indexed_expression(expr, IndexedNormalizationConfig(max_passes=8))
    assert norm is not None


def test_stronger_indexed_equal_uses_central_pipeline():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    delta = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    p, q = indices("p^ q_")
    r, s = indices("r^ s_")
    left = indexed(delta, p, q)
    right = indexed(delta, r, s)
    assert stronger_indexed_equal(left, right)
