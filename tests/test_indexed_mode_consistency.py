import pytest

from tensoratlas import IndexedNormalizationConfig, coordinate_chart, indexed, indexed_equal, indices, normalize_indexed_expression, TensorObject
from tensoratlas.tensor_algebra import kronecker_delta_tensor


def _sample_expr():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    delta = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i, j = indices("i^ j_")
    return indexed(delta, i, j)


def test_invalid_normalization_mode_raises():
    expr = _sample_expr()
    with pytest.raises(ValueError):
        normalize_indexed_expression(expr, IndexedNormalizationConfig(normalization_mode="unknown"))


def test_invalid_simplification_level_raises():
    expr = _sample_expr()
    with pytest.raises(ValueError):
        normalize_indexed_expression(expr, IndexedNormalizationConfig(simplification_level="mystery"))


def test_public_entry_points_honor_mode_consistently():
    expr = _sample_expr()
    strict = normalize_indexed_expression(expr, IndexedNormalizationConfig(normalization_mode="strict", simplification_level="cheap"))
    heuristic = normalize_indexed_expression(expr, IndexedNormalizationConfig(normalization_mode="heuristic", simplification_level="cheap"))
    assert indexed_equal(strict, heuristic, config=IndexedNormalizationConfig(simplification_level="cheap"))
