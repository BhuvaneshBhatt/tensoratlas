import warnings

import sympy as sp

from tensoratlas import (
    PossibleZeroQWarning,
    IndexedNormalizationConfig,
    cache_stats,
    clear_all_caches,
    coordinate_chart,
    indexed,
    indices,
    indexed_equal,
    normalize_indexed_expression,
    possibly_zero,
    tensor_product,
    TensorObject,
    to_indexed_tensor_form,
)
from tensoratlas.tensor_algebra import metric_tensor


def _sample_expr():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0, 1),)})
    i, j = indices("i_ j_")
    return indexed(g, i, j)


def test_normalization_idempotent_for_sample_expr():
    expr = _sample_expr()
    n1 = normalize_indexed_expression(expr)
    n2 = normalize_indexed_expression(n1)
    assert indexed_equal(n1, n2)


def test_tensorform_boundary_idempotent_for_sample_expr():
    expr = _sample_expr()
    nf1 = to_indexed_tensor_form(expr)
    nf2 = to_indexed_tensor_form(normalize_indexed_expression(expr))
    assert nf1 == nf2


def test_cache_stats_surface_updates_after_workload():
    clear_all_caches()
    expr = _sample_expr()
    _ = normalize_indexed_expression(expr)
    stats = cache_stats()
    assert "normal_form" in stats
    assert stats["normal_form"]["size"] >= 1
    assert stats["symbolic_decision"]["currsize"] >= 0


def test_strict_and_heuristic_modes_agree_on_basic_delta_expr():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0, 1),)})
    i, j = indices("i_ j_")
    expr = indexed(g, i, j)
    strict = normalize_indexed_expression(expr, IndexedNormalizationConfig(normalization_mode="strict"))
    heuristic = normalize_indexed_expression(expr, IndexedNormalizationConfig(normalization_mode="heuristic"))
    assert indexed_equal(strict, heuristic)


def test_possibly_zero_warns_by_default_on_uncertain_input():
    x = sp.Symbol("x")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        result = possibly_zero(sp.sin(x))
    assert result is True
    assert any(isinstance(w.message, PossibleZeroQWarning) for w in rec)


def test_possibly_zero_warning_message_includes_expression_text():
    x = sp.Symbol("x")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        possibly_zero(sp.sin(x))
    messages = [str(w.message) for w in rec if isinstance(w.message, PossibleZeroQWarning)]
    assert any(msg == "Could not decide whether or not sin(x) is zero." for msg in messages)
