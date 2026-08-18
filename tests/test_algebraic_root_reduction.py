import sympy as sp

from tensoratlas import (
    algebraically_equal_basic,
    basic_root_cache_stats,
    basic_root_canonicalize,
    clear_basic_root_cache,
    compress_root_products,
    compress_root_sums,
    is_plain_algebraic_form,
)


def test_negative_result_cache_records_hit_for_non_algebraic_input():
    clear_basic_root_cache()
    assert basic_root_canonicalize(sp.pi) is None
    assert basic_root_canonicalize(sp.pi) is None
    stats = basic_root_cache_stats()
    assert stats["hits"] >= 1


def test_algebraically_equal_basic_recognizes_simple_radicals():
    assert algebraically_equal_basic(sp.sqrt(2) + sp.sqrt(8), 3 * sp.sqrt(2)) is True


def test_plain_algebraic_form_rejects_pi_plus_one():
    assert not is_plain_algebraic_form(sp.pi + 1)


def test_compress_root_sums_applies_vieta_for_complete_family():
    x = sp.Symbol("x")
    roots = [sp.CRootOf(x**3 - 2, k) for k in range(3)]
    expr = sp.Add(*roots, sp.Integer(1), evaluate=False)
    out = compress_root_sums(expr)
    assert sp.simplify(out - 1) == 0


def test_compress_root_products_applies_vieta_for_complete_family():
    x = sp.Symbol("x")
    y = sp.Symbol("y")
    roots = [sp.CRootOf(x**3 - 2, k) for k in range(3)]
    expr = sp.Mul(*roots, y, evaluate=False)
    out = compress_root_products(expr)
    assert sp.simplify(out - 2 * y) == 0
