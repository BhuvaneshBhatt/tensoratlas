import sympy as sp

from tensoratlas import (
    basic_root_canonicalize,
    extract_algebraic_atoms,
    is_nested_algebraic_form,
    is_plain_algebraic_form,
    normalize_algebraic_forms,
    rewrite_special_angle_forms,
    reduce_with_minpoly_relations,
    prove_zero_via_minpolys,
    clear_basic_root_cache,
    basic_root_cache_stats,
)


def test_basic_root_canonicalize_reduces_simple_radical_sum():
    expr = sp.Add(sp.sqrt(2), sp.sqrt(8), -3 * sp.sqrt(2), evaluate=False)
    assert basic_root_canonicalize(expr) == 0


def test_rewrite_special_angle_forms_handles_exp_i_pi_over_three():
    expr = sp.exp(sp.I * sp.pi / 3)
    out = rewrite_special_angle_forms(expr)
    assert sp.simplify(out - (sp.Rational(1, 2) + sp.sqrt(3) * sp.I / 2)) == 0


def test_algebraic_form_predicates_accept_basic_algebraics():
    expr = sp.Add(sp.sqrt(2), sp.root(2, 3), evaluate=False)
    assert is_plain_algebraic_form(expr)
    assert is_nested_algebraic_form(expr)


def test_extract_algebraic_atoms_finds_special_atoms():
    expr = sp.Add(sp.sqrt(2), sp.exp(sp.I * sp.pi / 3), evaluate=False)
    atoms = extract_algebraic_atoms(expr)
    assert atoms


def test_normalize_algebraic_forms_preserves_exact_zero():
    expr = sp.Add(sp.exp(sp.I * sp.pi / 3), -sp.Rational(1, 2) - sp.sqrt(3) * sp.I / 2, evaluate=False)
    out = normalize_algebraic_forms(expr)
    assert sp.simplify(out) == 0


def test_plain_algebraic_form_rejects_pi():
    assert not is_plain_algebraic_form(sp.pi)


def test_extract_algebraic_atoms_includes_atomic_constants():
    atoms = extract_algebraic_atoms(sp.I + sp.GoldenRatio)
    assert sp.I in atoms
    assert sp.GoldenRatio in atoms


def test_reduce_with_minpoly_relations_detects_zero():
    expr = sp.Add(sp.sqrt(2), sp.sqrt(8), -3 * sp.sqrt(2), evaluate=False)
    assert reduce_with_minpoly_relations(expr) == 0
    assert prove_zero_via_minpolys(expr) is True


def test_basic_root_cache_records_hits():
    clear_basic_root_cache()
    expr = sp.Add(sp.sqrt(2), sp.sqrt(8), -3 * sp.sqrt(2), evaluate=False)
    basic_root_canonicalize(expr)
    basic_root_canonicalize(expr)
    stats = basic_root_cache_stats()
    assert stats["hits"] >= 1
