import sympy as sp

from tensoratlas.abstract_tensor import (
    abstract_index_type,
    riemann_tensor_head,
    ricci_tensor_head,
    derivative_tensor_head,
    tableau_reduce,
    differential_bianchi_reduce,
    commute_covariant_derivatives,
    schouten_from_ricci,
    ricci_from_schouten,
    weyl_from_riemann_schouten,
    decompose_curvature_workflow,
    differential_curvature_invariant_signature,
    reduce_differential_curvature_invariants_with_report,
)
from sympy.tensor.tensor import tensor_indices


def _idx(itype, names):
    parts = [n for n in names.replace(',', ' ').split() if n]
    out = []
    for n in parts:
        made = tensor_indices(n, itype)
        out.append(made[0] if isinstance(made, tuple) else made)
    return tuple(out)


def test_tableau_reduce_on_product_changes_rank4_factor():
    L = abstract_index_type("L")
    T = riemann_tensor_head("T", L)
    S = ricci_tensor_head("S", L)
    a, b, c, d, e, f = _idx(L, "a b c d e f")
    expr = T(a, b, c, d) * S(e, f)
    reduced = tableau_reduce(expr, ((0, 1), (2, 3)))
    assert reduced.expr != expr


def test_differential_bianchi_reduce_rewrites_first_order_curvature_derivative():
    L = abstract_index_type("L")
    R = riemann_tensor_head("Rdb", L)
    DR = derivative_tensor_head("DRdb", L, R, derivative_order=1)
    a, b, c, d, e = _idx(L, "a b c d e")
    expr = DR(b, c, a, d, e)
    reduced = differential_bianchi_reduce(expr)
    assert str(reduced.expr) != str(expr)


def test_commute_covariant_derivatives_introduces_curvature_correction():
    L = abstract_index_type("L")
    V = ricci_tensor_head("V", L)
    DDV = derivative_tensor_head("DDV", L, V, derivative_order=2)
    a, b, c, d = _idx(L, "a b c d")
    expr = DDV(b, a, c, d)
    reduced = commute_covariant_derivatives(expr)
    assert "R(" in str(reduced.expr) or "Rdb" in str(reduced.expr) or str(reduced.expr) != str(expr)


def test_schouten_ricci_roundtrip_formulas_are_constructed():
    L = abstract_index_type("L")
    Ric = ricci_tensor_head("Ric", L)
    a, b = _idx(L, "a b")
    schouten_expr = schouten_from_ricci(Ric(a, b), dimension=4)
    ricci_expr = ricci_from_schouten(schouten_expr.expr, dimension=4)
    assert "P(" in str(schouten_expr.expr)
    assert "Ric(" in str(ricci_expr.expr)


def test_weyl_from_riemann_schouten_contains_weyl_and_schouten():
    L = abstract_index_type("L")
    R = riemann_tensor_head("Rfull", L)
    a, b, c, d = _idx(L, "a b c d")
    expr = weyl_from_riemann_schouten(R(a, b, c, d))
    text = str(expr.expr)
    assert "C(" in text and "P(" in text


def test_decompose_curvature_workflow_schouten_target():
    L = abstract_index_type("L")
    R = riemann_tensor_head("Rwf", L)
    a, b, c, d = _idx(L, "a b c d")
    expr = decompose_curvature_workflow(R(a, b, c, d), dimension=4, target="schouten")
    assert "P(" in str(expr.expr)


def test_differential_curvature_invariant_reduction_report():
    L = abstract_index_type("L")
    R = riemann_tensor_head("Rinv", L)
    DR = derivative_tensor_head("DRinv", L, R, derivative_order=1)
    a, b, c, d, e = _idx(L, "a b c d e")
    expr = DR(a, b, c, d, e) * DR(-a, -b, -c, -d, -e)
    sig = differential_curvature_invariant_signature(expr, dimension=4)
    reduced = reduce_differential_curvature_invariants_with_report(expr, dimension=4)
    assert sig
    assert reduced.report is not None
    assert reduced.report.dimension_used == 4
