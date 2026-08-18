import sympy as sp
from sympy.tensor.tensor import tensor_indices

from tensoratlas import (
    abstract_index_type,
    canonicalize_abstract_tensor_expr,
    curvature_invariant_basis,
    curvature_invariant_signature,
    reduce_curvature_invariants,
    reduce_curvature_invariants_with_report,
    riemann_tensor_head,
    weyl_tensor_head,
)


def test_curvature_invariant_basis_collapses_permuted_riemann_squared_terms():
    lor = abstract_index_type('L', dummy_name='L', dim=4)
    a, b, c, d = tensor_indices('a,b,c,d', lor)
    R = riemann_tensor_head('R', lor)
    expr = R(a, b, c, d) * R(-a, -b, -c, -d) + 2 * R(c, d, a, b) * R(-c, -d, -a, -b)
    basis = curvature_invariant_basis(expr)
    assert len(basis) == 1
    sig = curvature_invariant_signature(expr)
    assert len(set(sig)) == 1


def test_reduce_curvature_invariants_combines_like_terms_and_reports():
    lor = abstract_index_type('L', dummy_name='L', dim=4)
    a, b, c, d = tensor_indices('a,b,c,d', lor)
    R = riemann_tensor_head('R', lor)
    term = R(a, b, c, d) * R(-a, -b, -c, -d)
    reduced = reduce_curvature_invariants(term + 3 * term)
    assert canonicalize_abstract_tensor_expr(reduced - 4 * term) == 0
    wrapped = reduce_curvature_invariants_with_report(term + 3 * term)
    assert getattr(wrapped.report, 'term_multiplicities')
    assert list(wrapped.report.term_multiplicities.values()) == [sp.Integer(4)]


def test_reduce_curvature_invariants_respects_dimension_dependent_weyl_vanishing():
    lor = abstract_index_type('L', dummy_name='L', dim=3)
    a, b, c, d = tensor_indices('a,b,c,d', lor)
    C = weyl_tensor_head('C', lor)
    expr = C(a, b, c, d) * C(-a, -b, -c, -d)
    reduced = reduce_curvature_invariants(expr, dimension=3)
    assert reduced == 0
