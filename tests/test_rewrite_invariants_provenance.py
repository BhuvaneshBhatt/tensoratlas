from __future__ import annotations

import sympy as sp

from tensoratlas import (
    IndexType,
    Index,
    riemann_tensor_head,
    apply_curvature_identity_policy,
    invariant_basis_catalog,
    invariant_basis_database,
    invariant_relations,
    reduce_higher_order_curvature_invariants,
    reduce_to_invariant_basis,
    coordinate_chart,
    TensorObject,
    indexed,
    indices,
    indexed_canonical_report,
    indexed_equivalence_report,
)
from tensoratlas.tensor_algebra import kronecker_delta_tensor


def _u(i):
    return i.to_sympy()


def _d(i):
    return -i.to_sympy()


def test_rewrite_identity_policy_report_has_provenance_and_fixed_point_fields():
    _, report = apply_curvature_identity_policy(0, 'differential', dimension=3, with_report=True)
    assert report.fixed_point_passes >= 1
    assert 'original' in report.provenance and 'final' in report.provenance
    assert all(step.before_fingerprint and step.after_fingerprint for step in report.steps)


def test_rewrite_invariant_catalog_has_order_derivative_buckets_and_relations():
    V = IndexType('V', dimension=4)
    a, b, c, d = [Index(ch, V, 'u') for ch in 'abcd']
    R = riemann_tensor_head('R', V.to_sympy())
    expr = R(_u(a), _u(b), _d(c), _d(d)) * R(_d(a), _d(b), _u(c), _u(d))
    cat = invariant_basis_catalog(expr, dimension=4)
    db = invariant_basis_database(expr, dimension=4)
    rels = invariant_relations(expr, dimension=3)
    assert cat.by_order_and_derivative
    assert db.by_order_and_derivative
    assert any(tag == 'available_relation' for tag, _ in rels)


def test_rewrite_basis_reduction_reports_coefficients_and_trace():
    V = IndexType('V', dimension=4)
    a, b, c, d = [Index(ch, V, 'u') for ch in 'abcd']
    R = riemann_tensor_head('R', V.to_sympy())
    term = R(_u(a), _u(b), _d(c), _d(d)) * R(_d(a), _d(b), _u(c), _u(d))
    reduced, report = reduce_higher_order_curvature_invariants(term + term, dimension=4, with_report=True)
    reduced2, report2 = reduce_to_invariant_basis(term + term, dimension=4, with_report=True)
    assert reduced is not None and reduced2 is not None
    assert report.coefficient_map or report2.coefficient_map
    assert report.reduction_trace
    assert 'final' in report.provenance


def test_rewrite_indexed_report_has_signature_idempotence_and_equivalence_provenance():
    chart = coordinate_chart('Euclidean', 'Cartesian', 3)
    delta = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name='δ')
    i, j = indices('i^ j_')
    expr = indexed(delta, i, j)
    report = indexed_canonical_report(expr)
    eq_report = indexed_equivalence_report(expr, expr)
    assert report.structural_signature
    assert report.idempotent is True
    assert eq_report['equal'] is True
    assert 'left_provenance' in eq_report and 'right_provenance' in eq_report
