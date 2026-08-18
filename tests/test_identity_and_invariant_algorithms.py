import sympy as sp

from tensoratlas import (
    IndexType,
    Index,
    riemann_tensor_head,
    list_curvature_identity_libraries,
    list_curvature_identity_policies,
    apply_curvature_identity_policy,
    invariant_basis_database,
    invariant_relations,
    reduce_higher_order_curvature_invariants,
    differential_invariant_basis_catalog,
    differential_invariant_equivalent,
)


def _u(i):
    return i.to_sympy()


def _d(i):
    return -i.to_sympy()


def test_larger_identity_libraries_and_policies_present():
    libs = {lib.name for lib in list_curvature_identity_libraries(3)}
    assert {'core', 'differential', 'conversion', 'torsion', 'nonmetric', 'full'}.issubset(libs)
    policies = {p.name for p in list_curvature_identity_policies(3)}
    assert {'fast', 'differential', 'metric_affine', 'full'}.issubset(policies)


def test_identity_library_report_has_steps_and_provenance():
    V = IndexType('V', dimension=3)
    a, b, c, d = [Index(ch, V, 'u') for ch in 'abcd']
    R = riemann_tensor_head('R', V.to_sympy())
    expr = R(_u(a), _u(b), _d(c), _d(d))
    _, report = apply_curvature_identity_policy(expr, 'differential', dimension=3, with_report=True)
    assert report.library_name == 'differential'
    assert isinstance(report.steps, tuple)
    assert all(hasattr(step, 'identity_name') for step in report.steps)


def test_invariant_database_groups_by_order_and_derivative_count():
    V = IndexType('V', dimension=4)
    a, b, c, d = [Index(ch, V, 'u') for ch in 'abcd']
    R = riemann_tensor_head('R', V.to_sympy())
    expr = R(_u(a), _u(b), _d(c), _d(d)) * R(_d(a), _d(b), _u(c), _u(d))
    db = invariant_basis_database(expr, dimension=4)
    assert db.by_order_and_derivative


def test_reduce_higher_order_curvature_invariants_returns_report():
    V = IndexType('V', dimension=4)
    a, b, c, d = [Index(ch, V, 'u') for ch in 'abcd']
    R = riemann_tensor_head('R', V.to_sympy())
    term = R(_u(a), _u(b), _d(c), _d(d)) * R(_d(a), _d(b), _u(c), _u(d))
    reduced, report = reduce_higher_order_curvature_invariants(term + term, dimension=4, with_report=True)
    assert reduced is not None
    assert report.basis_elements


def test_dimension_sensitive_relations_and_equivalence():
    V = IndexType('V', dimension=3)
    a, b, c, d = [Index(ch, V, 'u') for ch in 'abcd']
    R = riemann_tensor_head('R', V.to_sympy())
    expr1 = R(_u(a), _u(b), _d(c), _d(d))
    expr2 = R(_u(a), _u(b), _d(c), _d(d))
    assert differential_invariant_equivalent(expr1, expr2, dimension=3)
    assert isinstance(invariant_relations(expr1, dimension=3), tuple)


def test_differential_invariant_basis_catalog_richer_keys():
    V = IndexType('V', dimension=4)
    a, b, c, d = [Index(ch, V, 'u') for ch in 'abcd']
    R = riemann_tensor_head('R', V.to_sympy())
    expr = R(_u(a), _u(b), _d(c), _d(d)) * R(_d(a), _d(b), _u(c), _u(d))
    cat = differential_invariant_basis_catalog(expr, dimension=4)
    assert cat
    assert all(isinstance(k, tuple) and len(k) == 3 for k in cat)
