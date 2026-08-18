import sympy as sp

from tensoratlas import (
    IndexType,
    Index,
    riemann_tensor_head,
    weyl_tensor_head,
    list_curvature_identity_libraries,
    apply_curvature_identity_library,
    invariant_basis_catalog,
    reduce_to_invariant_basis,
    unified_tensor_normal_form,
    compare_unified_normal_forms,
    bridge_tensor_expression,
    roundtrip_bridge,
    reduce_component_via_abstract,
    derivative_tensor_head,
    differential_invariant_basis_catalog,
    coordinate_chart,
    tensor_from_components,
    indices,
    indexed,
)


def _up(idx):
    return idx.to_sympy()


def _down(idx):
    return -idx.to_sympy()


def test_identity_libraries_include_dimension_specific_low_dim():
    libs = {lib.name: lib for lib in list_curvature_identity_libraries(3)}
    assert {"core", "differential", "schouten", "full", "dimension_specific"}.issubset(libs)
    assert "dimension_weyl_zero" in libs["dimension_specific"].identities


def test_apply_identity_library_reports_applied_rules():
    V = IndexType("V", dimension=3)
    a, b, c, d = [Index(ch, V, "u") for ch in "abcd"]
    C = weyl_tensor_head("C", V.to_sympy())
    expr = C(_up(a), _up(b), _up(c), _up(d))
    reduced, report = apply_curvature_identity_library(expr, library="full", dimension=3, with_report=True)
    assert reduced is not None
    assert report.library_name == "full"
    assert report.final_expr == 0


def test_invariant_basis_catalog_and_reduction_group_terms():
    V = IndexType("V", dimension=4)
    a, b, c, d = [Index(ch, V, "u") for ch in "abcd"]
    R = riemann_tensor_head("R", V.to_sympy())
    term = R(_up(a), _up(b), _down(c), _down(d)) * R(_down(a), _down(b), _up(c), _up(d))
    expr = 2 * term + term
    catalog = invariant_basis_catalog(expr, dimension=4)
    assert len(catalog.elements) >= 1
    reduced = reduce_to_invariant_basis(expr, catalog, dimension=4)
    assert reduced is not None


def test_unified_normal_form_matches_permuted_abstract_and_indexed_products():
    V = IndexType("V", dimension=4)
    a, b, c, d = [Index(ch, V, "u") for ch in "abcd"]
    R = riemann_tensor_head("R", V.to_sympy())
    expr1 = R(_up(a), _up(b), _down(c), _down(d))
    expr2 = R(_up(a), _up(b), _down(c), _down(d))
    assert compare_unified_normal_forms(expr1, expr2, dimension=4)

    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    A = tensor_from_components(chart, [[1, 0], [0, 1]], 'ul', name='A')
    B = tensor_from_components(chart, [[2, 0], [0, 2]], 'ul', name='B')
    i, j_down, j_up, k_down = indices('i^ j_ j^ k_')
    left = indexed(A, i, j_down) * indexed(B, j_up, k_down)
    right = indexed(B, j_up, k_down) * indexed(A, i, j_down)
    nf_left = unified_tensor_normal_form(left)
    nf_right = unified_tensor_normal_form(right)
    assert nf_left.key == nf_right.key


def test_bridge_tensor_expression_and_reduce_component_via_abstract():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    obj = tensor_from_components(chart, [[1, 0], [0, 1]], 'll', name='G')
    abstract_obj, report = bridge_tensor_expression(obj, target='abstract', with_report=True)
    assert report.source_layer == 'component'
    assert abstract_obj is not None
    reduced = reduce_component_via_abstract(obj, tensor_registry={'G': obj}, dimension=2)
    assert reduced is not None


def test_identity_library_and_invariant_reduction_use_canonical_core_consistently():
    V = IndexType("V", dimension=4)
    a, b, c, d = [Index(ch, V, "u") for ch in "abcd"]
    R = riemann_tensor_head("R", V.to_sympy())
    term1 = R(_up(a), _up(b), _down(c), _down(d)) * R(_down(a), _down(b), _up(c), _up(d))
    term2 = R(_down(a), _down(b), _up(c), _up(d)) * R(_up(a), _up(b), _down(c), _down(d))
    cat1 = invariant_basis_catalog(term1, dimension=4)
    cat2 = invariant_basis_catalog(term2, dimension=4)
    assert tuple(cat1.by_signature) == tuple(cat2.by_signature)
    red1 = reduce_to_invariant_basis(term1, cat1, dimension=4)
    red2 = reduce_to_invariant_basis(term2, cat2, dimension=4)
    assert compare_unified_normal_forms(red1.expr, red2.expr, dimension=4)


def test_bridge_roundtrip_uses_canonical_core_for_abstract_leg():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    obj = tensor_from_components(chart, [[1, 0], [0, 1]], 'll', name='G')
    abstract_obj, report = bridge_tensor_expression(obj, target='abstract', with_report=True, dimension=2)
    assert 'canonical_core=canonical_tensor_expression' in report.notes
    roundtripped = roundtrip_bridge(obj, tensor_registry={'G': obj}, dimension=2)
    assert roundtripped is not None


def test_identity_library_report_tracks_canonicalized_original_and_final():
    V = IndexType("V", dimension=4)
    a, b, c, d = [Index(ch, V, "u") for ch in "abcd"]
    R = riemann_tensor_head("R", V.to_sympy())
    expr = R(_up(a), _up(b), _down(c), _down(d))
    reduced, report = apply_curvature_identity_library(expr, library="core", dimension=4, with_report=True)
    assert compare_unified_normal_forms(report.original_expr, report.original_expr, dimension=4)
    assert compare_unified_normal_forms(report.final_expr, reduced.expr, dimension=4)


def test_differential_invariant_catalog_uses_canonicalized_representatives():
    V = IndexType("V", dimension=4)
    a, b, c, d, e = [Index(ch, V, "u") for ch in "abcde"]
    R = riemann_tensor_head("R", V.to_sympy())
    DR = derivative_tensor_head(R, 1)
    term1 = DR(_up(a), _up(b), _down(c), _down(d), _up(e))
    term2 = DR(_up(a), _up(b), _down(c), _down(d), _up(e))
    cat1 = differential_invariant_basis_catalog(term1, dimension=4)
    cat2 = differential_invariant_basis_catalog(term2, dimension=4)
    assert tuple(cat1) == tuple(cat2)
