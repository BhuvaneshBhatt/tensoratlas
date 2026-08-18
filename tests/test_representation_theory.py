from sympy.tensor.tensor import tensor_indices

from tensoratlas import (
    abstract_index_type,
    fully_symmetric_head,
    tensor_head,
    index_type,
    tableau_projector,
    compose_tableau_projectors,
    apply_tableau_projector,
    symmetry_adapted_basis,
    decompose_irreducible,
    decompose_tableau_product,
    multiterm_projector_reduce,
    representation_reduce,
)


def _idx(itype, names):
    parts = names.split()
    out = []
    for n in parts:
        made = tensor_indices(n, itype)
        out.append(made[0] if isinstance(made, tuple) else made)
    return tuple(out)


def test_reusable_young_projector_cache_and_composition():
    p1 = tableau_projector(((0, 1), (2, 3)), symmetry_kind="shape22")
    p2 = tableau_projector(((0, 1), (2, 3)), symmetry_kind="shape22")
    assert p1 is p2
    combo = compose_tableau_projectors(p1, p2)
    assert combo == (p1,)


def test_symmetry_adapted_basis_generation_returns_unique_basis():
    V = abstract_index_type("Vsb")
    A = tensor_head("A", [V, V, V, V])
    a, b, c, d = _idx(V, "a b c d")
    expr = A(a, b, c, d)
    basis = symmetry_adapted_basis(expr, [((0, 1), (2, 3)), ((0, 1), (2, 3))])
    assert len(basis.projectors) == 1
    assert len(basis.basis) >= 1


def test_tableau_product_decomposition_handles_products():
    V = abstract_index_type("Vtd")
    A = fully_symmetric_head("A", [V, V, V, V])
    B = fully_symmetric_head("B", [V, V, V, V])
    a, b, c, d, e, f, g, h = _idx(V, "a b c d e f g h")
    expr = A(a, b, c, d) * B(e, f, g, h)
    comps = decompose_tableau_product(expr, [((0, 1), (2, 3))])
    assert len(comps) >= 1
    assert all(hasattr(c, 'expr') for c in comps)


def test_multiterm_projector_reduce_and_representation_reduce_use_canonicalization():
    V = abstract_index_type("Vmr")
    A = tensor_head("A", [V, V, V, V])
    a, b, c, d = _idx(V, "a b c d")
    expr1 = A(a, b, c, d) + A(c, d, a, b)
    expr2 = A(c, d, a, b) + A(a, b, c, d)
    red1 = multiterm_projector_reduce(expr1, [((0, 1), (2, 3))]).expr
    red2 = multiterm_projector_reduce(expr2, [((0, 1), (2, 3))]).expr
    assert str(red1) == str(red2)
    reduced, report = representation_reduce(expr1, [((0, 1), (2, 3))], with_report=True)
    assert hasattr(reduced, 'expr')
    assert len(report.projectors) == 1


def test_irreducible_decomposition_uses_projectors_and_components():
    V = abstract_index_type("Vid")
    A = tensor_head("A", [V, V, V, V])
    a, b, c, d = _idx(V, "a b c d")
    expr = A(a, b, c, d)
    report = decompose_irreducible(expr, [tableau_projector(((0, 1), (2, 3)) )])
    assert len(report.projectors) == 1
    assert len(report.components) == 1
    projected = apply_tableau_projector(expr, report.projectors[0])
    assert hasattr(projected, 'expr')
