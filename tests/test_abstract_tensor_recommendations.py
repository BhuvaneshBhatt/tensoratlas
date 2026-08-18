import sympy as sp

from tensoratlas import (
    IndexType,
    Index,
    metric,
    riemann_tensor_head,
    torsion,
    connection,
    covariant_derivative_operator,
    apply_covariant_derivative,
    derivative_commutator,
    decompose_tableau_product,
    representation_reduce,
    classify_differential_invariants,
    differential_invariant_basis_catalog,
    abstract_normal_form,
    compare_normal_forms,
    list_curvature_identities,
    apply_curvature_identity,
    simplify_with_identity_library,
    CurvatureIdentity,
)


def _setup():
    V = IndexType("V", dimension=4)
    a, b, c, d = [Index(ch, V, "u") for ch in "abcd"]
    return V, a, b, c, d


def test_covariant_derivative_operator_builds_derivative_heads():
    V, a, b, c, d = _setup()
    R = riemann_tensor_head("R", V.to_sympy())
    expr = R(a.to_sympy(), b.to_sympy(), c.to_sympy(), d.to_sympy())
    op = covariant_derivative_operator(V)
    out = apply_covariant_derivative(expr, (-a, -b), operator=op).expr
    assert "nabla_R" in str(out)


def test_connection_and_torsion_feed_commutator():
    V, a, b, c, d = _setup()
    R = riemann_tensor_head("R", V.to_sympy())
    expr = R(a.to_sympy(), b.to_sympy(), c.to_sympy(), d.to_sympy())
    conn = connection("LC", V, metric=metric(V), torsion=torsion("T", V))
    out = derivative_commutator(expr, -a, -b, connection=conn).expr
    s = str(out)
    assert "T(" in s
    assert "nabla_nabla_R" in s or "nabla_R" in s


def test_representation_helpers_return_components_and_reduce():
    V, a, b, c, d = _setup()
    R = riemann_tensor_head("R", V.to_sympy())
    expr = R(a.to_sympy(), b.to_sympy(), c.to_sympy(), d.to_sympy())
    comps = decompose_tableau_product(expr, [((0, 1), (2, 3))])
    assert len(comps) == 1
    red = representation_reduce(expr, [((0, 1), (2, 3))]).expr
    assert red == comps[0].expr


def test_differential_invariant_catalog_and_classification():
    V, a, b, c, d = _setup()
    R = riemann_tensor_head("R", V.to_sympy())
    expr = R(a.to_sympy(), b.to_sympy(), c.to_sympy(), d.to_sympy()) * R(
        -a.to_sympy(), -b.to_sympy(), -c.to_sympy(), -d.to_sympy()
    )
    desc = classify_differential_invariants(expr, dimension=4)
    assert desc
    catalog = differential_invariant_basis_catalog(expr, dimension=4)
    assert catalog
    first = next(iter(catalog.values()))
    assert first.polynomial_degree >= 1


def test_abstract_normal_form_compares_equivalent_inputs():
    V, a, b, c, d = _setup()
    R = riemann_tensor_head("R", V.to_sympy())
    expr1 = R(a.to_sympy(), b.to_sympy(), c.to_sympy(), d.to_sympy())
    expr2 = R(a.to_sympy(), b.to_sympy(), c.to_sympy(), d.to_sympy())
    nf = abstract_normal_form(expr1)
    assert nf.tensor_heads == ("R",)
    assert compare_normal_forms(expr1, expr2)


def test_identity_library_surface_and_application():
    names = {item.name for item in list_curvature_identities()}
    assert "first_bianchi" in names
    assert all(isinstance(item, CurvatureIdentity) for item in list_curvature_identities())

    V, a, b, c, d = _setup()
    R = riemann_tensor_head("R", V.to_sympy())
    expr = R(a.to_sympy(), b.to_sympy(), c.to_sympy(), d.to_sympy())
    out = apply_curvature_identity(expr, "riemann_to_weyl_ricci_scalar", dimension=4).expr
    assert "C(" in str(out) or "Ric(" in str(out)
    simplified = simplify_with_identity_library(expr, ["riemann_to_weyl_ricci_scalar"], dimension=4).expr
    assert simplified is not None
