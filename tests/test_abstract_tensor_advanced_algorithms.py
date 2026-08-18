from tensoratlas.abstract_tensor import (
    Index,
    index_type,
    tensor_head,
    riemann_tensor_head,
    symmetrize_indices,
    antisymmetrize_indices,
    young_project_indices,
    decompose_riemann_curvature,
    decompose_curvature_expression,
)


def test_symmetrize_indices_rank2_tensor():
    L = index_type("L", dimension=4)
    i = Index("i", L, "l")
    j = Index("j", L, "l")
    A = tensor_head("A", [L, L])
    text = str(symmetrize_indices(A(i, j), (0, 1)).expr)
    assert "A(-i, -j)" in text
    assert "A(-j, -i)" in text
    assert "1/2" in text


def test_antisymmetrize_indices_rank2_tensor():
    L = index_type("L", dimension=4)
    i = Index("i", L, "l")
    j = Index("j", L, "l")
    A = tensor_head("A", [L, L])
    text = str(antisymmetrize_indices(A(i, j), (0, 1)).expr)
    assert "A(-i, -j)" in text
    assert "A(-j, -i)" in text
    assert "1/2" in text


def test_young_project_indices_shape_22_on_rank4_tensor():
    L = index_type("L", dimension=4)
    i, j, k, l = [Index(s, L, "l") for s in "ijkl"]
    T = tensor_head("T", [L, L, L, L])
    out = young_project_indices(T(i, j, k, l), ((0, 1), (2, 3))).expr
    text = str(out)
    assert "T(" in text
    assert text != "0"


def test_decompose_riemann_curvature_dimension4_contains_weyl_and_ricci_parts():
    L = index_type("L", dimension=4)
    i, j, k, l = [Index(s, L, "l") for s in "ijkl"]
    R = riemann_tensor_head("R", L.to_sympy())
    out = decompose_riemann_curvature(R(i.to_sympy(), j.to_sympy(), k.to_sympy(), l.to_sympy()), dimension=4).expr
    text = str(out)
    assert "C(" in text
    assert "Ric(" in text
    assert "metric(" in text
    assert "R*" in text or "*R" in text


def test_decompose_riemann_curvature_dimension3_eliminates_weyl_part():
    L = index_type("L", dimension=3)
    i, j, k, l = [Index(s, L, "l") for s in "ijkl"]
    R = riemann_tensor_head("R", L.to_sympy())
    out = decompose_riemann_curvature(R(i.to_sympy(), j.to_sympy(), k.to_sympy(), l.to_sympy()), dimension=3).expr
    text = str(out)
    assert "C(" not in text
    assert "Ric(" in text


def test_decompose_curvature_expression_expands_products_termwise():
    L = index_type("L", dimension=4)
    i, j, k, l, m, n = [Index(s, L, "l") for s in "ijklmn"]
    R = riemann_tensor_head("R", L.to_sympy())
    S = tensor_head("S", [L, L])
    expr = S(m, n).expr * R(i.to_sympy(), j.to_sympy(), k.to_sympy(), l.to_sympy())
    out = decompose_curvature_expression(expr, dimension=4).expr
    text = str(out)
    assert "S(" in text
    assert "Ric(" in text
