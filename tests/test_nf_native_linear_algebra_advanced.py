import sympy as sp

from tensoratlas.normal_forms import TNFMatrix


def test_nfmatrix_rank_deficient_pseudoinverse_penrose_conditions():
    a = TNFMatrix(2, 2, ((1, 2), (2, 4)))
    p = a.pinv()
    A = a.to_sympy()
    P = p.to_sympy()
    assert (A * P * A - A).applyfunc(sp.simplify) == sp.zeros(2)
    assert (P * A * P - P).applyfunc(sp.simplify) == sp.zeros(2)
    assert (A * P - (A * P).T).applyfunc(sp.simplify) == sp.zeros(2)
    assert (P * A - (P * A).T).applyfunc(sp.simplify) == sp.zeros(2)


def test_nfmatrix_jordan_form_defective_case_matches_matrix_identity():
    a = TNFMatrix(2, 2, ((1, 1), (0, 1)))
    P, J = a.jordan_form()
    A = a.to_sympy()
    assert (P * J * P.inv() - A).applyfunc(sp.simplify) == sp.zeros(2)
    assert J == sp.Matrix([[1, 1], [0, 1]])


def test_nfmatrix_generalized_eigenspace_basis_contains_chain_vector():
    a = TNFMatrix(2, 2, ((1, 1), (0, 1)))
    basis = a.generalized_eigenspace_basis(sp.Integer(1), 2)
    assert len(basis) >= 1
    A = a.to_sympy()
    I = sp.eye(2)
    assert any(((A - I) ** 2 * v).applyfunc(sp.simplify) == sp.zeros(2, 1) for v in basis)


def test_nfmatrix_jordan_form_handles_larger_defective_block():
    a = TNFMatrix(3, 3, ((2, 1, 0), (0, 2, 1), (0, 0, 2)))
    P, J = a.jordan_form()
    A = a.to_sympy()
    assert (P * J * P.inv() - A).applyfunc(sp.simplify) == sp.zeros(3)
    assert J == sp.Matrix([[2, 1, 0], [0, 2, 1], [0, 0, 2]])


def test_nfmatrix_jordan_form_handles_mixed_block_sizes_same_eigenvalue():
    a = TNFMatrix(4, 4, ((3, 1, 0, 0), (0, 3, 0, 0), (0, 0, 3, 1), (0, 0, 0, 3)))
    P, J = a.jordan_form()
    A = a.to_sympy()
    assert (P * J * P.inv() - A).applyfunc(sp.simplify) == sp.zeros(4)
    assert J == A
