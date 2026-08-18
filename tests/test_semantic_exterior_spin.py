import sympy as sp

from tensoratlas import (
    CoordinateChart,
    frame_basis,
    clifford_algebra,
    exterior_form_nf,
    hodge_star_nf,
    codifferential_nf,
    interior_product_nf,
    lie_derivative_nf,
    hodge_laplacian_nf,
    gamma_string_simplify,
    gamma_trace,
    antisymmetrized_gamma_product,
    spin_connection,
    dirac_operator,
)


def test_hodge_and_codifferential_basic_euclidean_2d():
    x, y = sp.symbols('x y')
    alpha = exterior_form_nf({(0,): x, (1,): y}, dimension=2, basis_labels=('dx', 'dy'))
    star = hodge_star_nf(alpha).form
    assert star.terms[(0,)] == -y
    assert star.terms[(1,)] == x
    delta = codifferential_nf(alpha, (x, y))
    assert delta.terms[()] == -(sp.diff(x, x) + sp.diff(y, y))


def test_interior_lie_and_laplacian_nf():
    x, y = sp.symbols('x y')
    beta = exterior_form_nf({(0, 1): x * y}, dimension=2)
    interior = interior_product_nf((1, 0), beta)
    assert interior.terms[(1,)] == x * y
    lie = lie_derivative_nf((x, y), beta, (x, y))
    assert lie.cartan_identity_residual.terms == {}
    lap0 = hodge_laplacian_nf(exterior_form_nf({(): x**2 + y**2}, dimension=2), (x, y))
    assert lap0.terms[()] == -4


def test_gamma_simplification_and_trace():
    cl = clifford_algebra(3, (3, 0, 0), basis_labels=('0', '1', '2'))
    g0, g1, g2 = tuple(sp.Symbol(f'gamma{i}', commutative=False) for i in ('0', '1', '2'))
    rep = gamma_string_simplify(g0 * g0 + g1 * g1, cl)
    assert sp.expand(rep.output_expr) == 2
    anti = antisymmetrized_gamma_product((0, 1), cl)
    assert sp.expand(gamma_string_simplify(anti - g0 * g1, cl).output_expr) == 0
    assert gamma_trace(g0 * g1, cl) == 0


def test_spin_connection_and_dirac_operator_flat_frame():
    x, y = sp.symbols('x y')
    chart = CoordinateChart('g', 'R2', 2, ('x', 'y'), metric_func=lambda c: sp.eye(2))
    frame = frame_basis('e', chart, lambda c: sp.eye(2), orthonormal=True)
    conn = spin_connection(frame, metric_signature=(1, 1))
    cl = clifford_algebra(2, (2, 0, 0), basis_labels=('0', '1'))
    psi = sp.Function('psi')(x, y)
    report = dirac_operator(psi, conn, cl, (x, y))
    out = sp.expand(report.dirac_expression)
    assert 'gamma0' in str(out) and 'gamma1' in str(out)
    assert 'Derivative(psi(x, y), x)' in str(out)
    assert 'Derivative(psi(x, y), y)' in str(out)
