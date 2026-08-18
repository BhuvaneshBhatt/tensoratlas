import sympy as sp

from tensoratlas import (
    coordinate_chart, transform_coordinates, coordinate_map, ScalarField, VectorField, TensorField,
    identity_tensor, tensor_product, tensor_transpose, symmetrize_slots, pull_back, push_forward
)


def test_cartesian_to_polar_point():
    cart = coordinate_chart("Euclidean", "Cartesian", dimension=2)
    polar = coordinate_chart("Euclidean", "Polar", dimension=2)
    pt = transform_coordinates(cart, polar, (1, 1))
    assert sp.simplify(pt[0] - sp.sqrt(2)) == 0


def test_scalar_transform():
    x, y = sp.symbols("x y", real=True)
    cart = coordinate_chart("Euclidean", "Cartesian", dimension=2)
    polar = coordinate_chart("Euclidean", "Polar", dimension=2)
    mapping = coordinate_map(cart, polar)
    f = ScalarField(cart, x**2 + y**2)
    out = f.transform(mapping)
    r, theta = polar.symbols()
    assert sp.simplify(out.expr - r**2) == 0


def test_paraboloidal_roundtrip_point():
    cart = coordinate_chart("Euclidean", "Cartesian", dimension=3)
    parab = coordinate_chart("Euclidean", "Paraboloidal", dimension=3)
    u, v, phi = parab.symbols()
    point = (sp.Integer(2), sp.Integer(1), sp.pi/4)
    xyz = transform_coordinates(parab, cart, point)
    uvphi = transform_coordinates(cart, parab, tuple(xyz))
    assert sp.simplify(uvphi[0] - 2) == 0
    assert sp.simplify(uvphi[1] - 1) == 0


def test_prolate_roundtrip_symbolic_point():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    prolate = coordinate_chart("Euclidean", "ProlateSpheroidal", 3)
    mu, nu, phi = prolate.symbols()
    xyz = transform_coordinates(prolate, cart, (mu, nu, phi))
    back = transform_coordinates(cart, prolate, tuple(xyz))
    assert len(back) == 3


def test_metric_lower_raise_vector():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    v = VectorField(polar, sp.Matrix([[1], [2]]), "contravariant")
    lowered = v.lower_index()
    assert sp.simplify(lowered.components[0] - 1) == 0
    assert sp.simplify(lowered.components[1] - 2 * r**2) == 0
    raised = lowered.raise_index()
    assert sp.simplify(raised.components[1] - 2) == 0


def test_scalar_gradient_and_laplacian_polar():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    f = ScalarField(polar, r**2)
    grad = f.gradient()
    assert sp.simplify(grad.components[0] - 2*r) == 0
    assert sp.simplify(grad.components[1]) == 0
    assert sp.simplify(f.laplacian() - 4) == 0



def test_polar_christoffel_symbols():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    gamma = polar.christoffel_symbols((r, theta))
    assert sp.simplify(gamma[0, 1, 1] + r) == 0
    assert sp.simplify(gamma[1, 0, 1] - 1/r) == 0
    assert sp.simplify(gamma[1, 1, 0] - 1/r) == 0


def test_scalar_covariant_derivative_and_hessian():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    f = ScalarField(polar, r**2)
    cov = f.covariant_derivative()
    assert sp.simplify(cov.components[0] - 2*r) == 0
    hess = f.hessian()
    assert sp.simplify(hess.components[0, 0] - 2) == 0
    assert sp.simplify(hess.components[1, 1] - 2*r**2) == 0


def test_vector_covariant_derivative_polar_radial_unit():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    v = VectorField(polar, sp.Matrix([[1], [0]]), "contravariant")
    nabla = v.covariant_derivative()
    assert sp.simplify(nabla.components[0, 0]) == 0
    assert sp.simplify(nabla.components[1, 1] - 1/r) == 0


def test_curl_in_cylindrical_coordinates():
    cyl = coordinate_chart("Euclidean", "Cylindrical", 3)
    r, theta, z = cyl.symbols()
    v = VectorField(cyl, sp.Matrix([[0], [1], [0]]), "contravariant")
    curl = v.curl()
    assert sp.simplify(curl.components[2] - 2) == 0


def test_vector_connection_laplacian_cartesian_constant_zero():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    v = VectorField(cart, sp.Matrix([[1], [2], [3]]), "contravariant")
    lap = v.connection_laplacian()
    assert all(sp.simplify(lap.components[i]) == 0 for i in range(3))


def test_flat_space_curvature_zero_cartesian_and_polar():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    assert sp.simplify(cart.scalar_curvature()) == 0
    assert sp.simplify(polar.scalar_curvature()) == 0
    riem = polar.riemann_tensor()
    assert all(sp.simplify(riem[idx]) == 0 for idx in [(0,0,0,0),(0,1,0,1),(1,0,1,0),(1,1,1,1)])


def test_polar_geodesic_equation_radial_line():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    lam = sp.Symbol('lam', real=True)
    rfun = sp.Function('r')(lam)
    thetafun = sp.Function('theta')(lam)
    eqs = polar.geodesic_equations((rfun, thetafun), lam)
    assert sp.simplify(eqs[0].lhs - (sp.diff(rfun, lam, 2) - rfun*sp.diff(thetafun, lam)**2)) == 0
    assert sp.simplify(eqs[1].lhs - (sp.diff(thetafun, lam, 2) + 2*sp.diff(rfun, lam)*sp.diff(thetafun, lam)/rfun)) == 0


def test_exterior_derivative_and_hodge_star_cartesian():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    x, y, z = cart.symbols()
    f = ScalarField(cart, x*y)
    df = f.exterior_derivative()
    assert df.variance_spec == 'l'
    assert sp.simplify(df.components[0] - y) == 0
    assert sp.simplify(df.components[1] - x) == 0
    assert sp.simplify(df.components[2]) == 0

    omega = df.exterior_derivative()
    assert omega.variance_spec == 'll'
    assert all(sp.simplify(omega.components[idx]) == 0 for idx in [(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)])

    dx = VectorField(cart, sp.Matrix([[1],[0],[0]]), 'covariant').as_tensor()
    star_dx = dx.hodge_star()
    assert star_dx.variance_spec == 'll'
    assert sp.simplify(star_dx.components[1,2] - 1) == 0
    assert sp.simplify(star_dx.components[2,1] + 1) == 0


def test_codifferential_of_exact_one_form_zero_cartesian():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    x, y, z = cart.symbols()
    f = ScalarField(cart, x**2 + y**2 + z**2)
    df = f.exterior_derivative()
    delta_df = df.codifferential()
    assert isinstance(delta_df, ScalarField)
    assert sp.simplify(delta_df.expr + 6) == 0


def test_lie_derivative_of_coordinate_vectors_zero():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    X = VectorField(cart, sp.Matrix([[1],[0]]), 'contravariant')
    Y = VectorField(cart, sp.Matrix([[0],[1]]), 'contravariant')
    bracket = X.lie_derivative(Y)
    assert all(sp.simplify(bracket.components[i]) == 0 for i in range(2))


def test_curvature_decomposition_and_weyl_zero_in_low_dimension():
    sph = coordinate_chart("Euclidean", "Spherical", 3)
    dec = sph.curvature_decomposition()
    assert dec["weyl"].shape == (3, 3, 3, 3)
    assert all(sp.simplify(dec["weyl"][idx]) == 0 for idx in [(0,0,0,0),(0,1,0,1),(1,2,1,2)])
    assert all(sp.simplify(dec["einstein"][idx]) == 0 for idx in [(0,0),(1,1),(2,2)])


def test_coordinate_killing_vectors_and_first_integrals_spherical():
    sph = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = sph.symbols()
    kills = sph.coordinate_killing_vectors()
    assert (0, 0, 1) in kills
    assert sph.is_killing_vector((0, 0, 1))
    lam = sp.Symbol('lam', real=True)
    rf = sp.Function('r')(lam)
    thf = sp.Function('theta')(lam)
    phf = sp.Function('phi')(lam)
    integrals = sph.geodesic_first_integrals((rf, thf, phf), lam)
    assert sp.simplify(integrals[phi] - rf**2 * sp.sin(thf)**2 * sp.diff(phf, lam)) == 0


def test_orthonormal_vector_conversion_polar():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    v = VectorField(polar, sp.Matrix([[1], [2/r]]), 'contravariant')
    orth = v.to_orthonormal_components()
    assert sp.simplify(orth[0] - 1) == 0
    assert sp.simplify(orth[1] - 2) == 0
    back = VectorField.from_orthonormal_components(polar, orth, 'contravariant')
    assert sp.simplify(back.components[1] - 2/r) == 0


def test_wedge_and_interior_cartesian():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    x, y, z = cart.symbols()
    dx = VectorField(cart, sp.Matrix([[1], [0], [0]]), 'covariant').as_tensor()
    dy = VectorField(cart, sp.Matrix([[0], [1], [0]]), 'covariant').as_tensor()
    two = dx.wedge(dy)
    assert two.variance_spec == 'll'
    assert sp.simplify(two.components[0,1] - 1) == 0
    assert sp.simplify(two.components[1,0] + 1) == 0
    X = VectorField(cart, sp.Matrix([[0], [1], [0]]), 'contravariant')
    interior = two.interior_product(X)
    assert sp.simplify(interior.components[0] + 1) == 0
    assert sp.simplify(interior.components[1]) == 0


def test_tensor_divergence_vector_matches_vector_divergence():
    cyl = coordinate_chart("Euclidean", "Cylindrical", 3)
    r, theta, z = cyl.symbols()
    v = VectorField(cyl, sp.Matrix([[r], [0], [z]]), 'contravariant')
    div1 = v.divergence()
    div2 = v.as_tensor().divergence().expr
    assert sp.simplify(div1 - div2) == 0
    assert sp.simplify(div1 - 3) == 0


def test_ricci_and_lichnerowicz_laplacians_flat_cartesian_zero():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    v = VectorField(cart, sp.Matrix([[1], [2], [3]]), 'contravariant')
    assert all(sp.simplify(v.ricci_laplacian().components[i]) == 0 for i in range(3))
    g = cart.metric(cart.symbols())
    arr = sp.MutableDenseNDimArray.zeros(3,3)
    for i in range(3):
        for j in range(3):
            arr[i,j] = g[i,j]
    T = TensorField(cart, arr, 'll')
    L = T.lichnerowicz_laplacian()
    assert all(sp.simplify(L.components[i,j]) == 0 for i in range(3) for j in range(3))


def test_lie_bracket_and_flow_equations():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    X = VectorField(cart, sp.Matrix([[x], [0]]), 'contravariant')
    Y = VectorField(cart, sp.Matrix([[0], [y]]), 'contravariant')
    bracket = X.lie_bracket(Y)
    assert all(sp.simplify(bracket.components[i]) == 0 for i in range(2))
    t = sp.Symbol('t', real=True)
    fx = sp.Function('x')(t)
    fy = sp.Function('y')(t)
    eqs = X.flow_equations((fx, fy), t)
    assert sp.simplify(eqs[0].lhs - sp.diff(fx, t)) == 0
    assert sp.simplify(eqs[0].rhs - fx) == 0
    assert sp.simplify(eqs[1].rhs) == 0


def test_killing_equations_and_affine_solver_cartesian2d():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    eqs = cart.killing_equations()
    assert len(eqs) == 3
    fams = cart.solve_killing_vectors_affine()
    # Expect one affine family containing free translation/rotation constants.
    assert len(fams) >= 1
    x, y = cart.symbols()
    fam = fams[0]
    # Substitute a pure rotation choice and verify the Killing equations vanish.
    B01, B10 = sp.symbols('B01 B10', real=True)
    trial = tuple(comp.subs({sp.Symbol('a0'):0, sp.Symbol('a1'):0, sp.Symbol('B00'):0, sp.Symbol('B11'):0, sp.Symbol('B01'):-1, sp.Symbol('B10'):1}) for comp in fam)
    for eq in cart.killing_equations(trial):
        expr = eq.lhs if hasattr(eq, 'lhs') else eq
        assert expr == True or sp.simplify(expr) == 0


def test_geodesic_rhs_cartesian_constant_velocity():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    rhs = cart.geodesic_rhs((1, 2, 3, 4))
    assert rhs == (3, 4, 0, 0)


def test_parallel_transport_cartesian_constant_vector():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    samples = [(0, (0, 0)), (1, (1, 0)), (2, (2, 0))]
    out = cart.integrate_parallel_transport(samples, (5, 7))
    assert all(sp.simplify(out[-1][1][i] - v) == 0 for i, v in enumerate((5, 7)))


def test_tensor_algebra_helpers_basic():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    I = identity_tensor(cart)
    assert I.variance_spec == 'ul'
    assert sp.simplify(I.components[0,0] - 1) == 0
    dx = VectorField(cart, sp.Matrix([[1],[0]]), 'covariant').as_tensor()
    dy = VectorField(cart, sp.Matrix([[0],[1]]), 'covariant').as_tensor()
    prod = tensor_product(dx, dy)
    assert prod.variance_spec == 'll'
    trans = tensor_transpose(prod, (1,0))
    assert sp.simplify(trans.components[1,0] - 1) == 0
    sym = symmetrize_slots(dx.wedge(dy), antisymmetric=True)
    assert sp.simplify(sym.components[0,1] - 1) == 0


def test_pushforward_pullback_basic_polar():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    mapping = coordinate_map(cart, polar)
    x, y = cart.symbols()
    f = ScalarField(cart, x)
    pf = pull_back(mapping, f)
    r, theta = polar.symbols()
    assert sp.simplify(pf.expr - r*sp.cos(theta)) == 0
    cov = VectorField(cart, sp.Matrix([[1],[0]]), 'covariant')
    pulled = pull_back(mapping, cov)
    assert pulled.chart == polar
    vec = VectorField(cart, sp.Matrix([[x],[y]]), 'contravariant')
    pushed = push_forward(mapping, vec)
    assert pushed.chart == polar
