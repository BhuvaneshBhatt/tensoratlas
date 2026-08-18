import sympy as sp

from tensoratlas import ScalarField, VectorField, coordinate_chart, coordinate_map, transform_coordinates


# These tests translate a small, direct subset of the reference tensor-analysis
# tests from the uploaded Tensors.zip archive into the tensoratlas API.
# Source files used:
# - coordinate transformation regression case: Cartesian to polar
# - coordinate transformation regression case: polar to Cartesian
# - coordinate transformation regression case: Cartesian to spherical
# - coordinate transformation regression case: spherical to Cartesian
# - vector-calculus regression case: gradient
# - vector-calculus regression case: divergence
# - vector-calculus regression case: Laplacian
# - coordinate-field transformation regression case


def test_translated_to_polar_symbolic_point_formula():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    x, y = cart.symbols()
    mapped = transform_coordinates(cart, polar, (x, y))

    assert sp.simplify(mapped[0] - sp.sqrt(x**2 + y**2)) == 0
    assert mapped[1] == sp.atan2(y, x)



def test_translated_from_polar_symbolic_point_formula():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    mapped = transform_coordinates(polar, cart, (r, theta))

    assert sp.simplify(mapped[0] - r * sp.cos(theta)) == 0
    assert sp.simplify(mapped[1] - r * sp.sin(theta)) == 0



def test_translated_polar_roundtrip_for_concrete_point():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)

    xy_point = transform_coordinates(polar, cart, (sp.Integer(2), sp.pi / 3))
    roundtrip = transform_coordinates(cart, polar, tuple(xy_point))

    assert sp.simplify(roundtrip[0] - 2) == 0
    assert sp.simplify(roundtrip[1] - sp.pi / 3) == 0


def test_translated_to_spherical_symbolic_point_formula():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    x, y, z = cart.symbols()
    mapped = transform_coordinates(cart, spherical, (x, y, z))

    assert sp.simplify(mapped[0] - sp.sqrt(x**2 + y**2 + z**2)) == 0
    assert mapped[1] == sp.atan2(sp.sqrt(x**2 + y**2), z)
    assert mapped[2] == sp.atan2(y, x)



def test_translated_from_spherical_symbolic_point_formula():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()
    mapped = transform_coordinates(spherical, cart, (r, theta, phi))

    assert sp.simplify(mapped[0] - r * sp.sin(theta) * sp.cos(phi)) == 0
    assert sp.simplify(mapped[1] - r * sp.sin(theta) * sp.sin(phi)) == 0
    assert sp.simplify(mapped[2] - r * sp.cos(theta)) == 0



def test_translated_scalar_field_from_polar_to_cartesian():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    x, y = cart.symbols()
    mapping = coordinate_map(polar, cart)

    scalar_field = ScalarField(polar, r**2 * sp.cos(theta))
    transformed = scalar_field.transform(mapping)

    assert transformed.chart == cart
    assert sp.simplify(transformed.expr - x * sp.sqrt(x**2 + y**2)) == 0



def test_translated_radial_vector_field_from_polar_to_cartesian():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    x, y = cart.symbols()
    mapping = coordinate_map(polar, cart)

    radial = VectorField(polar, sp.Matrix([[1], [0]]), "contravariant")
    transformed = radial.transform(mapping)

    assert transformed.chart == cart
    assert sp.simplify(transformed.components[0] - x / sp.sqrt(x**2 + y**2)) == 0
    assert sp.simplify(transformed.components[1] - y / sp.sqrt(x**2 + y**2)) == 0



def test_translated_polar_gradient_formula():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    field_name = sp.Function("f")
    scalar_field = ScalarField(polar, field_name(r, theta))
    gradient = scalar_field.gradient()

    assert sp.simplify(gradient.components[0] - sp.diff(field_name(r, theta), r)) == 0
    assert sp.simplify(gradient.components[1] - sp.diff(field_name(r, theta), theta) / r**2) == 0



def test_translated_polar_divergence_formula():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    first_component = sp.Function("f")
    second_component = sp.Function("g")
    vector_field = VectorField(
        polar,
        sp.Matrix([[first_component(r, theta)], [second_component(r, theta)]]),
        "contravariant",
    )

    divergence = sp.expand(vector_field.divergence())
    expected = (
        sp.diff(first_component(r, theta), r)
        + first_component(r, theta) / r
        + sp.diff(second_component(r, theta), theta)
    )
    assert sp.simplify(divergence - expected) == 0



def test_translated_polar_laplacian_formula():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    field_name = sp.Function("f")
    scalar_field = ScalarField(polar, field_name(r, theta))

    laplacian = sp.expand(scalar_field.laplacian())
    expected = (
        sp.diff(field_name(r, theta), (r, 2))
        + sp.diff(field_name(r, theta), r) / r
        + sp.diff(field_name(r, theta), (theta, 2)) / r**2
    )
    assert sp.simplify(laplacian - expected) == 0


def test_translated_cylindrical_to_cartesian_symbolic_point_formula():
    cylindrical = coordinate_chart("Euclidean", "Cylindrical", 3)
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    r, theta, z = cylindrical.symbols()
    mapped = transform_coordinates(cylindrical, cart, (r, theta, z))

    assert sp.simplify(mapped[0] - r * sp.cos(theta)) == 0
    assert sp.simplify(mapped[1] - r * sp.sin(theta)) == 0
    assert sp.simplify(mapped[2] - z) == 0




def test_translated_constant_scalar_field_is_invariant_under_transform():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    mapping = coordinate_map(polar, cart)

    scalar_field = ScalarField(polar, sp.Integer(1))
    transformed = scalar_field.transform(mapping)

    assert transformed.chart == cart
    assert transformed.expr == 1



def test_translated_negative_radial_vector_from_polar_to_cartesian():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    x, y = cart.symbols()
    mapping = coordinate_map(polar, cart)

    inward_radial = VectorField(polar, sp.Matrix([[-1], [0]]), "contravariant")
    transformed = inward_radial.transform(mapping)

    assert transformed.chart == cart
    assert sp.simplify(transformed.components[0] + x / sp.sqrt(x**2 + y**2)) == 0
    assert sp.simplify(transformed.components[1] + y / sp.sqrt(x**2 + y**2)) == 0



def test_translated_cartesian_curl_formula_in_three_dimensions():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    x, y, z = cart.symbols()
    first_component = sp.Function("f")
    second_component = sp.Function("g")
    third_component = sp.Function("h")
    vector_field = VectorField(
        cart,
        sp.Matrix(
            [
                [first_component(x, y, z)],
                [second_component(x, y, z)],
                [third_component(x, y, z)],
            ]
        ),
        "contravariant",
    )

    curl_field = vector_field.curl()
    expected = sp.Matrix(
        [
            [
                -sp.diff(second_component(x, y, z), z)
                + sp.diff(third_component(x, y, z), y)
            ],
            [
                sp.diff(first_component(x, y, z), z)
                - sp.diff(third_component(x, y, z), x)
            ],
            [
                -sp.diff(first_component(x, y, z), y)
                + sp.diff(second_component(x, y, z), x)
            ],
        ]
    )

    assert curl_field.chart == cart
    assert sp.simplify(curl_field.components - expected) == sp.zeros(3, 1)



def test_translated_cartesian_curl_of_concrete_vector_field():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    x, y, z = cart.symbols()
    vector_field = VectorField(cart, sp.Matrix([[y], [-x], [z]]), "contravariant")

    curl_field = vector_field.curl()

    assert sp.simplify(curl_field.components - sp.Matrix([[0], [0], [-2]])) == sp.zeros(3, 1)


def test_translated_cartesian_position_vector_to_polar_components():
    # Reference source:
    # coordinate-field transformation regression case (test 2)
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    x, y = cart.symbols()
    mapping = coordinate_map(cart, polar)

    position_vector = VectorField(cart, sp.Matrix([[x], [y]]), "contravariant")
    transformed = position_vector.transform(mapping)

    assert transformed.chart == polar
    assert sp.simplify(transformed.components[0] - polar.symbols()[0]) == 0
    assert sp.simplify(transformed.components[1]) == 0



def test_translated_cartesian_scalar_to_spherical_formula():
    # Reference source:
    # coordinate-field transformation regression case (test 3)
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    x, y, z = cart.symbols()
    r, theta, phi = spherical.symbols()

    scalar_field = ScalarField(cart, x**2 + y**2)
    transformed = scalar_field.transform(coordinate_map(cart, spherical))

    assert transformed.chart == spherical
    assert sp.simplify(transformed.expr - r**2 * sp.sin(theta)**2) == 0



def test_translated_cylindrical_roundtrip_for_concrete_point():
    # reference source family:
    # coordinate-map regression case
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    cylindrical = coordinate_chart("Euclidean", "Cylindrical", 3)

    xyz_point = transform_coordinates(cylindrical, cart, (sp.Integer(2), sp.pi / 3, sp.Integer(5)))
    roundtrip = transform_coordinates(cart, cylindrical, tuple(xyz_point))

    assert sp.simplify(roundtrip[0] - 2) == 0
    assert sp.simplify(roundtrip[1] - sp.pi / 3) == 0
    assert sp.simplify(roundtrip[2] - 5) == 0



def test_translated_divergence_of_concrete_cartesian_vector_field():
    # Reference source:
    # documentation-style divergence regression case (test 4)
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    x, y, z = cart.symbols()
    vector_field = VectorField(cart, sp.Matrix([[y], [-x], [z]]), "contravariant")

    assert sp.simplify(vector_field.divergence() - 1) == 0





def test_translated_cylindrical_laplacian_formula():
    # Reference source:
    # documentation-style Laplacian regression case (test 2)
    cylindrical = coordinate_chart("Euclidean", "Cylindrical", 3)
    r, theta, z = cylindrical.symbols()
    field_name = sp.Function("f")
    scalar_field = ScalarField(cylindrical, field_name(r, theta, z))

    laplacian = sp.expand(scalar_field.laplacian())
    expected = (
        sp.diff(field_name(r, theta, z), (r, 2))
        + sp.diff(field_name(r, theta, z), r) / r
        + sp.diff(field_name(r, theta, z), (theta, 2)) / r**2
        + sp.diff(field_name(r, theta, z), (z, 2))
    )
    assert sp.simplify(laplacian - expected) == 0



def test_translated_polar_laplacian_of_sine_of_radius_squared():
    # Reference source:
    # documentation-style Laplacian regression case (test 3)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    scalar_field = ScalarField(polar, sp.sin(r**2))

    expected = 4 * sp.cos(r**2) - 4 * r**2 * sp.sin(r**2)
    assert sp.simplify(scalar_field.laplacian() - expected) == 0



def test_translated_cartesian_laplacian_of_x_squared():
    # Reference source:
    # documentation-style Laplacian regression case (test 4)
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    x, y, z = cart.symbols()
    scalar_field = ScalarField(cart, x**2)

    assert sp.simplify(scalar_field.laplacian() - 2) == 0



def test_translated_spherical_laplacian_formula():
    # Reference source:
    # documentation-style Laplacian regression case (test 9)
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()
    potential = sp.Function("V")
    scalar_field = ScalarField(spherical, potential(r, theta, phi))

    laplacian = sp.expand(scalar_field.laplacian())
    expected = sp.expand(
        sp.diff(potential(r, theta, phi), (r, 2))
        + 2 * sp.diff(potential(r, theta, phi), r) / r
        + sp.diff(potential(r, theta, phi), (theta, 2)) / r**2
        + sp.diff(potential(r, theta, phi), theta) / (r**2 * sp.tan(theta))
        + sp.diff(potential(r, theta, phi), (phi, 2)) / (r**2 * sp.sin(theta)**2)
    )
    assert laplacian == expected


def test_translated_spherical_dipole_potential_to_cartesian():
    # Reference source:
    # coordinate-field transformation regression case (test 10)
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()
    x, y, z = cart.symbols()
    p = sp.symbols("p")
    mapping = coordinate_map(spherical, cart)

    scalar_field = ScalarField(spherical, p * sp.cos(theta) / r**2)
    transformed = scalar_field.transform(mapping)

    assert transformed.chart == cart
    assert sp.simplify(transformed.expr - p * z / (x**2 + y**2 + z**2) ** sp.Rational(3, 2)) == 0



def test_translated_negative_gradient_of_spherical_dipole_potential():
    # Reference source:
    # coordinate-field transformation regression case (test 11)
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()
    p = sp.symbols("p")

    scalar_field = ScalarField(spherical, p * sp.cos(theta) / r**2)
    gradient = scalar_field.gradient()
    electric_field = VectorField(spherical, -gradient.components, gradient.variance)

    expected = sp.Matrix(
        [
            [2 * p * sp.cos(theta) / r**3],
            [p * sp.sin(theta) / r**4],
            [0],
        ]
    )
    assert sp.simplify(electric_field.components - expected) == sp.zeros(3, 1)



def test_translated_spherical_dipole_field_to_cartesian():
    # Reference source:
    # coordinate-field transformation regression case (test 12)
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()
    x, y, z = cart.symbols()
    p = sp.symbols("p")
    mapping = coordinate_map(spherical, cart)

    scalar_field = ScalarField(spherical, p * sp.cos(theta) / r**2)
    gradient = scalar_field.gradient()
    electric_field = VectorField(spherical, -gradient.components, gradient.variance)
    transformed = electric_field.transform(mapping)

    expected = sp.Matrix(
        [
            [3 * p * x * z / (x**2 + y**2 + z**2) ** sp.Rational(5, 2)],
            [3 * p * y * z / (x**2 + y**2 + z**2) ** sp.Rational(5, 2)],
            [
                -p * (x**2 + y**2) / (x**2 + y**2 + z**2) ** sp.Rational(5, 2)
                + 2 * p * z**2 / (x**2 + y**2 + z**2) ** sp.Rational(5, 2)
            ],
        ]
    )

    assert transformed.chart == cart
    assert sp.simplify(transformed.components - expected) == sp.zeros(3, 1)



def test_translated_dipole_transform_commutes_with_negative_gradient():
    # Reference source:
    # coordinate-field transformation regression case (test 13)
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()
    p = sp.symbols("p")
    mapping = coordinate_map(spherical, cart)

    spherical_potential = ScalarField(spherical, p * sp.cos(theta) / r**2)
    cartesian_potential = spherical_potential.transform(mapping)

    spherical_gradient = spherical_potential.gradient()
    transformed_negative_gradient = VectorField(
        spherical,
        -spherical_gradient.components,
        spherical_gradient.variance,
    ).transform(mapping)

    cartesian_gradient = ScalarField(cart, cartesian_potential.expr).gradient()
    negative_cartesian_gradient = VectorField(
        cart,
        -cartesian_gradient.components,
        cartesian_gradient.variance,
    )

    assert sp.simplify(
        transformed_negative_gradient.components - negative_cartesian_gradient.components
    ) == sp.zeros(3, 1)



def test_translated_cylindrical_gradient_formula():
    # Reference source:
    # documentation-style gradient regression case (test 2)
    cylindrical = coordinate_chart("Euclidean", "Cylindrical", 3)
    r, theta, z = cylindrical.symbols()
    field_name = sp.Function("f")
    scalar_field = ScalarField(cylindrical, field_name(r, theta, z))
    gradient = scalar_field.gradient()

    expected = sp.Matrix(
        [
            [sp.diff(field_name(r, theta, z), r)],
            [sp.diff(field_name(r, theta, z), theta) / r**2],
            [sp.diff(field_name(r, theta, z), z)],
        ]
    )
    assert sp.simplify(gradient.components - expected) == sp.zeros(3, 1)



def test_translated_cylindrical_scalar_laplacian_formula():
    # Reference source:
    # documentation-style Laplacian regression case (test 6)
    cylindrical = coordinate_chart("Euclidean", "Cylindrical", 3)
    r, theta, z = cylindrical.symbols()
    field_name = sp.Function("f")
    scalar_field = ScalarField(cylindrical, field_name(r, theta, z))

    laplacian = sp.expand(scalar_field.laplacian())
    expected = (
        sp.diff(field_name(r, theta, z), (r, 2))
        + sp.diff(field_name(r, theta, z), r) / r
        + sp.diff(field_name(r, theta, z), (theta, 2)) / r**2
        + sp.diff(field_name(r, theta, z), (z, 2))
    )
    assert sp.simplify(laplacian - expected) == 0



def test_translated_spherical_scalar_laplacian_formula():
    # Reference source:
    # documentation-style Laplacian regression case (test 8)
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()
    field_name = sp.Function("f")
    scalar_field = ScalarField(spherical, field_name(r, theta, phi))

    laplacian = sp.expand(scalar_field.laplacian())
    expected = (
        sp.diff(field_name(r, theta, phi), (r, 2))
        + 2 * sp.diff(field_name(r, theta, phi), r) / r
        + sp.diff(field_name(r, theta, phi), (theta, 2)) / r**2
        + sp.diff(field_name(r, theta, phi), theta) / (r**2 * sp.tan(theta))
        + sp.diff(field_name(r, theta, phi), (phi, 2)) / (r**2 * sp.sin(theta) ** 2)
    )
    assert sp.simplify(laplacian - expected) == 0


def test_translated_cartesian_to_polar_concrete_point():
    # Reference source:
    # coordinate-map regression case
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)

    mapped = transform_coordinates(cart, polar, (sp.Integer(1), -sp.Integer(1)))

    assert sp.simplify(mapped[0] - sp.sqrt(2)) == 0
    assert sp.simplify(mapped[1] + sp.pi / 4) == 0



def test_translated_spherical_to_cartesian_concrete_point():
    # Reference source:
    # coordinate-map regression case
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    cart = coordinate_chart("Euclidean", "Cartesian", 3)

    mapped = transform_coordinates(spherical, cart, (sp.Integer(1), sp.pi / 4, sp.pi / 2))
    expected = sp.Matrix([[0], [sp.sqrt(2) / 2], [sp.sqrt(2) / 2]])

    assert sp.simplify(sp.Matrix(mapped) - expected) == sp.zeros(3, 1)



def test_translated_spherical_metric_tensor_formula():
    # Reference source:
    # documentation-style coordinate metadata regression case
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()

    expected = sp.diag(1, r**2, r**2 * sp.sin(theta) ** 2)
    assert spherical.metric() == expected



def test_translated_spherical_scale_factors_formula():
    # Reference source:
    # documentation-style coordinate metadata regression case
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()

    assert spherical.scale_factors() == (1, sp.Abs(r), sp.Abs(r * sp.sin(theta)))



def test_translated_spherical_volume_factor_formula():
    # Reference source:
    # documentation-style coordinate metadata regression case
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()

    assert sp.simplify(spherical.sqrt_metric_det() - r**2 * sp.Abs(sp.sin(theta))) == 0



def test_translated_coordinate_map_data_properties_list_contains_core_entries():
    # Reference source:
    # documentation-style coordinate-map metadata regression case
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    mapping = coordinate_map(cart, polar)
    properties = set(mapping.map_properties())

    expected_subset = {
        "source",
        "target",
        "mapping_exprs",
        "inverse_mapping_exprs",
        "simplified_inverse_mapping_exprs",
        "jacobian",
        "jacobian_determinant",
        "inverse_available",
        "symbolic_inverse_kind",
    }
    assert expected_subset.issubset(properties)



def test_translated_cartesian_to_polar_mapping_jacobian_determinant():
    # Reference source:
    # documentation-style coordinate-map metadata regression case
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    x, y = cart.symbols()
    mapping = coordinate_map(cart, polar)

    assert sp.simplify(mapping.jacobian_det((x, y)) - 1 / sp.sqrt(x**2 + y**2)) == 0



def test_translated_cartesian_to_polar_mapping_jacobian_formula():
    # Reference source:
    # documentation-style coordinate-map metadata regression case
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    x, y = cart.symbols()
    mapping = coordinate_map(cart, polar)

    expected = sp.Matrix(
        [
            [x / sp.sqrt(x**2 + y**2), y / sp.sqrt(x**2 + y**2)],
            [-y / (x**2 + y**2), x / (x**2 + y**2)],
        ]
    )
    assert sp.simplify(mapping.jacobian((x, y)) - expected) == sp.zeros(2, 2)



def test_translated_polar_to_cartesian_mapping_and_inverse_jacobian_are_inverse():
    # Reference source:
    # documentation-style coordinate-map metadata regression case
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r_pos = sp.Symbol("r_pos", positive=True, real=True)
    theta = sp.Symbol("theta", real=True)
    x, y = cart.symbols()
    mapping = coordinate_map(polar, cart)

    jacobian = mapping.jacobian((r_pos, theta))
    inverse_jacobian = mapping.inverse_jacobian((x, y)).subs(
        {x: r_pos * sp.cos(theta), y: r_pos * sp.sin(theta)}
    )
    product = sp.simplify(jacobian * inverse_jacobian)

    assert product == sp.eye(2)



def test_translated_spherical_mapping_jacobian_determinant_formula():
    # Reference source:
    # documentation-style coordinate-map metadata regression case
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    r, theta, phi = spherical.symbols()
    mapping = coordinate_map(spherical, cart)

    assert sp.simplify(mapping.jacobian_det((r, theta, phi)) - r**2 * sp.sin(theta)) == 0



def test_translated_cartesian_gradient_of_sine_of_radius_squared():
    # Reference source:
    # documentation-style gradient regression case
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    scalar_field = ScalarField(cart, sp.sin(x**2 + y**2))

    expected = sp.Matrix([[2 * x * sp.cos(x**2 + y**2)], [2 * y * sp.cos(x**2 + y**2)]])
    assert sp.simplify(scalar_field.gradient().components - expected) == sp.zeros(2, 1)



def test_translated_cartesian_gradient_of_xyz():
    # Reference source:
    # documentation-style gradient regression case
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    x, y, z = cart.symbols()
    scalar_field = ScalarField(cart, x * y * z)

    assert sp.simplify(scalar_field.gradient().components - sp.Matrix([[y * z], [x * z], [x * y]])) == sp.zeros(3, 1)



def test_translated_spherical_divergence_of_constant_vector_components():
    # Reference source:
    # documentation-style divergence regression case
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()
    vector_field = VectorField(spherical, sp.Matrix([[1], [1], [1]]), "contravariant")

    expected = 2 / r + sp.cot(theta)
    assert sp.simplify(vector_field.divergence() - expected) == 0


from tensoratlas import (
    list_charts,
    list_maps,
    list_charts_with_orthogonal_metric,
    chart_property_names,
    mapping_property_names,
    transform_field,
)


def test_registered_chart_catalog_contains_expected_entries():
    charts = set(list_charts())

    assert ("Euclidean", "Polar", 2) in charts
    assert ("Euclidean", "Spherical", 3) in charts
    assert ("Euclidean", "Cylindrical", 3) in charts


def test_registered_orthogonal_chart_catalog_contains_expected_entries():
    charts = set(list_charts_with_orthogonal_metric())

    assert ("Euclidean", "Polar", 2) in charts
    assert ("Euclidean", "Spherical", 3) in charts
    assert ("Euclidean", "Cylindrical", 3) in charts


def test_registered_map_catalog_contains_expected_entries():
    maps = set(list_maps())

    assert (("Euclidean", "Cartesian", 2), ("Euclidean", "Polar", 2)) in maps
    assert (("Euclidean", "Polar", 2), ("Euclidean", "Cartesian", 2)) in maps
    assert (("Euclidean", "Cartesian", 3), ("Euclidean", "Spherical", 3)) in maps
    assert (("Euclidean", "Spherical", 3), ("Euclidean", "Cartesian", 3)) in maps


def test_spherical_chart_property_names_include_expected_entries():
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    props = set(chart_property_names(spherical))

    assert "metric_tensor" in props
    assert "scale_factors" in props
    assert "sqrt_metric_det" in props
    assert "coordinate_domains" in props


def test_cartesian_to_polar_mapping_property_names_include_expected_entries():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    props = set(mapping_property_names(coordinate_map(cart, polar)))

    assert "mapping_exprs" in props
    assert "jacobian" in props
    assert "jacobian_determinant" in props
    assert "inverse_available" in props


def test_prolate_spheroidal_to_cartesian_mapping_formula():
    xi = sp.Symbol("xi", real=True)
    eta = sp.Symbol("eta", real=True)
    phi = sp.Symbol("phi", real=True)
    a = sp.Symbol("a", positive=True, real=True)

    prolate = coordinate_chart("Euclidean", "ProlateSpheroidal", 3)
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    mapped = transform_coordinates(prolate, cart, (xi, eta, phi))

    assert sp.simplify(mapped[0] - a * sp.cos(phi) * sp.sin(eta) * sp.sinh(xi)) == 0
    assert sp.simplify(mapped[1] - a * sp.sin(phi) * sp.sin(eta) * sp.sinh(xi)) == 0
    assert sp.simplify(mapped[2] - a * sp.cos(eta) * sp.cosh(xi)) == 0


def test_prolate_spheroidal_to_cartesian_concrete_point():
    a = sp.Symbol("a", positive=True, real=True)
    prolate = coordinate_chart("Euclidean", "ProlateSpheroidal", 3)
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    mapped = transform_coordinates(prolate, cart, (sp.Integer(2), sp.pi / 4, sp.Integer(0)))

    assert sp.simplify(mapped[0] - a * sp.sinh(2) / sp.sqrt(2)) == 0
    assert sp.simplify(mapped[1]) == 0
    assert sp.simplify(mapped[2] - a * sp.cosh(2) / sp.sqrt(2)) == 0


def test_spherical_curl_of_purely_radial_field_is_zero():
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()
    radial_profile = sp.Function("radial_profile")
    field = VectorField(spherical, sp.Matrix([[radial_profile(r)], [0], [0]]), "contravariant")

    assert field.curl().components == sp.zeros(3, 1)


def test_cylindrical_unit_axial_vector_transforms_to_cartesian_unit_axial_vector():
    cylindrical = coordinate_chart("Euclidean", "Cylindrical", 3)
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    field = VectorField(cylindrical, sp.Matrix([[0], [0], [1]]), "contravariant")
    transformed = transform_field(field, cylindrical, cart)

    assert transformed.components == sp.Matrix([[0], [0], [1]])


def test_translated_prolate_metric_formula():
    prolate = coordinate_chart("Euclidean", "ProlateSpheroidal", 3)
    mu, nu, phi = prolate.symbols()
    a = sp.Symbol("a", positive=True, real=True)

    expected = sp.diag(
        a**2 * (sp.sin(nu)**2 + sp.sinh(mu)**2),
        a**2 * (sp.sin(nu)**2 + sp.sinh(mu)**2),
        a**2 * sp.sin(nu)**2 * sp.sinh(mu)**2,
    )
    assert prolate.metric() == expected


def test_translated_prolate_inverse_metric_formula():
    prolate = coordinate_chart("Euclidean", "ProlateSpheroidal", 3)
    mu, nu, phi = prolate.symbols()
    a = sp.Symbol("a", positive=True, real=True)

    expected = sp.diag(
        1 / (a**2 * (sp.sin(nu)**2 + sp.sinh(mu)**2)),
        1 / (a**2 * (sp.sin(nu)**2 + sp.sinh(mu)**2)),
        1 / (a**2 * sp.sin(nu)**2 * sp.sinh(mu)**2),
    )
    assert sp.simplify(prolate.inverse_metric() - expected) == sp.zeros(3, 3)


def test_translated_elliptic_to_cartesian_mapping_formula():
    elliptic = coordinate_chart("Euclidean", "Elliptic", 2)
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    mu, nu = elliptic.symbols()
    a = sp.Symbol("a", positive=True, real=True)

    mapped = transform_coordinates(elliptic, cart, (mu, nu))

    assert sp.simplify(mapped[0] - a * sp.cos(nu) * sp.cosh(mu)) == 0
    assert sp.simplify(mapped[1] - a * sp.sin(nu) * sp.sinh(mu)) == 0


def test_translated_elliptic_scale_factors_formula():
    elliptic = coordinate_chart("Euclidean", "Elliptic", 2)
    mu, nu = elliptic.symbols()
    a = sp.Symbol("a", positive=True, real=True)
    expected = a * sp.sqrt(sp.sin(nu)**2 + sp.sinh(mu)**2)

    scale_factors = elliptic.scale_factors()

    assert sp.simplify(scale_factors[0] - expected) == 0
    assert sp.simplify(scale_factors[1] - expected) == 0


def test_translated_bipolar_to_cartesian_mapping_formula():
    bipolar = coordinate_chart("Euclidean", "Bipolar", 2)
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    sigma, tau = bipolar.symbols()
    a = sp.Symbol("a", positive=True, real=True)

    mapped = transform_coordinates(bipolar, cart, (sigma, tau))
    common_den = sp.cosh(tau) - sp.cos(sigma)

    assert sp.simplify(mapped[0] - a * sp.sinh(tau) / common_den) == 0
    assert sp.simplify(mapped[1] - a * sp.sin(sigma) / common_den) == 0


def test_translated_cylindrical_to_spherical_mapping_formula():
    cylindrical = coordinate_chart("Euclidean", "Cylindrical", 3)
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, z = cylindrical.symbols()

    mapped = transform_coordinates(cylindrical, spherical, (r, theta, z))

    assert sp.simplify(mapped[0] - sp.sqrt(r**2 + z**2)) == 0
    assert mapped[1] == sp.atan2(r, z)
    assert mapped[2] == theta


def test_translated_cartesian_divergence_of_curl_is_zero():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    x, y, z = cart.symbols()
    first_component = sp.Function("f")
    second_component = sp.Function("g")
    third_component = sp.Function("h")
    vector_field = VectorField(
        cart,
        sp.Matrix(
            [
                [first_component(x, y, z)],
                [second_component(x, y, z)],
                [third_component(x, y, z)],
            ]
        ),
        "contravariant",
    )

    assert sp.simplify(vector_field.curl().divergence()) == 0


def test_translated_cartesian_curl_of_gradient_is_zero():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    x, y, z = cart.symbols()
    scalar_name = sp.Function("psi")
    scalar_field = ScalarField(cart, scalar_name(x, y, z))

    assert scalar_field.gradient().curl().components == sp.zeros(3, 1)
