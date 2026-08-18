import random

import sympy as sp

from tensoratlas import TensorField, VectorField, coordinate_chart, coordinate_map, transform_coordinates


def test_orthonormal_component_roundtrip_in_polar_chart():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    vector = VectorField.from_components(polar, sp.Matrix([[r], [2]]), variance="contravariant", convention="orthonormal")
    orth = vector.components_in("orthonormal")
    assert sp.simplify(orth[0, 0] - r) == 0
    assert sp.simplify(orth[1, 0] - 2) == 0
    coord = vector.components_in("coordinate_basis")
    assert sp.simplify(coord[0, 0] - r) == 0
    assert sp.simplify(coord[1, 0] - 2 / r) == 0


def test_chart_domain_methods_and_metadata_completeness():
    spherical = coordinate_chart("Euclidean", "Spherical", 3)
    r, theta, phi = spherical.symbols()
    domain = spherical.domain_assumptions((r, theta, phi))
    assert domain is not None
    assert spherical.validate_point((sp.Integer(2), sp.pi / 3, sp.Integer(0)))
    assert not spherical.validate_point((sp.Integer(-1), sp.pi / 3, sp.Integer(0)))
    singularities = spherical.singularity_loci((r, theta, phi))
    assert singularities
    completeness = spherical.metadata_completeness()
    assert completeness["metric"]
    assert completeness["scale_factors"]
    assert completeness["coordinate_domains"]


def test_rank2_tensor_transform_identity_roundtrip():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    mapping = coordinate_map(cart, polar)
    x, y = cart.symbols()
    components = sp.MutableDenseNDimArray([[x**2, x * y], [x * y, y**2 + 1]])
    tensor = TensorField(cart, components, "ll")
    polar_tensor = tensor.transform(mapping)
    back = polar_tensor.transform(coordinate_map(polar, cart))
    for i in range(2):
        for j in range(2):
            assert sp.simplify(back.components[i, j] - tensor.components[i, j]) == 0


def test_randomized_coordinate_roundtrips_and_raise_lower_properties():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    to_polar = coordinate_map(cart, polar)
    to_cart = coordinate_map(polar, cart)
    x, y = cart.symbols()
    for _ in range(5):
        a = random.randint(1, 5)
        b = random.randint(1, 5)
        point = sp.Matrix([a, b])
        polar_point = transform_coordinates(cart, polar, point, to_polar)
        roundtrip = transform_coordinates(polar, cart, polar_point, to_cart)
        assert sp.simplify(roundtrip[0] - a) == 0
        assert sp.simplify(roundtrip[1] - b) == 0
    vec = VectorField(cart, sp.Matrix([[x + y], [x - y]]), "contravariant")
    lowered = vec.lower_index()
    raised = lowered.raise_index()
    assert sp.simplify(raised.components[0, 0] - vec.components[0, 0]) == 0
    assert sp.simplify(raised.components[1, 0] - vec.components[1, 0]) == 0
