import sympy as sp

from tensoratlas.core import (
    CoordinateMap,
    catalog_transition_map,
    list_standard_coordinates,
    standard_coordinate_entry,
    standard_coordinate_map_to_cartesian,
    standard_metric,
    coordinate_curl,
    coordinate_gradient,
    coordinate_hessian,
    coordinate_laplacian,
    coordinate_divergence,
    vector_laplacian,
)


def test_standard_coordinate_catalog_contains_expected_families():
    names = set(list_standard_coordinates())
    expected = {
        "cartesian2", "cartesian3", "cartesian4", "polar", "cylindrical", "spherical",
        "parabolic", "parabolic_cylindrical", "elliptic_cylindrical", "prolate_spheroidal",
        "oblate_spheroidal", "bipolar", "toroidal", "bispherical", "minkowski_cartesian4",
        "schwarzschild", "flrw_flat",
    }
    assert expected <= names


def test_spherical_map_has_inverse_jacobian_and_singularity_metadata():
    cmap = standard_coordinate_map_to_cartesian("spherical")
    assert isinstance(cmap, CoordinateMap)
    r, theta, phi = cmap.source_symbols
    assert cmap.forward == (
        r * sp.sin(theta) * sp.cos(phi),
        r * sp.sin(theta) * sp.sin(phi),
        r * sp.cos(theta),
    )
    assert cmap.inverse is not None
    det = sp.factor(cmap.jacobian_determinant())
    assert sp.simplify(det - r**2 * sp.sin(theta)) == 0
    assert cmap.inverse_jacobian() is not None
    reasons = {item.reason for item in cmap.singularities}
    assert "origin r=0" in reasons
    assert "polar axis sin(theta)=0" in reasons
    assert cmap.is_locally_invertible_condition() == sp.Ne(r**2 * sp.sin(theta), 0)


def test_polar_vector_calculus_public_api():
    r, theta = sp.symbols("r theta", positive=True, real=True)
    metric = standard_metric("polar", (r, theta))
    f = r**2 * sp.cos(theta)
    grad = coordinate_gradient(f, (r, theta), metric=metric)
    assert sp.simplify(grad[0] - 2 * r * sp.cos(theta)) == 0
    assert sp.simplify(grad[1] + sp.sin(theta)) == 0
    lap = coordinate_laplacian(f, (r, theta), metric=metric)
    assert sp.simplify(lap - 3 * sp.cos(theta)) == 0
    hess = coordinate_hessian(f, (r, theta), metric=metric)
    assert sp.simplify(hess[0][0] - 2 * sp.cos(theta)) == 0
    assert sp.simplify(hess[1][1] - r**2 * sp.cos(theta)) == 0


def test_cylindrical_divergence_and_curl():
    rho, phi, z = sp.symbols("rho phi z", positive=True, real=True)
    metric = standard_metric("cylindrical", (rho, phi, z))
    div = coordinate_divergence((rho, 0, z), (rho, phi, z), metric=metric)
    assert sp.simplify(div - 3) == 0
    curl = coordinate_curl((0, 0, rho**2), (rho, phi, z), metric=metric)
    assert sp.simplify(curl[0]) == 0
    assert sp.simplify(curl[1] + 2) == 0
    assert sp.simplify(curl[2]) == 0


def test_vector_laplacian_cartesian_componentwise_for_constant_metric():
    x, y, z = sp.symbols("x y z", real=True)
    result = vector_laplacian((x**2, y**2, z**2), (x, y, z), metric=sp.eye(3))
    assert result == (2, 2, 2)


def test_schwarzschild_catalog_metric_and_domains():
    entry = standard_coordinate_entry("schwarzschild")
    t, r, theta, phi = entry.symbols()
    M = sp.Symbol("M", positive=True, real=True)
    metric = entry.metric((t, r, theta, phi), {"M": M})
    assert sp.simplify(metric[0, 0] + (1 - 2 * M / r)) == 0
    assert sp.simplify(metric[1, 1] - 1 / (1 - 2 * M / r)) == 0
    reasons = {item.reason for item in entry.singularities((t, r, theta, phi), {"M": M})}
    assert "Schwarzschild horizon in this chart" in reasons

def test_catalog_transition_map_exposes_mapping_alias():
    mapping = catalog_transition_map("cartesian2", "polar")
    assert mapping.mapping == mapping.forward
    assert mapping.inverse is not None
