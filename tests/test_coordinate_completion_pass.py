import sympy as sp

from tensoratlas.core import (
    coordinate_domain_assumptions,
    coordinate_map_between,
    complete_coordinate_metadata,
    coordinate_curl,
    coordinate_hessian,
    coordinate_laplacian,
    coordinate_to_physical_vector,
    list_standard_coordinates,
    physical_curl,
    physical_to_coordinate_vector,
    scale_factors,
    standard_coordinate_entry,
    standard_coordinate_map_to_cartesian,
    standard_metric,
    vector_laplacian,
)


def test_every_catalog_entry_has_metric_domains_and_singularity_metadata():
    for name in list_standard_coordinates():
        entry = standard_coordinate_entry(name)
        coords = entry.symbols()
        metric = entry.metric(coords)
        assert metric.shape == (entry.dimension, entry.dimension)
        assert set(entry.domains) == set(entry.coordinate_names)
        metadata = complete_coordinate_metadata(name, coords)
        assert metadata["dimension"] == entry.dimension
        assert metadata["metric"].shape == (entry.dimension, entry.dimension)
        assert isinstance(metadata["singularities"], tuple)
        assert isinstance(coordinate_domain_assumptions(entry, coords), tuple)


def test_catalogued_maps_have_jacobians_and_inferred_singularities():
    mapped = []
    for name in list_standard_coordinates():
        entry = standard_coordinate_entry(name)
        if entry.to_cartesian_builder is None:
            continue
        cmap = standard_coordinate_map_to_cartesian(name)
        mapped.append(name)
        jac = cmap.jacobian()
        assert len(jac) == entry.dimension
        assert len(jac[0]) == entry.dimension
        assert cmap.jacobian_determinant() is not None
        assert cmap.is_locally_invertible_condition() is not None
    assert {"polar", "spherical", "toroidal", "bispherical"} <= set(mapped)


def test_standard_transition_map_spherical_to_cylindrical_branch():
    cmap = coordinate_map_between("spherical", "cylindrical")
    r, theta, phi = cmap.source_symbols
    assert sp.simplify(cmap.forward[0] - r * sp.sin(theta)) == 0
    assert sp.simplify(cmap.forward[1] - phi) == 0
    assert sp.simplify(cmap.forward[2] - r * sp.cos(theta)) == 0
    assert cmap.inverse is not None
    reasons = {s.reason for s in cmap.singularities}
    assert "Jacobian determinant vanishes" in reasons


def test_physical_coordinate_component_roundtrip_in_spherical():
    r, theta, phi = sp.symbols("r theta phi", positive=True, real=True)
    metric = standard_metric("spherical", (r, theta, phi))
    physical = (sp.Integer(1), sp.Integer(2), sp.Integer(3))
    coord = physical_to_coordinate_vector(physical, metric)
    back = coordinate_to_physical_vector(coord, metric)
    assert all(sp.simplify(a - b) == 0 for a, b in zip(back, physical))
    h = scale_factors(metric)
    assert sp.simplify(h[0] - 1) == 0
    assert sp.simplify(h[1] - r) == 0


def test_operator_conventions_are_explicit():
    r, theta = sp.symbols("r theta", positive=True, real=True)
    metric = standard_metric("polar", (r, theta))
    f = r**2 * sp.cos(theta)
    ordinary = coordinate_hessian(f, (r, theta), metric=metric, convention="coordinate")
    covariant = coordinate_hessian(f, (r, theta), metric=metric, convention="covariant")
    assert sp.simplify(ordinary[1][1] + r**2 * sp.cos(theta)) == 0
    assert sp.simplify(covariant[1][1] - r**2 * sp.cos(theta)) == 0
    lap = coordinate_laplacian(f, (r, theta), metric=metric, convention="laplace_beltrami")
    assert sp.simplify(lap - 3 * sp.cos(theta)) == 0


def test_vector_laplacian_componentwise_convention_is_separate():
    x, y, z = sp.symbols("x y z", real=True)
    result = vector_laplacian((x**2, y**2, z**2), (x, y, z), metric=sp.eye(3), convention="componentwise")
    assert result == (2, 2, 2)


def test_physical_curl_for_cylindrical_azimuthal_field():
    rho, phi, z = sp.symbols("rho phi z", positive=True, real=True)
    metric = standard_metric("cylindrical", (rho, phi, z))
    # Physical vector A = rho e_phi has coordinate component A^phi = 1.
    coord_curl = coordinate_curl((0, 1, 0), (rho, phi, z), metric=metric)
    phys = coordinate_to_physical_vector(coord_curl, metric)
    assert sp.simplify(phys[2] - 2) == 0
    assert physical_curl((0, rho, 0), (rho, phi, z), metric=metric)[2] == phys[2]
