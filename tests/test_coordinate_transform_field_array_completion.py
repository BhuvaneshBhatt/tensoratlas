import sympy as sp

from tensoratlas.core import (
    TensorArray,
    catalog_transition_map,
    coordinate_map_data,
    coordinate_transform_graph,
    solve_inverse_branches,
    standard_coordinate_system_data,
    standard_coordinate_entry,
    tensor_contract,
    tensor_dimensions,
    tensor_product,
    tensor_properties,
    tensor_transpose,
    transform_covector_field,
    transform_scalar_field,
    transform_tensor_density,
    transform_vector_field,
    transform_field,
)


def test_solve_inverse_branch_and_transform_data_for_polar():
    cmap = catalog_transition_map("polar", "cartesian2")
    branches = solve_inverse_branches(cmap)
    assert branches
    data = coordinate_map_data(cmap).as_dict()
    assert data["jacobian_determinant"] == cmap.jacobian_determinant()
    assert data["inverse_branches"]
    assert any("origin" in item.reason for item in data["singularities"])


def test_transform_graph_contains_standard_edges_and_added_entries():
    graph = coordinate_transform_graph()
    assert "spherical" in graph["cartesian3"]
    assert "cartesian3" in graph["spherical"]
    assert "cylindrical" in graph["spherical"]
    assert "rindler2" in graph
    assert "hyperbolic2_upper_half_plane" in graph


def test_standard_coordinate_system_data_for_every_catalog_entry():
    for name in ("polar", "spherical", "rindler2", "hyperbolic2", "isotropic_schwarzschild", "eddington_finkelstein_ingoing", "de_sitter_flat"):
        data = standard_coordinate_system_data(name)
        assert data["dimension"] == standard_coordinate_entry(name).dimension
        assert data["metric"].shape == (data["dimension"], data["dimension"])
        assert "singularities" in data
        assert "assumptions" in data


def test_transformed_scalar_vector_covector_polar_to_cartesian():
    cmap = catalog_transition_map("polar", "cartesian2")
    r, theta = cmap.source_symbols
    x, y = cmap.target_symbols
    scalar = transform_scalar_field(r**2, cmap)
    assert sp.simplify(scalar - (x**2 + y**2)) == 0

    vector = transform_vector_field((1, 0), cmap).components
    assert sp.simplify(vector[0] - x / sp.sqrt(x**2 + y**2)) == 0
    assert sp.simplify(vector[1] - y / sp.sqrt(x**2 + y**2)) == 0

    covector = transform_covector_field((1, 0), cmap).components
    assert sp.simplify(covector[0] - x / sp.sqrt(x**2 + y**2)) == 0
    assert sp.simplify(covector[1] - y / sp.sqrt(x**2 + y**2)) == 0


def test_transformed_tensor_density_weight():
    cmap = catalog_transition_map("polar", "cartesian2")
    r, theta = cmap.source_symbols
    x, y = cmap.target_symbols
    result = transform_tensor_density(((1, 0), (0, r**2)), cmap, ("down", "down"), 1)
    assert result.field_type == "tensor_density"
    assert result.density_weight == 1
    assert tensor_dimensions(result.as_tensor_array()) == (2, 2)


def test_transform_field_dispatcher():
    cmap = catalog_transition_map("polar", "cartesian2")
    r, theta = cmap.source_symbols
    scalar = transform_field(r, cmap, field_type="scalar")
    x, y = cmap.target_symbols
    assert sp.simplify(scalar - sp.sqrt(x**2 + y**2)) == 0
    vec = transform_field((1, 0), cmap, field_type="vector")
    assert vec.variance == ("up",)


def test_symbolic_array_tensor_operations():
    a = TensorArray((1, 2), ("up",), {"name": "a"})
    b = TensorArray((3, 4), ("down",), {"name": "b"})
    prod = tensor_product(a, b)
    assert tensor_dimensions(prod) == (2, 2)
    assert prod.component((1, 0)) == 6
    transposed = tensor_transpose(prod, (1, 0))
    assert transposed.component((0, 1)) == 6
    contracted = tensor_contract(prod, ((0, 1),))
    assert contracted.components == 11
    props = tensor_properties(prod)
    assert props["rank"] == 2
    assert props["variance"] == ("up", "down")

def test_tensor_contract_accepts_single_axis_pair_tuple():
    a = ((1, 2), (3, 4))
    b = ((0, 5), (6, 7))
    product = tensor_product(a, b)
    contracted = tensor_contract(product, (1, 2))
    assert contracted.dimensions == (2, 2)
    assert contracted.components == ((12, 19), (24, 43))

