import sympy as sp

from tensoratlas.core import (
    ComponentTensor,
    CoordinateSystem,
    CoordinateTransform,
    Manifold,
    TensorHead,
    christoffel_symbols_from_metric,
    einstein_component_tensor,
    inverse_metric_from_metric,
    metric_component_tensor,
    metric_geometry,
    ricci_component_tensor,
    riemann_component_tensor,
    scalar_curvature_component,
)


def test_component_tensor_uses_head_symmetry_for_storage_and_lookup():
    manifold = Manifold("M", dimension=2)
    coords = CoordinateSystem("cart", manifold, ("x", "y"))
    basis = coords.coordinate_basis()
    metric = metric_component_tensor("g", basis, ((1, 2), (2, 4)))
    assert metric.component(0, 1) == 2
    assert metric.component(1, 0) == 2
    assert (1, 0) not in metric.components

    form_head = TensorHead("F", (basis.index_type, basis.index_type), symmetry="antisymmetric", variance=("down", "down"))
    form = ComponentTensor(form_head, basis, {(1, 0): 3}, variance=("down", "down"))
    assert form.component(0, 1) == -3
    assert form.component(1, 0) == 3
    assert form.component(0, 0) == 0


def test_inverse_metric_from_metric_and_slot_raise_lower_roundtrip():
    manifold = Manifold("M", dimension=2)
    coords = CoordinateSystem("cart", manifold, ("x", "y"))
    basis = coords.coordinate_basis()
    metric = metric_component_tensor("g", basis, ((1, 0), (0, 4)))
    inv = inverse_metric_from_metric(metric)
    assert inv.component(0, 0) == 1
    assert inv.component(1, 1) == sp.Rational(1, 4)

    vector_head = TensorHead("V", (basis.index_type,), variance=("up",))
    covector_head = TensorHead("Vflat", (basis.index_type,), variance=("down",))
    vector = ComponentTensor(vector_head, basis, {(0,): 2, (1,): 3}, variance=("up",))
    lowered = vector.lowered_with(metric, 0, head=covector_head)
    assert lowered.to_dense() == [2, 12]
    raised = lowered.raised_with(inv, 0, head=vector_head)
    assert raised.to_dense() == [2, 3]


def test_christoffel_symbols_for_polar_metric():
    manifold = Manifold("M", dimension=2)
    r, theta = sp.symbols("r theta", positive=True)
    coords = CoordinateSystem("polar", manifold, ("r", "theta"))
    basis = coords.coordinate_basis()
    metric = metric_component_tensor("g", basis, ((1, 0), (0, r**2)))
    gamma = christoffel_symbols_from_metric(metric, coordinates=(r, theta))
    assert sp.simplify(gamma.coefficient(0, 1, 1) + r) == 0
    assert sp.simplify(gamma.coefficient(1, 0, 1) - 1 / r) == 0
    assert sp.simplify(gamma.coefficient(1, 1, 0) - 1 / r) == 0


def test_flat_polar_metric_has_zero_curvature():
    manifold = Manifold("M", dimension=2)
    r, theta = sp.symbols("r theta", positive=True)
    coords = CoordinateSystem("polar", manifold, ("r", "theta"))
    basis = coords.coordinate_basis()
    metric = metric_component_tensor("g", basis, ((1, 0), (0, r**2)))
    geometry = metric_geometry(metric, coordinates=(r, theta))
    assert geometry.scalar_curvature == 0
    assert geometry.ricci.components == {}
    assert geometry.einstein.components == {}


def test_unit_sphere_metric_curvature_objects():
    manifold = Manifold("S2", dimension=2)
    theta, phi = sp.symbols("theta phi", positive=True)
    coords = CoordinateSystem("sphere", manifold, ("theta", "phi"))
    basis = coords.coordinate_basis()
    metric = metric_component_tensor("g", basis, ((1, 0), (0, sp.sin(theta) ** 2)))
    geometry = metric_geometry(metric, coordinates=(theta, phi))
    assert sp.simplify(geometry.riemann.component(0, 1, 0, 1) - sp.sin(theta) ** 2) == 0
    assert sp.simplify(geometry.ricci.component(0, 0) - 1) == 0
    assert sp.simplify(geometry.ricci.component(1, 1) - sp.sin(theta) ** 2) == 0
    assert sp.simplify(geometry.scalar_curvature - 2) == 0
    assert geometry.einstein.components == {}


def test_coordinate_transform_builds_basis_transform():
    manifold = Manifold("M", dimension=2)
    source = CoordinateSystem("uv", manifold, ("u", "v"))
    target = CoordinateSystem("xy", manifold, ("x", "y"))
    transform = CoordinateTransform(source, target, ((2, 0), (0, 3)), ((sp.Rational(1, 2), 0), (0, sp.Rational(1, 3))))
    basis_transform = transform.basis_transform()
    vector_head = TensorHead("V", (source.index_type,), variance=("up",))
    vector = ComponentTensor(vector_head, source.coordinate_basis(), {(0,): 5}, variance=("up",))
    assert vector.transform(basis_transform).component(0) == 10
