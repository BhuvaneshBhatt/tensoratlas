from tensoratlas.core import (
    Basis,
    BasisTransform,
    ComponentTensor,
    ConnectionCoefficients,
    CoordinateSystem,
    Manifold,
    TensorHead,
    TensorKernelError,
    metric_component_tensor,
)


def test_coordinate_system_and_basis_validate_dimension():
    manifold = Manifold("M", dimension=2)
    coords = CoordinateSystem("polar", manifold, ("r", "theta"), domain={"r": "positive"})
    basis = coords.coordinate_basis()
    assert coords.dimension == 2
    assert basis.dimension == 2
    assert basis.structure_coefficient(0, 0, 1) == 0


def test_sparse_component_tensor_roundtrip_and_update():
    manifold = Manifold("M", dimension=2)
    coords = CoordinateSystem("cart", manifold, ("x", "y"))
    basis = coords.coordinate_basis()
    vector = TensorHead("V", (basis.index_type,), variance=("up",))
    comps = ComponentTensor(vector, basis, {(0,): 3})
    assert comps.component(0) == 3
    assert comps.component(1) == 0
    assert comps.with_component((1,), 5).to_dense() == [3, 5]


def test_basis_transform_handles_contravariant_and_covariant_slots():
    manifold = Manifold("M", dimension=2)
    coords = CoordinateSystem("cart", manifold, ("x", "y"))
    source = coords.coordinate_basis("e")
    target = Basis("f", coords, kind="frame")
    transform = BasisTransform(
        source,
        target,
        matrix=((2, 0), (0, 3)),
        inverse_matrix=((0.5, 0), (0, 1 / 3)),
    )
    vector = TensorHead("V", (source.index_type,), variance=("up",))
    covector = TensorHead("w", (source.index_type,), variance=("down",))
    assert ComponentTensor(vector, source, {(0,): 7}, variance=("up",)).transform(transform).component(0) == 14
    assert ComponentTensor(covector, source, {(1,): 6}, variance=("down",)).transform(transform).component(1) == 2


def test_metric_component_constructor_and_shape_validation():
    manifold = Manifold("M", dimension=2)
    coords = CoordinateSystem("cart", manifold, ("x", "y"))
    basis = coords.coordinate_basis()
    metric = metric_component_tensor("g", basis, ((1, 0), (0, 4)))
    assert metric.component(1, 1) == 4
    try:
        metric_component_tensor("bad", basis, ((1,),))
    except TensorKernelError:
        pass
    else:
        raise AssertionError("invalid metric shape should fail")


def test_connection_torsion_subtracts_noncoordinate_structure_coefficients():
    manifold = Manifold("M", dimension=2)
    coords = CoordinateSystem("chart", manifold, ("u", "v"))
    frame = Basis("frame", coords, kind="noncoordinate", structure_coefficients={(0, 0, 1): 5})
    connection = ConnectionCoefficients(frame, {(0, 0, 1): 7, (0, 1, 0): 2})
    assert connection.torsion_component(0, 0, 1) == 0
    torsion = connection.torsion_tensor()
    assert torsion.component(0, 0, 1) == 0
    assert torsion.component(0, 1, 0) == 0
