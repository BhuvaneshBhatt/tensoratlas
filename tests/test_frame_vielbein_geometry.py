import sympy as sp

from tensoratlas.core import (
    Basis,
    ComponentTensor,
    CoordinateSystem,
    Manifold,
    TensorHead,
    Vielbein,
    christoffel_symbols_from_metric,
    covariant_derivative_components,
    dual_coframe,
    metric_component_tensor,
    spin_connection_from_vielbein,
)


def test_dual_coframe_pairing_and_metric_from_vielbein():
    manifold = Manifold("M", 2)
    coords = CoordinateSystem("polar", manifold, ("r", "theta"))
    coordinate_basis = coords.coordinate_basis()
    frame_basis = Basis("orthonormal", coords, kind="orthonormal")
    r = sp.Symbol("r", positive=True)
    vielbein = Vielbein(
        coordinate_basis,
        frame_basis,
        frame_to_coordinate=((1, 0), (0, 1 / r)),
        coordinate_to_frame=((1, 0), (0, r)),
        signature=(1, 1),
    )
    coframe = dual_coframe(frame_basis)
    assert coframe.pairing(0, 0) == 1
    assert coframe.pairing(0, 1) == 0
    metric = vielbein.metric_from_signature()
    assert metric.component(0, 0) == 1
    assert sp.simplify(metric.component(1, 1) - r**2) == 0


def test_spin_connection_from_polar_vielbein():
    manifold = Manifold("M", 2)
    coords = CoordinateSystem("polar", manifold, ("r", "theta"))
    coordinate_basis = coords.coordinate_basis()
    frame_basis = Basis("orthonormal", coords, kind="orthonormal")
    r, theta = sp.symbols("r theta", positive=True)
    metric = metric_component_tensor("g", coordinate_basis, ((1, 0), (0, r**2)))
    connection = christoffel_symbols_from_metric(metric, coordinates=(r, theta))
    vielbein = Vielbein(
        coordinate_basis,
        frame_basis,
        frame_to_coordinate=((1, 0), (0, 1 / r)),
        coordinate_to_frame=((1, 0), (0, r)),
        signature=(1, 1),
    )
    spin = spin_connection_from_vielbein(vielbein, connection, coordinates=(r, theta))
    assert sp.simplify(spin.coefficient(0, 1, 1) + 1) == 0
    assert sp.simplify(spin.coefficient(1, 0, 1) - 1) == 0


def test_component_covariant_derivative_of_vector_in_polar_coordinates():
    manifold = Manifold("M", 2)
    coords = CoordinateSystem("polar", manifold, ("r", "theta"))
    basis = coords.coordinate_basis()
    r, theta = sp.symbols("r theta", positive=True)
    metric = metric_component_tensor("g", basis, ((1, 0), (0, r**2)))
    connection = christoffel_symbols_from_metric(metric, coordinates=(r, theta))
    vector_head = TensorHead("V", (basis.index_type,), variance=("up",))
    vector = ComponentTensor(vector_head, basis, {(0,): r}, variance=("up",))
    deriv = covariant_derivative_components(vector, connection, coordinates=(r, theta))
    assert sp.simplify(deriv.component(0, 0) - 1) == 0
    assert deriv.component(0, 1) == 0
    assert deriv.component(1, 0) == 0
    assert sp.simplify(deriv.component(1, 1) - 1) == 0
