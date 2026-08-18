import sympy as sp

from tensoratlas import (
    ScalarField,
    TensorField,
    VectorField,
    coordinate_chart,
    covariant_derivative,
    gradient,
    divergence,
    curl,
    laplacian,
    hessian,
    lie_derivative,
    tensor_covariant_derivative,
    tensor_hessian,
    tensor_lie_derivative,
    tensor_exterior_derivative,
    tensor_graph,
    tensor_interop_report,
    tensor_product,
)
from tensoratlas.normal_forms import tnf_build_array


def test_unified_calculus_dispatch_wrappers():
    chart = coordinate_chart('Euclidean', 'Cartesian', 3)
    x, y, z = chart.symbols()
    f = ScalarField(chart, x * y + z)
    v = VectorField(chart, sp.Matrix([[x], [y], [z]]))
    one_form = TensorField(chart, tnf_build_array((3,), lambda idx: (x, y, z)[idx[0]]), 'l')

    assert gradient(f).components.shape == (3, 1)
    assert covariant_derivative(f).components.shape == (3, 1)
    assert sp.simplify(divergence(v) - 3) == 0
    assert curl(v).components.shape == (3, 1)
    assert laplacian(f).free_symbols <= {x, y, z}
    assert hessian(f).components.shape == (3, 3)
    assert tensor_covariant_derivative(v).components.shape == (3, 3)
    assert tensor_hessian(f).components.shape == (3, 3)
    assert tensor_exterior_derivative(one_form).components.shape == (3, 3)


def test_tensor_lie_derivative_wrapper_and_tensorfield_curl():
    chart = coordinate_chart('Euclidean', 'Cartesian', 3)
    x, y, z = chart.symbols()
    X = VectorField(chart, sp.Matrix([[1], [0], [0]]))
    Y = VectorField(chart, sp.Matrix([[x], [y], [z]]))
    LY = lie_derivative(Y, X)
    assert isinstance(LY, VectorField)
    assert abs(int(sp.simplify(LY.components[0, 0]))) == 1

    one_form = TensorField(chart, tnf_build_array((3,), lambda idx: (y, -x, z)[idx[0]]), 'l')
    c = curl(one_form)
    assert isinstance(c, VectorField)
    assert c.components.shape == (3, 1)
    tl = tensor_lie_derivative(Y.as_tensor(), X)
    assert isinstance(tl, TensorField)


def test_tensor_graph_reports_contraction_plan_order():
    chart = coordinate_chart('Euclidean', 'Cartesian', 3)
    x, y, z = chart.symbols()
    A = TensorField(chart, tnf_build_array((3,), lambda idx: (x, y, z)[idx[0]]), 'u')
    B = TensorField(chart, tnf_build_array((3,), lambda idx: (1, 2, 3)[idx[0]]), 'l')
    obj = tensor_product(A, B)
    report = tensor_interop_report(obj)
    assert report['all_invariants_hold'] is True


def test_adaptive_geodesic_and_parallel_transport_helpers_exist_and_run():
    chart = coordinate_chart('Euclidean', 'Cartesian', 2)
    geod = chart.integrate_geodesic_adaptive((0, 0), (1, 0), 0.0, 1.0, dt=0.2, tol=1e-8)
    assert len(geod) >= 2
    rhs = chart.compile_parallel_transport_rhs(lambda t: (t, 0.0), lambda t: (1.0, 0.0))
    assert len(rhs(0.0, (1.0, 0.0))) == 2
    pt = chart.integrate_parallel_transport_adaptive(lambda t: (t, 0.0), lambda t: (1.0, 0.0), (1.0, 0.0), 0.0, 1.0, dt=0.2)
    assert len(pt) >= 2
