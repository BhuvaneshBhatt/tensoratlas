import sympy as sp
from tensoratlas import (
    coordinate_chart, TensorObject, TensorIndex, IndexedTensor,
    cotangent_basis, tangent_basis, frame_basis,
    frame_metric, frame_structure_coefficients, frame_connection_coefficients,
    canonicalize_indexed_expression, stronger_indexed_equal,
    validate_bundle_consistency, BundleCompatibilityError,
)
from tensoratlas.tensor_algebra import levi_civita_symbol, kronecker_delta_tensor


def test_user_defined_frame_metric_and_connection_cartesian():
    chart = coordinate_chart("Euclidean", "Cartesian", dimension=2)
    x, y = chart.symbols()
    e = frame_basis("rot", chart, lambda c: sp.Matrix([[sp.cos(c[0]), -sp.sin(c[0])], [sp.sin(c[0]), sp.cos(c[0])]]))
    g = frame_metric(e, (x, y))
    assert sp.simplify(g[0,0] - 1) == 0 and sp.simplify(g[1,1]-1) == 0
    C = frame_structure_coefficients(e, (x, y))
    # rotating frame depends on x, so at least one structure coefficient is nonzero
    assert any(sp.simplify(C[k,i,j]) != 0 for k in range(2) for i in range(2) for j in range(2))
    Gamma = frame_connection_coefficients(e, (x, y))
    assert any(sp.simplify(Gamma[k,i,j]) != 0 for k in range(2) for i in range(2) for j in range(2))


def test_bundle_consistency_detection():
    iA = TensorIndex('i','u','A')
    iB = TensorIndex('i','l','B')
    try:
        validate_bundle_consistency((iA, iB))
        assert False
    except BundleCompatibilityError:
        assert True


def test_strengthen_bundle_and_equality():
    chart = coordinate_chart("Euclidean", "Cartesian", dimension=2)
    tb = tangent_basis(chart)
    arr = sp.MutableDenseNDimArray.zeros(2)
    arr[(0,)] = 1
    arr[(1,)] = 2
    V = TensorObject(chart, arr, 'u', (tb,), name='V')
    a = IndexedTensor(V, (TensorIndex('i','u'),))
    b = IndexedTensor(V, (TensorIndex('i','u', tb.metadata['bundle'].name),))
    assert stronger_indexed_equal(a, b)


def test_epsilon_epsilon_to_delta_rewrite_rank3():
    chart = coordinate_chart("Euclidean", "Cartesian", dimension=3)
    eps_l = TensorObject.from_tensor_field(levi_civita_symbol(chart, 'lll'))
    eps_u = TensorObject.from_tensor_field(levi_civita_symbol(chart, 'uuu'))
    expr = IndexedTensor(eps_l, (TensorIndex('i','l'), TensorIndex('a','l'), TensorIndex('b','l'))) * IndexedTensor(eps_u, (TensorIndex('j','u'), TensorIndex('a','u'), TensorIndex('b','u')))
    can = canonicalize_indexed_expression(expr)
    delta = TensorObject.from_tensor_field(kronecker_delta_tensor(chart))
    arr = sp.MutableDenseNDimArray.zeros(3,3)
    for q in range(3):
        arr[q,q] = 2
    scaled = TensorObject(chart, arr, 'ul', (tangent_basis(chart), cotangent_basis(chart)))
    expected = IndexedTensor(scaled, (TensorIndex('j','u'), TensorIndex('i','l')))
    # identity: eps_iab eps^jab = 2 delta^j_i
    assert stronger_indexed_equal(can, expected)
