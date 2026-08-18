import sympy as sp

from tensoratlas import TensorField, coordinate_chart, metric_tensor
from tensoratlas.normal_forms import TNFMatrix, TNFTensorArray, as_tnf_array
from tensoratlas.tensor_indices import _classify_special_tensor


def test_nfmatrix_entrywise_and_matrix_ops():
    a = TNFMatrix(2, 2, ((1, 2), (3, 4)))
    b = TNFMatrix(2, 2, ((5, 6), (7, 8)))

    assert (a + b) == TNFMatrix(2, 2, ((6, 8), (10, 12)))
    assert (2 * a) == TNFMatrix(2, 2, ((2, 4), (6, 8)))
    assert (a * 2) == TNFMatrix(2, 2, ((2, 4), (6, 8)))
    assert (a @ TNFMatrix(2, 1, ((1,), (0,)))) == TNFMatrix(2, 1, ((1,), (3,)))


def test_as_tnf_array_accepts_nested_sequences():
    arr = as_tnf_array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert isinstance(arr, TNFTensorArray)
    assert arr.shape == (2, 2, 2)
    assert arr[(1, 0, 1)] == 6


def test_tensorfield_accepts_nested_sequences():
    cart = coordinate_chart('Euclidean', 'Cartesian', 2)
    tensor = TensorField(cart, [[1, 0], [0, 1]], 'll')
    assert tensor.components.shape == (2, 2)
    assert tensor.components[(1, 1)] == 1


def test_special_tensor_classification_uses_nf_metric_path():
    polar = coordinate_chart('Euclidean', 'Polar', 2)
    g = metric_tensor(polar, 'll')
    assert _classify_special_tensor(g) == 'metric_ll'


def test_volume_form_uses_chart_sqrt_metric_det():
    from tensoratlas import coordinate_chart, volume_form

    sph = coordinate_chart('Euclidean', 'Spherical', 3)
    vol = volume_form(sph)
    r, theta, _ = sph.symbols()
    assert vol.components[(0, 1, 2)] == sp.simplify(sp.Abs(r**2 * sp.sin(theta)))
