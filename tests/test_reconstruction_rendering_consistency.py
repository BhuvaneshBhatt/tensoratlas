from tensoratlas import coordinate_chart, indexed, indexed_equal, indices, normalize_indexed_expression, to_indexed_tensor_form, TensorObject
from tensoratlas.tensor_algebra import kronecker_delta_tensor, metric_tensor, tensor_product


def test_reconstruction_roundtrip_preserves_nf_equality():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    delta = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    metric = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0, 1),)})
    i, j, k, l = indices("i^ j_ j^ k_")
    expr = tensor_product(indexed(delta, i, j), indexed(metric, k, l))
    nf = to_indexed_tensor_form(expr)
    rebuilt = normalize_indexed_expression(expr)
    assert nf == to_indexed_tensor_form(rebuilt)
    assert indexed_equal(expr, rebuilt)


def test_reconstruction_keeps_distinct_free_names_for_repeated_slot_types():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    metric = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0, 1),)})
    i, j, k, l = indices("i_ j_ k_ l_")
    expr = tensor_product(indexed(metric, i, j), indexed(metric, k, l))
    rebuilt = normalize_indexed_expression(expr)
    text = str(rebuilt)
    assert text.count("f") >= 2
