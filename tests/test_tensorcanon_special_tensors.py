
import pytest
import sympy as sp

from tensoratlas import (
    coordinate_chart,
    TensorObject,
    TensorIndex,
    indexed,
    indices,
    normalize_indexed_expression,
    stronger_indexed_equal,
    indexed_equal,
)
from tensoratlas.tensor_algebra import metric_tensor, permutation_tensor, kronecker_delta_tensor
from tensoratlas.tensor_indices import BundleCompatibilityError
from tensoratlas.basis import orthonormal_tangent_basis, orthonormal_cotangent_basis


def test_symmetry_aware_canonical_forms_metric_swap():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0, 1),)})
    i, j = indices("i_ j_")
    expr1 = indexed(g, i, j)
    expr2 = indexed(g, j, i)
    assert stronger_indexed_equal(expr1, expr2)


def test_metric_epsilon_normalization_inside_pipeline():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    ginv = TensorObject.from_tensor_field(metric_tensor(chart, "uu"), name="gU")
    epsL = TensorObject.from_tensor_field(permutation_tensor(chart, "lll"), name="eps")
    i, a, b, c = indices("i^ a^ b_ c_")
    left = indexed(ginv, i, a) * indexed(epsL, a.dual(), b, c)
    epsU1 = epsL.raise_slots([0])
    right = indexed(epsU1, i, b, c)
    assert indexed_equal(left, right)


def test_multibundle_reasoning_rejects_bad_addition():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    delta_std = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="d")
    delta_orth = delta_std.change_basis((orthonormal_tangent_basis(chart), orthonormal_cotangent_basis(chart)))
    iT = TensorIndex("i", "u", f"T({chart.chart_name})")
    jT = TensorIndex("j", "l", f"T({chart.chart_name})")
    iE = TensorIndex("i", "u", f"e({chart.chart_name})")
    jE = TensorIndex("j", "l", f"e({chart.chart_name})")
    expr = indexed(delta_std, iT, jT) + indexed(delta_orth, iE, jE)
    with pytest.raises(BundleCompatibilityError):
        normalize_indexed_expression(expr)
