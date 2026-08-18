
import sympy as sp
from tensoratlas import coordinate_chart, TensorObject, indexed, indices, normalize_indexed_expression, stronger_indexed_equal, frame_basis, curvature_two_forms, first_structure_equation_residuals, young_projector_data, young_irreducible_canonicalize
from tensoratlas.tensor_algebra import metric_tensor, permutation_tensor

def test_young_projector_data_and_irreducible_step():
    chart = coordinate_chart('Euclidean', 'Cartesian', 3)
    tf = permutation_tensor(chart, 'lll')
    to = TensorObject.from_tensor_field(tf, name='eps', symmetry_metadata={'antisymmetric': ((0,1,2),), 'young_tableaux': (((0,), (1,), (2,)),)})
    data = young_projector_data(((0,1),(2,)))
    assert data['hook_product'] != 0
    out = young_irreducible_canonicalize(to)
    assert out.components.shape == to.components.shape

def test_cartan_engine_identity_frame():
    chart = coordinate_chart('Euclidean', 'Cartesian', 3)
    fr = frame_basis('e', chart, lambda coords: sp.eye(chart.dimension))
    curv = curvature_two_forms(fr)
    residuals = first_structure_equation_residuals(fr)
    assert isinstance(curv, dict)
    assert isinstance(residuals, tuple)
    assert len(residuals) == chart.dimension

def test_epsilon_metric_partial_contraction_pipeline():
    chart = coordinate_chart('Euclidean', 'Cartesian', 3)
    gU = TensorObject.from_tensor_field(metric_tensor(chart, 'uu'), name='gU')
    epsL = TensorObject.from_tensor_field(permutation_tensor(chart, 'lll'), name='eps')
    i,a,b,c = indices('i^ a^ b_ c_')
    expr = indexed(gU, i, a) * indexed(epsL, a.dual(), b, c)
    norm = normalize_indexed_expression(expr)
    assert norm is not None

def test_symmetric_metric_still_equal_under_swap():
    chart = coordinate_chart('Euclidean', 'Cartesian', 3)
    g = TensorObject.from_tensor_field(metric_tensor(chart, 'll'), name='g', symmetry_metadata={'symmetric': ((0,1),)})
    i,j = indices('i_ j_')
    assert stronger_indexed_equal(indexed(g, i, j), indexed(g, j, i))
