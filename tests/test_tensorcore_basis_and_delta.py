import sympy as sp

from tensoratlas import (
    coordinate_chart, TensorObject, VectorField, TensorIndex, cotangent_basis, tangent_basis,
    kronecker_delta_tensor, levi_civita_symbol, metric_tensor, indices, frame_basis, coframe_basis,
    DifferentialForm, compose_tensors, stronger_indexed_equal
)


def test_bundle_aware_indices_validate():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    v = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[1],[2]]), 'contravariant'))
    good = TensorIndex('i', 'u', v.slot_bases[0].metadata['bundle'].name)
    assert v.with_indices(good).indices[0].bundle == good.bundle


def test_user_defined_frame_roundtrip():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, th = polar.symbols()
    e = frame_basis('E', polar, lambda c: sp.diag(1, c[0]), orthonormal=True)
    theta = coframe_basis('Theta', polar, lambda c: sp.diag(1, 1/c[0]), orthonormal=True)
    v = TensorObject.from_vector_field(VectorField(polar, sp.Matrix([[0],[1]]), 'contravariant'))
    vo = v.change_basis((e,))
    back = vo.change_basis((tangent_basis(polar),))
    assert sp.simplify(back.components[(1,)] - 1) == 0


def test_metric_delta_epsilon_rewrite_basic():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    delta = TensorObject.from_tensor_field(kronecker_delta_tensor(cart), name='d')
    eps1 = TensorObject.from_tensor_field(levi_civita_symbol(cart, 'lll'), name='eps')
    eps2 = TensorObject.from_tensor_field(levi_civita_symbol(cart, 'uuu'), name='eps')
    i,j,k = indices('i_ j_ k_')
    iu,ju,ku = indices('i^ j^ k^')
    out = (eps1.with_indices(i,j,k) * eps2.with_indices(iu,ju,ku)).evaluate()
    assert out.expr == 6
    ii,jj = indices('i^ i_')
    tr = delta.with_indices(ii,jj).evaluate()
    assert tr.expr == 3


def test_stronger_global_canonicalization_dummy_and_order():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x,y = cart.symbols()
    v = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[x],[y]]), 'contravariant'), name='V')
    w = TensorObject.from_tensor_field(v.to_vector_field().lower_index().as_tensor(), name='W')
    a = (v.with_indices(*indices('i^')) * w.with_indices(*indices('i_'))).evaluate()
    b = (w.with_indices(*indices('k_')) * v.with_indices(*indices('k^'))).evaluate()
    assert stronger_indexed_equal(a, b) or sp.simplify(a.expr - b.expr)==0


def test_cleaner_composition_dsl_and_spectral_tools():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    arr = sp.MutableDenseNDimArray.zeros(2,2)
    arr[0,0]=2; arr[1,1]=3
    A = TensorObject(cart, arr, 'ul', (tangent_basis(cart), cotangent_basis(cart)), name='A')
    B = compose_tensors(A, A)
    assert B.components[(0,0)] == 4
    assert set(A.eigenvals().keys()) == {2,3}
    assert A.singular_values()


def test_degree_aware_differential_form_layer():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x,y = cart.symbols()
    one = TensorObject(cart, sp.MutableDenseNDimArray([x,y]), 'l', (cotangent_basis(cart),), name='omega')
    form = DifferentialForm(one)
    assert form.degree == 1
    two = form.d()
    assert two.degree == 2


def test_rank2_bilinear_tools():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    g = TensorObject.from_tensor_field(metric_tensor(cart,'ll'), name='g')
    v = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[1],[2]]), 'contravariant'))
    q = g.quadratic_form(v)
    assert sp.simplify(q - 5) == 0
