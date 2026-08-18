import sympy as sp

from tensoratlas import coordinate_chart, transform_coordinates
from tensoratlas.basis import gram_schmidt_frame
from tensoratlas.fields import TensorField
from tensoratlas.tensor_core import TensorObject, deviatoric_part, double_contraction, to_voigt
from tensoratlas.normal_forms import as_tnf_matrix, tnf_build_array


def test_new_coordinate_families_are_registered():
    assert coordinate_chart("Minkowski", "Cartesian", 4).dimension == 4
    assert coordinate_chart("Rindler", "Standard", 4).dimension == 4
    assert coordinate_chart("FRW", "Standard", 4).dimension == 4
    assert coordinate_chart("Hyperbolic", "PoincareDisk", 2).dimension == 2


def test_rindler_to_minkowski_transform():
    rind = coordinate_chart("Rindler", "Standard", 4)
    mink = coordinate_chart("Minkowski", "Cartesian", 4)
    rho, eta, y, z = sp.symbols('rho eta y z', positive=True, real=True)
    out = transform_coordinates(rind, mink, (rho, eta, y, z))
    assert out[0] == rho * sp.sinh(eta)
    assert out[1] == rho * sp.cosh(eta)


def test_torsion_and_nonmetricity_vanish_for_levi_civita_on_euclidean_plane():
    chart = coordinate_chart("Euclidean", "Polar", 2)
    coords = chart.symbols()
    torsion = chart.torsion_tensor(coords)
    nonmetricity = chart.nonmetricity_tensor(coords)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                assert sp.simplify(torsion[i, j, k]) == 0
                assert sp.simplify(nonmetricity[i, j, k]) == 0


def test_bianchi_residual_vanishes_on_sphere():
    chart = coordinate_chart("Euclidean", "Polar", 2)
    resid = chart.algebraic_bianchi_residual(chart.symbols())
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    assert sp.simplify(resid[a, b, c, d]) == 0


def test_deviatoric_and_double_contraction_and_voigt():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    base = sp.Matrix([[2, 1, 0], [1, 2, 0], [0, 0, 2]])
    arr = tnf_build_array((3, 3), lambda idx: base[idx])
    tf = TensorField(chart, arr, 'll')
    obj = TensorObject.from_tensor_field(tf)
    dev = deviatoric_part(obj)
    assert sp.simplify(dev.components[0, 0]) == 0
    dc = double_contraction(obj, obj)
    assert sp.simplify(dc.expr - 14) == 0
    vv = to_voigt(obj)
    assert vv.shape == (6, 1)


def test_gram_schmidt_frame_identity_in_cartesian():
    chart = coordinate_chart("Euclidean", "Cartesian", 2)
    frame = gram_schmidt_frame(chart, as_tnf_matrix(sp.eye(2)), coords=chart.symbols())
    assert frame == sp.eye(2)
