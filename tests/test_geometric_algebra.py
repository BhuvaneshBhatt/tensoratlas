import pytest
import sympy as sp

from tensoratlas.geometric_algebra import GeometricAlgebra, reflect, rotor, rotate


def test_rejects_non_diagonal_metric_until_supported():
    a = sp.symbols("a")
    with pytest.raises(NotImplementedError):
        GeometricAlgebra(("e1", "e2"), metric=sp.Matrix([[1, a], [a, 1]]))


def test_repeated_exterior_blade_is_zero_but_basis_product_uses_metric():
    ga = GeometricAlgebra(("e1", "e2"), metric=(2, 3))
    assert ga.blade([0, 0]).is_zero()
    assert (ga.basis_product([0, 0])).coeffs == {(): sp.Integer(2)}


def test_lorentzian_basis_square_and_anticommutation():
    ga = GeometricAlgebra.spacetime()
    g0, g1, *_ = ga.basis_vectors()
    assert (g0 * g0).coeffs == {(): sp.Integer(-1)}
    assert (g1 * g1).coeffs == {(): sp.Integer(1)}
    assert (g0 * g1 + g1 * g0).is_zero()


def test_rotor_and_reflection_helpers_return_multivectors():
    ga = GeometricAlgebra.euclidean(2)
    e1, e2 = ga.basis_vectors()
    R = rotor(sp.symbols("theta"), e1.wedge(e2))
    rotated = rotate(e1, R)
    reflected = reflect(e1, e2)
    assert rotated.algebra == ga
    assert reflected.algebra == ga
