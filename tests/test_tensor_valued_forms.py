import pytest

from tensoratlas.tensor_valued_forms import (
    TensorValuedForm,
    cartan_first_equation,
    cartan_second_equation,
    connection_form,
    exterior_derivative_tvform,
    solder_form,
)


def test_rejects_component_key_rank_mismatch():
    with pytest.raises(ValueError, match="component key rank"):
        TensorValuedForm(1, (1, -1), {(0,): 1})


def test_rejects_bad_variance_entry():
    with pytest.raises(ValueError, match="variance"):
        TensorValuedForm(1, (0,), {(0,): 1})


def test_addition_rejects_incompatible_degree_or_variance():
    a = TensorValuedForm(1, (1,), {(0,): 1})
    b = TensorValuedForm(2, (1,), {(0,): 1})
    with pytest.raises(ValueError):
        _ = a + b


def test_formal_exterior_derivative_mode_is_explicit():
    a = TensorValuedForm(1, (1,), {(0,): 1})
    assert exterior_derivative_tvform(a).degree == 2
    with pytest.raises(NotImplementedError):
        exterior_derivative_tvform(a, mode="coordinate")


def test_cartan_equations_use_all_connection_slots():
    theta = TensorValuedForm(1, (1,), {(0,): 1})
    omega = TensorValuedForm(1, (1, -1), {(1, 0): 2})
    torsion = cartan_first_equation(theta, omega)
    assert (1,) in torsion.components


def test_cartan_first_and_second_equation_shapes():
    theta = solder_form((0, 1))
    omega = connection_form((0, 1))
    torsion = cartan_first_equation(theta, omega)
    curvature = cartan_second_equation(omega)
    assert torsion.degree == 2 and torsion.variance == (1,)
    assert curvature.degree == 2 and curvature.variance == (1, -1)
