from tensoratlas.differential_forms_frame import (
    FrameCalculusPolicy,
    basis_one_form,
    canonicalize_wedge,
    wedge_forms,
    hodge_star_form,
    exterior_covariant_derivative,
    frame_vector,
    frame_to_coframe,
    coframe_to_frame,
    connection_one_form,
    spin_connection_one_form,
    torsion_two_form,
    curvature_two_form,
    first_cartan_structure_equation,
    second_cartan_structure_equation,
    gamma_product,
    simplify_gamma_product,
    gamma_anticommutator_expr,
    dirac_operator_expr,
    lichnerowicz_example,
)
from tensoratlas.semantic_ir import TensorExpr, gamma_object_ir


def test_wedge_canonicalization_and_graded_signs():
    a = basis_one_form("theta1")
    b = basis_one_form("theta0")
    ab = wedge_forms(a, b)
    assert isinstance(ab, TensorExpr)
    assert ab.kind == "form:wedge"
    assert ab.metadata["basis_labels"] == ("theta0", "theta1")
    assert ab.metadata["coefficient"] == -1
    assert wedge_forms(a, a).kind == "zero"


def test_hodge_star_uses_signature_and_orientation_policy():
    policy = FrameCalculusPolicy(dimension=3, signature=(1, 1, 1), orientation="positive", coframe="theta")
    star = hodge_star_form(basis_one_form("theta0"), policy=policy)
    assert star.kind == "form:wedge"
    assert star.metadata["degree"] == 2
    assert star.metadata["basis_labels"] == ("theta1", "theta2")


def test_exterior_covariant_derivative_and_cartan_equations():
    theta = basis_one_form("theta0")
    dtheta = exterior_covariant_derivative(theta, connection="D")
    assert dtheta.kind == "form:exterior_covariant_derivative"
    assert dtheta.metadata["degree"] == 2
    first = first_cartan_structure_equation(0, connection="omega")
    second = second_cartan_structure_equation(0, 1, connection="omega")
    assert first.kind == "cartan:first_structure_equation"
    assert first.metadata["equals"].kind == "torsion:two_form"
    assert second.metadata["equals"].kind == "curvature:two_form"


def test_frame_coframe_conversion_and_connection_forms():
    v = frame_vector(0, frame="e")
    cov = frame_to_coframe(v, frame="e", coframe="theta")
    assert cov.kind == "form:basis"
    assert cov.payload == "theta0"
    assert coframe_to_frame(cov, frame="e", coframe="theta").kind == "frame:vector"
    omega = connection_one_form("omega", 0, 1)
    spin = spin_connection_one_form(upper=0, lower=1)
    assert omega.metadata["degree"] == 1
    assert spin.kind == "spin:connection_one_form"


def test_torsion_and_curvature_two_forms():
    assert torsion_two_form("omega", 0).metadata["degree"] == 2
    assert curvature_two_form("omega", 0, 1).metadata["degree"] == 2


def test_gamma_clifford_and_dirac_examples():
    ga = gamma_object_ir("gamma", indices=("a",))
    gb = gamma_object_ir("gamma", indices=("a",))
    simplified = simplify_gamma_product(gamma_product(ga, gb))
    assert simplified.kind == "metric:trace"
    anti = gamma_anticommutator_expr("a", "b")
    assert anti.kind == "gamma:anticommutator"
    assert anti.metadata["equals"].kind == "metric:component"
    dirac = dirac_operator_expr("psi", gamma_indices=("a", "b"))
    assert dirac.kind == "dirac:operator"
    assert len(dirac.children) == 2
    assert lichnerowicz_example("psi").kind == "dirac:lichnerowicz"
