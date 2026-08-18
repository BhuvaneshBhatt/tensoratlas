from tensoratlas.declarations import standard_riemannian_registry
from tensoratlas.semantic_ir import TensorExpr, canonical_ir_key, covariant_derivative_ir
from tensoratlas.variational_tensor_expr import (
    BoundaryTerm,
    curvature_squared_variation_example,
    determinant_density_ir,
    einstein_hilbert_variation,
    extract_euler_lagrange_tensor_expr,
    f_of_r_variation_example,
    integrate_by_parts_tensor_expr,
    metric_ir,
    metric_variation_ir,
    mul_ir,
    perturbative_metric_expansion,
    variation_of_connection,
    variation_of_curvature,
    variation_of_inverse_metric,
    variation_of_metric,
    variation_of_metric_determinant,
)


def test_metric_inverse_and_determinant_variations_are_tensor_expr():
    reg = standard_riemannian_registry()
    dg = variation_of_metric(reg, ("a", "b"))
    dginv = variation_of_inverse_metric(reg, ("a", "b"))
    dsqrtg = variation_of_metric_determinant(reg)
    assert isinstance(dg, TensorExpr)
    assert dg.kind == "metric_variation"
    assert dginv.metadata["variation_rule"] == "delta_inverse_metric"
    assert dsqrtg.metadata["variation_rule"] == "delta_sqrt_det_metric"
    assert canonical_ir_key(dsqrtg)


def test_connection_and_curvature_variations_use_tensor_expr():
    reg = standard_riemannian_registry()
    dgamma = variation_of_connection(reg, indices=("a", "b", "c"))
    driem = variation_of_curvature(reg, "Riemann", indices=("a", "b", "c", "d"))
    dric = variation_of_curvature(reg, "Ricci", indices=("a", "b"))
    dscalar = variation_of_curvature(reg, "ScalarCurvature")
    assert isinstance(dgamma, TensorExpr)
    assert dgamma.metadata["variation_rule"] == "delta_levi_civita_connection"
    assert driem.metadata["variation_rule"] == "delta_riemann_from_delta_connection"
    assert dric.kind == "contract"
    assert dscalar.metadata["variation_rule"] == "delta_scalar_curvature"


def test_integration_by_parts_tracks_boundary_terms():
    reg = standard_riemannian_registry()
    coeff = metric_ir(reg, ("a", "b"))
    varied_derivative = covariant_derivative_ir(metric_variation_ir(reg, ("a", "b")), index="c", connection="CD")
    expr = mul_ir(coeff, varied_derivative)
    report = integrate_by_parts_tensor_expr(expr, field="g", discard_boundary=True)
    assert report.boundary_terms
    assert isinstance(report.boundary_terms[0], BoundaryTerm)
    assert report.steps == ("move_covariant_derivative_off_variation",)
    assert report.bulk.kind in {"neg", "mul", "add"}


def test_euler_lagrange_extraction_strips_metric_variation_factor():
    reg = standard_riemannian_registry()
    density = determinant_density_ir(reg)
    coeff = metric_ir(reg, ("a", "b"))
    expr = mul_ir(density, coeff, metric_variation_ir(reg, ("a", "b")))
    report = extract_euler_lagrange_tensor_expr(expr, field="g")
    assert report.euler_lagrange.metadata["role"] == "euler_lagrange"
    assert report.euler_lagrange.kind in {"mul", "indexed_tensor", "metric_density"}


def test_einstein_hilbert_variation_has_bulk_and_boundary():
    reg = standard_riemannian_registry()
    report = einstein_hilbert_variation(reg)
    assert report.action_density.metadata["action"] == "einstein_hilbert"
    assert report.boundary_terms[0].reason == "einstein_hilbert_boundary_term"
    assert report.euler_lagrange.metadata["variation_result"] == "einstein_hilbert_bulk"
    assert report.after_integration_by_parts.kind == "add"


def test_f_of_r_and_curvature_squared_examples_are_tensor_expr():
    reg = standard_riemannian_registry()
    fr = f_of_r_variation_example("f", reg)
    r2 = curvature_squared_variation_example("Scalar2", reg)
    assert fr.action_density.metadata["action"] == "f_of_R"
    assert fr.euler_lagrange.metadata["role"] == "f_R_euler_lagrange"
    assert r2.action_density.metadata["invariant"] == "Scalar2"
    assert r2.euler_lagrange.kind == "higher_curvature_eom"


def test_perturbative_expansion_around_background_metric():
    reg = standard_riemannian_registry()
    report = perturbative_metric_expansion(reg, background_metric="gbar", perturbation="h", order=2)
    assert report.expanded.metadata["expansion"] == "background_metric"
    assert set(report.terms_by_order) == {0, 1, 2}
    assert report.terms_by_order[0].metadata["background"] is True
    assert report.terms_by_order[1].kind == "mul"
