from tensoratlas.declarations import declaration_registry
from tensoratlas.semantic_ir import indexed_tensor_expr, ir_node, curvature_ir, covariant_derivative_ir, symbol_ir, TensorExpr
from tensoratlas.tensor_expr_canonicalization import (
    CanonicalizationPolicy,
    IdentityRule,
    YoungSymmetryRule,
    canonical_tensor_expr_key,
    canonicalize_tensor_expr,
)


def _registry():
    reg = declaration_registry().declare_manifold("M", 4, signature=(-1, 1, 1, 1))
    reg = reg.declare_bundle("TM", "M").declare_index_family("latin", "TM", tuple("abcdefghi"), dummy_prefix="d")
    reg = reg.declare_dummy_pool("latin_dummies", "latin", prefix="d")
    reg = reg.declare_tensor("A", ("TM", "TM"), ("l", "l"), symmetries=(("antisymmetric", (0, 1)),))
    reg = reg.declare_tensor("S", ("TM", "TM"), ("l", "l"), symmetries=(("symmetric", (0, 1)),))
    reg = reg.declare_metric("g", "M", "TM")
    reg = reg.declare_connection("CD", "TM", metric="g", torsion_free=True, metric_compatible=True)
    return reg.declare_commutation_rule("riemann_commutator", "CD")


def test_dummy_index_renaming_is_bundle_aware_and_key_stable():
    reg = _registry()
    expr1 = ir_node("mul", indexed_tensor_expr("S", (("a", "l"), ("b", "l"))), indexed_tensor_expr("S", (("b", "l"), ("c", "l"))))
    expr2 = ir_node("mul", indexed_tensor_expr("S", (("a", "l"), ("e", "l"))), indexed_tensor_expr("S", (("e", "l"), ("c", "l"))))
    assert canonical_tensor_expr_key(expr1, registry=reg) == canonical_tensor_expr_key(expr2, registry=reg)
    report = canonicalize_tensor_expr(expr1, registry=reg)
    assert any(step.rule == "dummy_index_renaming" for step in report.steps)


def test_slot_symmetry_canonicalization_and_antisymmetric_zero():
    reg = _registry()
    symmetric = indexed_tensor_expr("S", (("b", "l"), ("a", "l")))
    report = canonicalize_tensor_expr(symmetric, registry=reg)
    assert report.canonical.metadata["indices"][0][0] == "a"
    antisym_repeat = indexed_tensor_expr("A", (("a", "l"), ("a", "l")))
    assert canonicalize_tensor_expr(antisym_repeat, registry=reg).canonical.kind == "zero"


def test_young_tableau_style_symmetry_support():
    reg = _registry().declare_tensor("Y", ("TM", "TM", "TM"), ("l", "l", "l"))
    y = indexed_tensor_expr("Y", (("c", "l"), ("b", "l"), ("a", "l")))
    report = canonicalize_tensor_expr(y, registry=reg, young_rules=(YoungSymmetryRule("Y", rows=((0, 1),), columns=((1, 2),)),))
    assert isinstance(report.canonical, TensorExpr)
    assert any(step.rule == "slot_symmetry" for step in report.steps)


def test_dimension_dependent_identity_rules():
    reg = declaration_registry().declare_manifold("M", 3).declare_bundle("TM", "M")
    weyl = ir_node("curvature:weyl", payload="W", dimension=3)
    report = canonicalize_tensor_expr(weyl, registry=reg)
    assert report.canonical.kind == "zero"


def test_metric_delta_contraction_normalization():
    reg = _registry()
    delta = indexed_tensor_expr("delta", (("a", "u"), ("b", "l")))
    tensor = indexed_tensor_expr("S", (("b", "l"), ("c", "l")))
    report = canonicalize_tensor_expr(ir_node("mul", delta, tensor), registry=reg)
    factors = report.canonical.children
    assert len(factors) == 1
    assert factors[0].metadata["indices"][0][0] == "a"


def test_canonical_product_ordering():
    reg = _registry()
    a = indexed_tensor_expr("S", (("a", "l"), ("b", "l")))
    b = indexed_tensor_expr("A", (("c", "l"), ("d", "l")))
    assert canonical_tensor_expr_key(ir_node("mul", a, b), registry=reg) == canonical_tensor_expr_key(ir_node("mul", b, a), registry=reg)


def test_covariant_derivative_ordering_records_commutation_policy():
    reg = _registry()
    x = symbol_ir("X")
    expr = covariant_derivative_ir(covariant_derivative_ir(x, index="a", connection="CD"), index="b", connection="CD")
    report = canonicalize_tensor_expr(expr, registry=reg)
    outer = report.canonical
    assert outer.kind == "covariant_derivative"
    assert outer.metadata.get("commutation_ordered") is True
    assert any(step.rule == "covariant_derivative_ordering" for step in report.steps)


def test_multiterm_identity_rule_and_provenance():
    reg = _registry()
    target = ir_node("curvature:bianchi_normal", payload="R_bianchi")
    rule = IdentityRule("first_bianchi_multiterm", "curvature_expr:cyclic_sum", target=target)
    expr = ir_node("curvature_expr:cyclic_sum", curvature_ir("Riemann", name="R"))
    report = canonicalize_tensor_expr(expr, registry=reg, identity_rules=(rule,))
    assert report.canonical.kind == "curvature:bianchi_normal"
    assert any(step.rule == "identity_rules" for step in report.steps)
    assert report.canonical.provenance.steps


def test_confluence_diagnostics_are_reported_for_overlapping_slot_rules():
    reg = _registry().declare_tensor(
        "B",
        ("TM", "TM", "TM"),
        ("l", "l", "l"),
        symmetries=(("symmetric", (0, 1)), ("antisymmetric", (1, 2))),
    )
    report = canonicalize_tensor_expr(indexed_tensor_expr("B", (("c", "l"), ("b", "l"), ("a", "l"))), registry=reg)
    assert any(diag.issue == "overlapping_slot_symmetries" for diag in report.diagnostics)
