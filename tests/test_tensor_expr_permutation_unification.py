from tensoratlas.semantic_ir import indexed_tensor_expr, ir_node
from tensoratlas.tensor_expr_canonicalization import (
    LinearIdentityRule,
    LinearIdentityTerm,
    SlotPermutationRule,
    SlotSymmetryGroup,
    YoungProjector,
    YoungSymmetryRule,
    canonical_tensor_expr_key,
    canonicalize_tensor_expr,
)
from tensoratlas.declarations import declaration_registry


def test_signed_slot_permutation_group_canonicalizes_full_orbit():
    group = SlotSymmetryGroup(3, (SlotPermutationRule((1, 0, 2), -1), SlotPermutationRule((0, 2, 1), -1)))
    indices, sign, perm, zero = group.canonicalize_indices(("c", "a", "b"))
    assert indices == ("a", "b", "c")
    assert sign == 1
    assert perm == (1, 2, 0)
    assert not zero


def test_young_projector_supplies_row_and_column_generators():
    projector = YoungProjector(rows=((0, 1),), columns=((0, 2),))
    tensor = indexed_tensor_expr("Y", ("c", "b", "a"))
    terms = projector.project_terms(tensor)
    signs = {sign for sign, _ in terms}
    assert signs == {-1, 1}
    assert len(terms) >= 4


def test_central_canonicalizer_uses_permutation_group_for_declared_symmetry():
    reg = declaration_registry().declare_manifold("M", 4).declare_bundle("TM", "M")
    reg = reg.declare_tensor("A", ("TM", "TM", "TM"), variance="lll", symmetries=(("antisymmetric", (0, 1)), ("antisymmetric", (1, 2))))
    expr = indexed_tensor_expr("A", ("c", "b", "a"), variance_spec="lll")
    report = canonicalize_tensor_expr(expr, registry=reg)
    assert report.canonical.metadata["indices"] == ("a", "b", "c")
    assert report.canonical.metadata["slot_symmetry_sign"] == -1
    assert report.canonical.metadata["slot_symmetry_group_size"] > 2


def test_young_rule_routes_through_central_canonicalizer():
    expr1 = indexed_tensor_expr("Y", ("b", "a", "c"))
    expr2 = indexed_tensor_expr("Y", ("a", "b", "c"))
    rule = YoungSymmetryRule("Y", rows=((0, 1),), columns=((1, 2),))
    assert canonical_tensor_expr_key(expr1, young_rules=(rule,)) == canonical_tensor_expr_key(expr2, young_rules=(rule,))


def test_linear_multiterm_identity_reduces_additive_ir():
    a = indexed_tensor_expr("R", ("a", "b", "c", "d"))
    b = indexed_tensor_expr("R", ("b", "c", "a", "d"))
    c = indexed_tensor_expr("R", ("c", "a", "b", "d"))
    expr = ir_node("add", a, b, c)
    identity = LinearIdentityRule("first_bianchi_R", (LinearIdentityTerm(1, a), LinearIdentityTerm(1, b), LinearIdentityTerm(1, c)))
    report = canonicalize_tensor_expr(expr, identity_rules=(identity,))
    assert report.canonical.kind == "zero"
    assert any(step.rule == "identity_rules" for step in report.steps)



def test_tensorform_engine_exposes_central_tensor_expr_report():
    from tensoratlas.tensorform_engine import tensor_form_expr_canonical_report
    expr = indexed_tensor_expr("T", ("j", "i"))
    report = tensor_form_expr_canonical_report(expr)
    assert report is not None
    assert report.canonical.kind == "indexed_tensor"
