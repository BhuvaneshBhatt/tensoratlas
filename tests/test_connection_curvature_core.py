from tensoratlas.connection_curvature import (
    canonicalize_geometry_ir,
    connection_profile,
    convert_curvature_convention_ir,
    covariant_derivative_commutator_ir,
    curvature_decomposition_ir,
    einstein_tensor_expr,
    ensure_connection_curvature_declarations,
    first_bianchi_identity_ir,
    nonmetricity_identity_ir,
    reduce_connection_curvature_identities,
    ricci_tensor_expr,
    riemann_tensor_expr,
    scalar_curvature_tensor_expr,
    schouten_tensor_expr,
    second_bianchi_identity_ir,
    torsion_identity_ir,
    torsion_tensor_expr,
    weyl_tensor_expr,
)
from tensoratlas.declarations import CurvatureConventionPolicy, declaration_registry, standard_riemannian_registry
from tensoratlas.semantic_ir import ir_node


def test_levis_civita_profile_and_curvature_declarations():
    reg = ensure_connection_curvature_declarations(standard_riemannian_registry("M", 4), "CD")
    profile = connection_profile(reg, "CD")
    assert profile.kind == "levi_civita"
    assert profile.torsion_free is True
    assert profile.metric_compatible is True
    assert "Riemann[CD]" in reg.tensors
    assert "Ricci[CD]" in reg.tensors
    assert "Weyl[CD]" in reg.tensors
    assert "Schouten[CD]" in reg.tensors
    assert "Einstein[CD]" in reg.tensors
    assert "Torsion[CD]" in reg.tensors
    assert "Nonmetricity[CD]" in reg.tensors


def test_affine_torsionful_and_nonmetric_connection_profiles():
    reg = declaration_registry().declare_manifold("M", 4).declare_bundle("TM", "M")
    reg = reg.declare_metric("g", "M", "TM")
    reg = reg.declare_connection("A", "TM", metric="g", torsion_free=False, metric_compatible=False)
    reg = ensure_connection_curvature_declarations(reg, "A")
    profile = connection_profile(reg, "A")
    assert profile.kind == "torsionful_nonmetric_affine"
    assert profile.has_torsion is True
    assert profile.has_nonmetricity is True


def test_curvature_tensor_constructors_share_tensor_expr_registry_data():
    reg = ensure_connection_curvature_declarations(standard_riemannian_registry("M", 4), "CD")
    R = riemann_tensor_expr(reg, "CD", ("a", "b", "c", "d"))
    Ric = ricci_tensor_expr(reg, "CD", ("a", "b"))
    Scal = scalar_curvature_tensor_expr(reg, "CD")
    W = weyl_tensor_expr(reg, "CD", ("a", "b", "c", "d"))
    P = schouten_tensor_expr(reg, "CD", ("a", "b"))
    G = einstein_tensor_expr(reg, "CD", ("a", "b"))
    assert R.kind == "indexed_tensor" and R.metadata["family"] == "Riemann"
    assert Ric.metadata["family"] == "Ricci"
    assert Scal.kind == "curvature_scalar"
    assert W.metadata["family"] == "Weyl"
    assert P.metadata["family"] == "Schouten"
    assert G.metadata["family"] == "Einstein"


def test_first_and_second_bianchi_reduce_to_zero_for_levis_civita_path():
    reg = ensure_connection_curvature_declarations(standard_riemannian_registry("M", 4), "CD")
    first = canonicalize_geometry_ir(first_bianchi_identity_ir(reg, "CD", ("a", "b", "c", "d")), reg)
    second = canonicalize_geometry_ir(second_bianchi_identity_ir(reg, "CD", ("e", "a", "b", "c", "d")), reg)
    assert first.canonical.kind == "zero"
    assert "first_bianchi" in first.applied_identities
    assert second.canonical.kind == "zero"
    assert "second_bianchi" in second.applied_identities


def test_torsion_and_nonmetricity_identities_reduce_for_levis_civita():
    reg = ensure_connection_curvature_declarations(standard_riemannian_registry("M", 4), "CD")
    torsion = canonicalize_geometry_ir(torsion_identity_ir(reg, "CD", ("a", "b", "c")), reg)
    nonmetricity = canonicalize_geometry_ir(nonmetricity_identity_ir(reg, "CD", ("a", "b", "c")), reg)
    assert torsion.canonical.kind == "zero"
    assert nonmetricity.canonical.kind == "zero"


def test_torsion_survives_for_torsionful_connection():
    reg = declaration_registry().declare_manifold("M", 4).declare_bundle("TM", "M")
    reg = reg.declare_connection("A", "TM", torsion_free=False)
    reg = ensure_connection_curvature_declarations(reg, "A")
    reduced, identities = reduce_connection_curvature_identities(torsion_tensor_expr(reg, "A", ("a", "b", "c")), reg)
    assert reduced.kind == "indexed_tensor"
    assert "torsion_free_connection" not in identities


def test_covariant_derivative_commutator_expands_with_curvature_and_torsion_terms():
    reg = declaration_registry().declare_manifold("M", 4).declare_bundle("TM", "M")
    reg = reg.declare_connection("A", "TM", torsion_free=False)
    reg = ensure_connection_curvature_declarations(reg, "A")
    operand = ir_node("symbol", payload="V")
    comm = covariant_derivative_commutator_ir(reg, "A", "a", "b", operand)
    reduced, identities = reduce_connection_curvature_identities(comm, reg)
    assert reduced.kind == "add"
    assert "covariant_derivative_commutator" in identities
    assert any(child.kind == "curvature_action" for child in reduced.children)
    assert any(child.kind == "torsion_commutator_term" for child in reduced.children)


def test_curvature_decomposition_is_dimension_aware_and_weyl_zero_below_four():
    reg3 = declaration_registry().declare_manifold("N", 3).declare_bundle("TN", "N")
    reg3 = reg3.declare_connection("D", "TN", torsion_free=True)
    reg3 = ensure_connection_curvature_declarations(reg3, "D")
    decomp = curvature_decomposition_ir(reg3, "D", ("a", "b", "c", "d"))
    assert decomp.metadata["target"] == "SchoutenMetricPart"
    zero_weyl = canonicalize_geometry_ir(weyl_tensor_expr(reg3, "D", ("a", "b", "c", "d")), reg3)
    assert zero_weyl.canonical.kind == "zero"


def test_convention_conversion_applies_sign_and_slot_order():
    source = CurvatureConventionPolicy(name="source", commutator_sign=1, riemann_slot_order=("up", "down1", "down2", "down3"))
    target = CurvatureConventionPolicy(name="target", commutator_sign=-1, riemann_slot_order=("down1", "up", "down2", "down3"))
    reg = declaration_registry().declare_manifold("M", 4).declare_bundle("TM", "M")
    reg = reg.declare_curvature_policy(source).declare_curvature_policy(target)
    reg = reg.declare_connection("D", "TM", curvature_policy="source")
    reg = ensure_connection_curvature_declarations(reg, "D")
    R = riemann_tensor_expr(reg, "D", ("a", "b", "c", "d"))
    report = convert_curvature_convention_ir(R, source=source, target=target)
    assert report.sign_factor == -1
    assert report.index_permutation == (1, 0, 2, 3)
    assert report.converted.metadata["indices"] == ("b", "a", "c", "d")
    assert report.converted.metadata["convention"] == "target"
