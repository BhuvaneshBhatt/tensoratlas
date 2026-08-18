import sympy as sp

from tensoratlas.declarations import (
    DeclarationError,
    CurvatureConventionPolicy,
    TensorSymmetryDeclaration,
    declaration_registry,
    standard_riemannian_registry,
)
from tensoratlas.semantic_ir import TensorExpr, canonical_ir_key, to_tensor_expr


def test_declarative_registry_builds_geometry_environment():
    x, y = sp.symbols("x y")
    reg = declaration_registry().declare_manifold("M", 2, signature=(2, 0), orientation="positive")
    reg = reg.declare_chart("cart", "M", (x, y), domain="R2")
    reg = reg.declare_bundle("TM", "M").declare_bundle("T*M", "M", kind="covector", dual_of="TM")
    reg = reg.declare_index_family("latin", "TM", ("a", "b", "c"), dummy_prefix="d")
    reg = reg.declare_dummy_pool("dummies", "latin")

    assert reg.manifolds["M"].dimension == 2
    assert reg.charts["cart"].coordinates == (x, y)
    assert reg.bundles["T*M"].dual_of == "TM"
    assert reg.index_families["latin"].dummy_symbol(3) == "d3"


def test_dummy_pool_allocation_is_immutable():
    reg = standard_riemannian_registry("M", 3)
    reg2, names = reg.allocate_dummy_indices("latin_dummies", 3)

    assert names == ("d0", "d1", "d2")
    assert reg.dummy_pools["latin_dummies"].counter == 0
    assert reg2.dummy_pools["latin_dummies"].counter == 3


def test_tensor_metric_connection_and_commutation_declarations():
    reg = declaration_registry().declare_manifold("M", 4, signature=(-1, 1, 1, 1))
    reg = reg.declare_bundle("TM", "M")
    reg = reg.declare_tensor(
        "F",
        ("TM", "TM"),
        ("l", "l"),
        symmetries=(TensorSymmetryDeclaration("antisymmetric", (0, 1)),),
        density_weight=0,
        dependencies=("A",),
    )
    reg = reg.declare_metric("g", "M", "TM", inverse_name="g_inv")
    reg = reg.declare_connection("CD", "TM", metric="g", torsion_free=True, metric_compatible=True)
    reg = reg.declare_commutation_rule("comm", "CD", applies_to="F")

    assert reg.tensors["F"].variance_string() == "ll"
    assert reg.metrics["g"].to_tensor_declaration().symmetries[0].kind == "symmetric"
    assert reg.connections["CD"].is_torsion_free()
    assert reg.connections["CD"].is_metric_compatible()
    assert reg.commutation_rules["comm"].torsion_term is False
    assert reg.commutation_rules["comm"].nonmetricity_term is False


def test_curvature_policy_controls_commutator_sign():
    reg = declaration_registry().declare_manifold("M", 2).declare_bundle("TM", "M")
    reg = reg.declare_curvature_policy(CurvatureConventionPolicy("wald_like", sign="wald", commutator_sign=-1))
    reg = reg.declare_connection("nabla", "TM", curvature_policy="wald_like", torsion_free=False)
    reg = reg.declare_commutation_rule("wald_comm", "nabla")

    assert reg.commutation_rules["wald_comm"].sign == -1
    assert reg.commutation_rules["wald_comm"].torsion_term is True


def test_registry_exports_tensor_expr_and_canonical_keys():
    reg = standard_riemannian_registry("M", 4, signature=(-1, 1, 1, 1))
    tensor = reg.tensor_expr("g", ("a", "b"))
    connection = reg.connection_ir("CD")
    comm = reg.commutator_ir("riemann_commutator", "a", "b", tensor)
    registry_ir = reg.to_ir()

    assert isinstance(tensor, TensorExpr)
    assert tensor.kind == "indexed_tensor"
    assert tensor.metadata["variance_spec"] == "ll"
    assert connection.kind == "declaration:connection"
    assert comm.kind == "covariant_derivative_commutator"
    assert registry_ir.kind == "declaration:registry"
    assert canonical_ir_key(registry_ir) == canonical_ir_key(to_tensor_expr(reg))


def test_declaration_validation_rejects_inconsistent_references():
    reg = declaration_registry().declare_manifold("M", 2)
    try:
        reg.declare_chart("bad", "M", (sp.Symbol("x"),))
    except DeclarationError as exc:
        assert "coordinate count" in str(exc)
    else:
        raise AssertionError("Expected DeclarationError")

    try:
        reg.declare_tensor("T", ("missing",), ("u",))
    except DeclarationError as exc:
        assert "Unknown bundle" in str(exc)
    else:
        raise AssertionError("Expected DeclarationError")
