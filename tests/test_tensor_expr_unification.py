from tensoratlas.semantic_ir import (
    TensorExpr,
    TensorExprKind,
    IRProvenance,
    abstract_tensor_expr,
    canonical_ir_key,
    covariant_derivative_ir,
    curvature_ir,
    gamma_object_ir,
    indexed_tensor_expr,
    normalize_tensor_expr,
    rewrite_with_provenance,
    spin_object_ir,
    to_tensor_expr,
    variation_ir,
)
from tensoratlas.curvature_relations import Riemann, curvature_object_to_ir, curvature_reduce_to_ir
from tensoratlas.curvature_normal_forms import canonicalize_curvature_ir_node
from tensoratlas.variational_gr import metric, riemann, variation


def test_tensor_expr_is_canonical_shared_node_type():
    node = abstract_tensor_expr("T", rank=2, indices=("a", "b"))
    assert isinstance(node, TensorExpr)
    assert node.kind == TensorExprKind.ABSTRACT_TENSOR.value
    assert canonical_ir_key(node)[0] == "abstract_tensor"


def test_typed_nodes_cover_core_geometry_families():
    nodes = [
        indexed_tensor_expr("T", (("a", "up"),)),
        covariant_derivative_ir(abstract_tensor_expr("f", rank=0), index="a"),
        curvature_ir("Riemann", rank=4, dimension=4, indices=("a", "b", "c", "d")),
        spin_object_ir("psi", indices=("A",)),
        gamma_object_ir("gamma", indices=("a", "A", "B"), dimension=4),
        variation_ir(abstract_tensor_expr("g", rank=2), field="g"),
    ]
    assert [n.kind for n in nodes] == [
        "indexed_tensor",
        "covariant_derivative",
        "curvature",
        "spin_object",
        "gamma_object",
        "variation",
    ]


def test_curvature_adapter_returns_canonical_tensor_expr():
    ir = curvature_object_to_ir(Riemann(4))
    assert isinstance(ir, TensorExpr)
    assert ir.kind in {"curvature", "curvature_symbol"}
    assert ir.metadata["family"] == "Riemann"
    reduced = curvature_reduce_to_ir(Riemann(4))
    assert isinstance(reduced, TensorExpr)
    assert canonical_ir_key(reduced)


def test_variational_gr_constructors_use_shared_tensor_expr():
    expr = variation(metric("a", "b"), field="g")
    assert isinstance(expr, TensorExpr)
    assert expr.kind == "variation"
    assert expr.metadata["field"] == "g"
    assert expr.children[0].kind == "abstract_tensor"
    curv = riemann("a", "b", "c", "d")
    assert curv.kind == "curvature"


def test_canonical_keys_are_ir_derived_and_commutative():
    a = abstract_tensor_expr("A")
    b = abstract_tensor_expr("B")
    left = normalize_tensor_expr(TensorExpr("add", children=(a, b)))
    right = normalize_tensor_expr(TensorExpr("add", children=(b, a)))
    assert canonical_ir_key(left) == canonical_ir_key(right)


def test_rewrite_provenance_is_shared_model():
    before = curvature_ir("Ricci", rank=2, dimension=4)
    after = curvature_ir("Einstein", rank=2, dimension=4)
    rewritten = rewrite_with_provenance(before, after, rule="ricci_to_einstein", source="test")
    assert isinstance(rewritten.provenance, IRProvenance)
    assert rewritten.provenance.steps[-1].rule == "ricci_to_einstein"
    assert rewritten.provenance.steps[-1].before_key == canonical_ir_key(before)
    assert rewritten.provenance.steps[-1].after_key == canonical_ir_key(after)


def test_curvature_normal_form_canonicalization_preserves_tensor_expr_type():
    ir = curvature_ir("Riemann", rank=4, dimension=4)
    canon = canonicalize_curvature_ir_node(ir)
    assert isinstance(canon, TensorExpr)
    assert canon.kind == "curvature"
    assert canon.provenance.steps


def test_to_tensor_expr_is_main_adapter_entrypoint():
    adapted = to_tensor_expr(Riemann(3))
    assert adapted.kind == "curvature"
    assert adapted.metadata["dimension"] == 3
