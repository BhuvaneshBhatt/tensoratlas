
from tensoratlas.variational_gr import *

def test_recursive_ibp_walks_tree_and_moves_derivative():
    expr = add(
        covariant_derivative(mul(symbol("A"), variation(symbol("B"))), index="m"),
        symbol("physical")
    )
    out = integrate_by_parts_recursive(expr)
    assert out.kind in {"add", "neg", "mul"}
    assert "physical" in str(ir_to_dict(out))

def test_strict_divergence_recognition():
    good = covariant_derivative(node("vector", indices=("a",)), index="a")
    bad = covariant_derivative(node("tensor", indices=("a","b")), index="a")
    assert is_total_divergence(good) is True
    assert is_total_divergence(bad) is False

def test_boundary_elimination_preserves_physical_terms():
    expr = add(
        covariant_derivative(node("vector", indices=("a",)), index="a"),
        symbol("physical")
    )
    out = eliminate_boundary_terms(expr)
    assert out.kind == "symbol"

def test_delta_r_expansion_chain_exists():
    dconn = variation_connection_from_metric()
    dRiem = variation_riemann_from_connection()
    dRic = variation_ricci_from_riemann()
    dR = variation_scalar_curvature_full()
    assert dconn.kind == "mul"
    assert dRiem.kind == "add"
    assert dRic.kind == "contract"
    assert dR.kind == "add"

def test_canonical_contraction_reduction_emerges_einstein():
    rep = derive_einstein_tensor_from_algebra()
    assert "after_canonical_contraction_reduction" in rep
    assert rep["einstein_tensor_factorization"].kind == "mul"
    kinds = [c.kind for c in rep["einstein_tensor_factorization"].children]
    assert "sqrt_det_metric" in kinds
    assert "delta_metric" in kinds
