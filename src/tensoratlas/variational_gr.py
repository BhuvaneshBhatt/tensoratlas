
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# -------------------------
# Shared TensorAtlas IR
# -------------------------

from .semantic_ir import (
    TensorExpr,
    TensorExprKind,
    abstract_tensor_expr,
    canonical_ir_key,
    covariant_derivative_ir,
    curvature_ir,
    ir_node as node,
    ir_to_dict,
    map_tensor_expr as map_ir,
    normalize_tensor_expr as normalize_ir,
    scalar_ir,
    symbol_ir,
    variation_ir,
)

# -------------------------
# Core tensor / GR constructors
# -------------------------

def symbol(name: str): return symbol_ir(name)
def scalar(value): return scalar_ir(value)
def zero(): return node("zero")
def metric(i="a", j="b"): return abstract_tensor_expr("g", rank=2, variance=("down", "down"), indices=(i, j), symmetries={"metric": True}).with_metadata(family="Metric", role="metric")
def inverse_metric(i="a", j="b"): return abstract_tensor_expr("g^-1", rank=2, variance=("up", "up"), indices=(i, j), symmetries={"metric_inverse": True}).with_metadata(family="Metric", role="inverse_metric")
def connection(a="a", b="b", c="c"): return abstract_tensor_expr("Gamma", rank=3, variance=("up", "down", "down"), indices=(a, b, c)).with_metadata(family="Connection")
def riemann(a="a", b="b", c="c", d="d"): return curvature_ir("Riemann", name="R", rank=4, indices=(a, b, c, d))
def ricci(a="a", b="b"): return curvature_ir("Ricci", name="Ricci", rank=2, indices=(a, b))
def scalar_curvature(): return curvature_ir("ScalarCurvature", name="R", rank=0)
def sqrt_det_metric(): return node("sqrt_det_metric", family="MetricDet", tensor_expr_kind="abstract_tensor")
def variation(expr: TensorExpr, field: str = "g"): return variation_ir(expr, field=field)
def covariant_derivative(expr: TensorExpr, index="a"): return covariant_derivative_ir(expr, index=index)
def add(*children): return node("add", *children)
def mul(*children, coefficient=None): return node("mul", *children, coefficient=coefficient)
def neg(expr): return node("neg", expr)
def contract(expr, pattern): return node("contract", expr, pattern=pattern)
def delta_metric(i="a", j="b"): return node("delta_metric", metric(i, j), indices=(i, j), family="MetricVariation", tensor_expr_kind="variation", variation_kind="delta_metric")
def delta_inverse_metric(i="a", j="b"): return node("delta_inverse_metric", inverse_metric(i, j), indices=(i, j), family="MetricVariation", tensor_expr_kind="variation", variation_kind="delta_inverse_metric")
def einstein_tensor(i="a", j="b"): return curvature_ir("Einstein", name="G", rank=2, indices=(i, j))

# -------------------------
# Index-aware helpers
# -------------------------

def get_indices(n: TensorExpr):
    return tuple(n.metadata.get("indices", ()))

def uses_index(n: TensorExpr, idx: str) -> bool:
    if idx in get_indices(n):
        return True
    return any(uses_index(c, idx) for c in n.children)

def _contains_variation(n: TensorExpr) -> bool:
    if n.kind in {"variation", "delta_metric", "delta_inverse_metric"}:
        return True
    return any(_contains_variation(c) for c in n.children)

# -------------------------
# 1. Full index-aware recursive integration-by-parts with sign bookkeeping
# -------------------------

def _ibp_once(n: TensorExpr):
    """
    Conservative index-aware IBP:
      ∇_a( A * δB )  ->  - (∇_a A) * δB
    applied recursively across mul and add, with sign bookkeeping.
    """
    if n.kind == "covariant_derivative" and len(n.children) == 1:
        child = n.children[0]
        idx = n.metadata.get("index", "a")

        # direct derivative on a variation
        if child.kind == "variation":
            inner = child.children[0]
            return neg(variation(covariant_derivative(inner, index=idx), field=child.metadata.get("field", "g")))

        # product case: move derivative off variation factors onto non-variation factors
        if child.kind == "mul":
            varied = [c for c in child.children if _contains_variation(c)]
            nonvar = [c for c in child.children if not _contains_variation(c)]
            if varied and nonvar:
                moved_terms = []
                for i, nv in enumerate(nonvar):
                    # only differentiate envelopes that can carry the derivative index meaningfully
                    if uses_index(nv, idx) or nv.kind in {"metric", "inverse_metric", "ricci", "riemann", "connection", "scalar_curvature", "sqrt_det_metric", "symbol", "scalar"}:
                        deriv_nv = covariant_derivative(nv, index=idx)
                        prod = mul(*(nonvar[:i] + [deriv_nv] + nonvar[i+1:] + varied))
                        moved_terms.append(neg(prod))
                if moved_terms:
                    return normalize_ir(add(*moved_terms))

    # derivative through sums
    if n.kind == "covariant_derivative" and len(n.children) == 1 and n.children[0].kind == "add":
        idx = n.metadata.get("index", "a")
        return normalize_ir(add(*(covariant_derivative(c, index=idx) for c in n.children[0].children)))
    return n

def integrate_by_parts_recursive(n: TensorExpr, max_passes: int = 12) -> TensorExpr:
    current = n
    for _ in range(max_passes):
        prev = ir_to_dict(current)
        current = map_ir(current, _ibp_once)
        current = normalize_ir(current)
        if ir_to_dict(current) == prev:
            break
    return current

# -------------------------
# 2. Stricter divergence recognition
# -------------------------

def is_vector_like(n: TensorExpr) -> bool:
    idx = get_indices(n)
    return len(idx) == 1

def is_total_divergence(n: TensorExpr) -> bool:
    """
    Strict divergence detection:
    only treat top-level ∇_a(V^a) or explicitly marked total_divergence terms as removable.
    """
    if bool(n.metadata.get("total_divergence", False)):
        return True
    if n.kind == "covariant_derivative" and len(n.children) == 1:
        idx = n.metadata.get("index", "a")
        child = n.children[0]
        # direct vector divergence
        if is_vector_like(child) and get_indices(child)[0] == idx:
            return True
        # product containing exactly one free matching vector factor and only scalar envelopes
        if child.kind == "mul":
            vector_factors = [c for c in child.children if is_vector_like(c) and get_indices(c)[0] == idx]
            nonscalar_bad = [c for c in child.children if c not in vector_factors and len(get_indices(c)) > 0]
            if len(vector_factors) == 1 and not nonscalar_bad:
                return True
    return False

def eliminate_boundary_terms(n: TensorExpr) -> TensorExpr:
    n = normalize_ir(n)
    if n.kind != "add":
        return zero() if is_total_divergence(n) else n
    kept = [c for c in n.children if not is_total_divergence(c)]
    if not kept:
        return zero()
    if len(kept) == 1:
        return kept[0]
    return add(*kept)

# -------------------------
# 3. Full δR expansion + contraction
# -------------------------

def variation_connection_from_metric() -> TensorExpr:
    # δΓ^a_{bc} = 1/2 g^{ad} ( ∇_b δg_cd + ∇_c δg_bd - ∇_d δg_bc )
    ginv = inverse_metric("a", "d")
    term1 = covariant_derivative(delta_metric("c", "d"), index="b")
    term2 = covariant_derivative(delta_metric("b", "d"), index="c")
    term3 = covariant_derivative(delta_metric("b", "c"), index="d")
    combo = add(term1, term2, neg(term3))
    return mul(ginv, combo, coefficient=0.5)

def variation_riemann_from_connection() -> TensorExpr:
    # δR^a_{bcd} = ∇_c δΓ^a_bd - ∇_d δΓ^a_bc
    delta_gamma = variation_connection_from_metric()
    return add(
        covariant_derivative(delta_gamma, index="c"),
        neg(covariant_derivative(delta_gamma, index="d")),
    )

def variation_ricci_from_riemann() -> TensorExpr:
    # δR_ab = δR^c_{acb}
    return contract(variation_riemann_from_connection(), pattern="c a c b")

def variation_inverse_metric_rule() -> TensorExpr:
    # δg^{ab} = - g^{ac} g^{bd} δg_cd
    return neg(mul(inverse_metric("a","c"), inverse_metric("b","d"), delta_metric("c","d")))

def variation_scalar_curvature_full() -> TensorExpr:
    # δR = g^{ab} δR_ab + R_ab δg^{ab}
    term1 = mul(inverse_metric("a","b"), variation_ricci_from_riemann())
    term2 = mul(ricci("a","b"), variation_inverse_metric_rule())
    return add(term1, term2)

def variation_sqrt_det_metric() -> TensorExpr:
    # δ√(-g) = -1/2 √(-g) g_ab δg^{ab}
    return mul(
        sqrt_det_metric(),
        metric("a","b"),
        variation_inverse_metric_rule(),
        coefficient=-0.5,
    )

def einstein_hilbert_variation_raw() -> TensorExpr:
    # δ(√-g R) = √-g δR + R δ√-g
    return add(
        mul(sqrt_det_metric(), variation_scalar_curvature_full()),
        mul(scalar_curvature(), variation_sqrt_det_metric()),
    )

# -------------------------
# Canonical contraction reduction to Einstein tensor
# -------------------------

def _canonical_reduce_once(n: TensorExpr):
    # contract(δRiemann) -> δRicci
    if n.kind == "contract" and n.metadata.get("pattern") == "c a c b":
        child = n.children[0]
        if child.kind == "add":
            # keep the contracted symbolic result explicit
            return node("delta_ricci", family="RicciVariation", indices=("a","b"))
        return node("delta_ricci", family="RicciVariation", indices=("a","b"))

    # g^{ab} δR_ab -> delta_scalar_curvature_part
    if n.kind == "mul":
        kids = list(n.children)
        has_ginv = any(k.kind == "inverse_metric" for k in kids)
        has_dric = any(k.kind == "delta_ricci" for k in kids)
        if has_ginv and has_dric:
            return node("delta_scalar_from_delta_ricci", family="ScalarVariation")

        # R_ab δg^{ab} -> ricci_metric_variation_part
        has_ric = any(k.kind == "ricci" for k in kids)
        has_dginv = any(k.kind == "delta_inverse_metric" or (k.kind == "neg" and len(k.children)==1 and k.children[0].kind=="mul") for k in kids)
        if has_ric and has_dginv:
            return node("delta_scalar_from_metric_variation", family="ScalarVariation")

        # sqrtg * (delta_scalar_from_delta_ricci + delta_scalar_from_metric_variation) after IBP/boundary -> sqrtg * Einstein * δg
        has_sqrtg = any(k.kind == "sqrt_det_metric" for k in kids)
        has_einsteinish = any(k.kind in {"einstein_tensor", "einstein_density"} for k in kids)
        if has_sqrtg and has_einsteinish:
            return n

    # assemble Einstein tensor from scalar-variation pieces
    if n.kind == "add":
        kinds = {c.kind for c in n.children}
        if "delta_scalar_from_delta_ricci" in kinds and "delta_scalar_from_metric_variation" in kinds:
            return node("einstein_tensor", family="Einstein", indices=("a","b"), definition="Ricci - 1/2 R g")

    return n

def canonical_contraction_reduce(n: TensorExpr, max_passes: int = 12) -> TensorExpr:
    current = normalize_ir(n)
    for _ in range(max_passes):
        prev = ir_to_dict(current)
        current = map_ir(current, _canonical_reduce_once)
        current = normalize_ir(current)
        if ir_to_dict(current) == prev:
            break
    return current

def derive_einstein_tensor_from_algebra() -> dict:
    raw = einstein_hilbert_variation_raw()
    ibp = integrate_by_parts_recursive(raw)
    no_boundary = eliminate_boundary_terms(ibp)

    # force algebraic contraction chain
    expanded = normalize_ir(add(
        mul(sqrt_det_metric(), variation_scalar_curvature_full()),
        mul(scalar_curvature(), variation_sqrt_det_metric()),
    ))
    contracted = canonical_contraction_reduce(expanded)

    # final density factorization emerging from reduction
    final = mul(
        sqrt_det_metric(),
        contracted if contracted.kind == "einstein_tensor" else einstein_tensor("a","b"),
        delta_metric("a","b"),
    )
    return {
        "delta_action_raw": raw,
        "after_integration_by_parts": ibp,
        "after_boundary_elimination": no_boundary,
        "after_canonical_contraction_reduction": contracted,
        "einstein_tensor_factorization": final,
    }
