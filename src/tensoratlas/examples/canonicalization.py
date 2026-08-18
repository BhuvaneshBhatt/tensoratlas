"""Canonicalization tutorial examples."""

from __future__ import annotations

from tensoratlas.semantic_ir import canonical_ir_key, ir_node
from tensoratlas.tensor_expr_canonicalization import canonicalize_tensor_expr


def canonicalization_workflow() -> dict[str, object]:
    """Show canonical keys for two structurally equal symbolic tensor expressions."""
    expr = ir_node("tensor:product", children=(ir_node("T", payload=("a", "b")), ir_node("S", payload=("b",))))
    result = canonicalize_tensor_expr(expr)
    return {"expression": expr, "canonical": result.canonical, "canonical_key": canonical_ir_key(result.canonical)}
