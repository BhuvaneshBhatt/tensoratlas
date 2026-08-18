# Architecture

`tensoratlas` is organized around several cooperating layers:

- a coordinate-calculus layer for charts, mappings, gradients, divergences, curls, Laplacians, and coordinate-fields
- a tensor layer for basis-aware tensor objects and structured tensor algebra
- an indexed layer for abstract-index canonicalization, rewriting, and TensorForm-based normalization
- a bounded algebraic scalar layer for conservative zero/equality reasoning and algebraic coefficient canonicalization

The package stays SymPy-adjacent: public functions return ordinary SymPy expressions, matrices, or small dataclasses whenever practical.

## Indexed subsystem split

The indexed layer is organized into dedicated façade modules that separate concerns at the package boundary:

- `indexed_expr.py` for indexed expression constructors and leaf types
- `indexed_spaces.py` for bundles and index spaces
- `indexed_patterns.py` for structural matching and rewrite patterns
- `indexed_render.py` for pretty-printing and tensor-normal-form rendering
- `indexed_normalization.py` for canonicalization and equality helpers
- `indexed_reconstruct.py` for deterministic reconstruction and diagnostics
- `indexed_api.py` for public entry points and reports

The `tensor_indices.py` module remains the deep implementation backend, while public imports flow through the narrower modules above.

## Scalar decision and algebraic-reduction split

Conservative scalar reasoning is implemented across:

- `symbolic_decision.py` for zero/equality decisions and warnings
- `basic_algebraic_reduce.py` for bounded algebraic canonicalization
- `simplification_core.py` and `simplification_policy.py` for shared simplification tiers

This split keeps user-facing decisions, bounded algebraic heuristics, and generic simplification logic separate.
