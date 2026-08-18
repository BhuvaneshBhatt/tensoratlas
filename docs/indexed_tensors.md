# Indexed tensors and canonicalization

## Why an indexed layer?

Many symbolic tensor identities are best written in indexed notation rather than as explicit component arrays. The indexed layer in `tensoratlas` supports that style while remaining connected to concrete tensor objects and bases.

## Main indexed objects

The public API includes:

- `TensorIndex`
- `IndexedTensor`
- `IndexedTensorExpr`
- `indexed(...)`
- `indices(...)`

These helpers make it possible to construct expressions such as symbolic contractions, traces, and symmetry-constrained indexed formulas.

## Canonicalization and normalization

The package includes a substantial normalization pipeline. Important exported tools include:

- `normalize_indexed_expression(...)`
- `canonicalize_indexed_expression(...)`
- `canonical_indexed_form(...)`
- `to_indexed_tensor_form(...)`
- `from_indexed_tensor_form(...)`
- `indexed_equal(...)`
- `stronger_indexed_equal(...)`
- `alpha_rename_dummies(...)`

These routines aim to identify expressions that are equivalent up to:

- dummy-index renaming
- tensor symmetries
- contraction structure
- selected algebraic normalization steps

## What is TensorForm?

`TensorForm` is the package's canonical internal representation for indexed tensor expressions. It is used to normalize, compare, simplify, and reconstruct indexed expressions in a way that is stable under dummy-index renaming, factor reordering, basis normalization, and selected special-tensor identities.

A typical pipeline is:

1. Parse a user-facing indexed expression such as an `IndexedTensor` or `IndexedTensorExpr`.
2. Convert it to `IndexedTensorForm` with `to_indexed_tensor_form(...)`.
3. Reduce and sort the resulting canonical terms.
4. Reconstruct an indexed expression when a boundary representation is needed.

## TensorForm structure

The main structured objects are:

- `TensorFormTerm`
- `IndexedTensorForm`

An `IndexedTensorForm` stores a tuple of `TensorFormTerm` instances. Each `TensorFormTerm` has four fields:

- `scalar`: the scalar coefficient
- `factors`: the canonical tensor-factor tuple for that term
- `free_signature`: the canonical description of the free indices
- `bundle_signature`: the canonical description of the bundles or index spaces involved

Conceptually, a TensorForm stores a symbolic sum

$$
\mathrm{TensorForm} = \sum_k c_k\,F_{k,1}F_{k,2}\cdots F_{k,m_k},
$$

but it stores that sum as structured canonical data rather than as an unconstrained expression tree. This makes equality and canonicalization more robust.

## Why TensorForm matters

TensorForm is designed so that expressions that differ only by superficial syntax still normalize to the same internal representation. For example,

$$
A^i{}_{j} B^j{}_{k}
\quad	ext{and}\quad
A^i{}_{m} B^m{}_{k}
$$

have different dummy names but the same contraction pattern. TensorForm removes that superficial difference by canonicalizing the dummy structure and factor data.

This is also the layer where the package can apply selected reductions involving metric tensors, Kronecker deltas, epsilon tensors, and related special indexed objects.

## Diagnostics and reports

The indexed layer also exposes diagnostics and reports, including normalization and helper-audit data. This is helpful during development of rewrite rules or when debugging why two tensor expressions do or do not canonicalize to the same TensorForm.

## Rewriting and pattern matching

The package exports rewrite machinery such as:

- `IndexedRewriteRule`
- `IndexedRewriteEngine`
- `rewrite_fixed_point(...)`
- `tensor_replace(...)`
- pattern objects for indexed matching

This makes it possible to encode symbolic tensor identities in a reusable rule-based style.

## Typical use case

A common workflow is:

1. Construct basis-aware tensors.
2. Build indexed expressions.
3. Canonicalize or normalize them.
4. Compare expressions modulo dummy-index relabeling and symmetry.
5. Apply rewrite rules until a desired TensorForm is reached.

## Related notebook sections

- [Section 12: Indexed notation](../notebooks/tensoratlas_demo.ipynb#12-indexed-notation)
- [Section 21: Young symmetrizers and irreducible tensor symmetries](../notebooks/tensoratlas_demo.ipynb#21-young-symmetrizers-and-irreducible-tensor-symmetries)
- [Section 33: Symmetry-aware canonicalization and tensor projectors](../notebooks/tensoratlas_demo.ipynb#33-symmetry-aware-canonicalization-and-tensor-projectors)
