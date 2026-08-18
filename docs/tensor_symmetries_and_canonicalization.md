# Tensor symmetries and canonicalization

TensorAtlas has a semantic tensor-expression layer and a reference permutation-group backend for monoterm canonicalization.

## Semantic objects

The core objects are:

- `Manifold`
- `IndexType`
- `AbstractIndex`
- `TensorHead`
- `SlotSymmetry`
- `TensorFactor`
- `TensorTerm`
- `TensorExpr`

Tensor heads can declare symmetry, variance, commutativity, and parity.  Slot symmetries may be built-in, such as symmetric, antisymmetric, Riemann/Weyl-style pair symmetries, or custom signed generators.

## Canonicalization workflow

The public wrapper is:

```python
canonicalize_tensor(expr, explain=True)
```

For one monomial, TensorAtlas:

1. validates strict Einstein index use;
2. applies conservative common-case zero tests and total-symmetry shortcuts;
3. encodes slots, labels, slot symmetries, repeated factor exchanges, and dummy-renaming groups;
4. calls a `CanonicalizationBackend`;
5. decodes the canonical permutation result back into a semantic tensor term.

The current backend is pure Python and reference-oriented.  It uses guarded explicit closure for double-coset oracle behavior.  It is correct for small and medium tests but is not intended as the final high-performance xPerm-class backend.

## Backend protocol

Future native code should implement:

```python
class CanonicalizationBackend:
    def schreier_sims(...): ...
    def canonicalize_double_coset(...): ...
```

The recommended native target is Rust + `pyo3`.  The Python backend remains the correctness oracle for randomized small-group tests.

## Strict index validation

Tensor monomial encoding currently uses strict Einstein validation.  A repeated index name in one index family may occur at most twice, and a dummy pair must have one up and one down occurrence unless the expression is a recognized forced-zero antisymmetric case.
