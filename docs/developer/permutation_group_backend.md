# Permutation group backend

The permutation backend lives in `tensoratlas.core.permutation_group_backend`.

## Public reference types

- `Permutation`
- `SignedPermutation`
- `PermutationGroup`
- `StabilizerChain`
- `CanonicalDoubleCosetResult`
- `CanonicalizationBackend`
- `PythonPermutationBackend`

## Reference/oracle functions

- `canonical_double_coset_reference(...)`
- `brute_force_double_coset(...)`

These functions deliberately enumerate explicit group closures under a closure-size guard.  They are used as correctness oracles and for small tests.  They are not production-fast xPerm replacements.

## Backend protocol

Backend implementations should provide:

```python
def schreier_sims(group, base=None): ...
def canonicalize_double_coset(left_group, representative, right_group, *, labels=None, base=None): ...
```

The default backend is `PythonPermutationBackend`, which routes double-coset canonicalization to the reference oracle.  An optional native backend can implement the same protocol.

## Native backend target

The intended fast backend target is Rust + `pyo3`:

- packed integer-array permutations;
- signed generator support;
- Schreier-Sims stabilizer chains without explicit full closure;
- Butler-Portugal-style double-coset canonicalization;
- base-selection heuristics supplied by the tensor encoder;
- Python tests that compare native results with the reference oracle for small groups.

## Closure guards

Explicit group closure is guarded by `DEFAULT_CLOSURE_LIMIT`.  If closure exceeds that limit, TensorAtlas raises `TensorKernelError` instead of silently hanging.
