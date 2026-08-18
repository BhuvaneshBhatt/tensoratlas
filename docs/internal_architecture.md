# Internal architecture

TensorAtlas currently has several layers.  New development should prefer the `tensoratlas.core` APIs and avoid adding deprecated aliases.

## Coordinate/component layer

- `coordinate_tools.py`: coordinate domains, singularities, standard catalog entries, coordinate maps.
- `coordinate_map_data.py`: inverse branches, Jacobians, transition maps, and richer transform properties.
- `transform_fields.py`: scalar, vector, covector, tensor, and density transformations.
- `vector_calculus.py`: coordinate-basis vector calculus with convention-aware result wrappers.
- `components.py`: coordinate systems, bases, coframes, component tensors, metric geometry, curvature components, vielbeins, and spin connections.

## Abstract tensor layer

- `manifolds.py`, `indices.py`, `tensor_heads.py`, `symmetries.py`, `tensor_expr.py`: semantic tensor IR.
- `tensor_monomial_encoding.py`: conversion from semantic tensor terms to permutation-canonicalization problems.
- `permutation_group_backend.py`: reference signed-permutation backend and backend protocol.
- `multiterm_identities.py`: sparse linear identities and bounded multiterm reductions.

## Shared utilities

`symbolic_utils.py` centralizes common SymPy-dependent helpers such as matrix conversion, nested component construction, bounded simplification, zero checks, and variance normalization.  New component code should use these helpers instead of creating local copies.

## Retained older modules

The repository still contains advanced internal modules because the test suite exercises them.  They should not be extended unless needed for a specific test-supported workflow.  New features should be implemented in `tensoratlas.core` and exposed through narrow public APIs.
