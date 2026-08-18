# Tensor algebra and indices

## Tensor objects and tensor expressions

The package defines practical tensor objects and composite tensor expressions through:

- `TensorObject`
- `TensorExpr`
- `TensorBasis`
- `TensorFrame`

These let you represent tensors with explicit basis information, combine them by tensor product, and manipulate slots in a structured way.

## Basis helpers

Important basis constructors include:

- `tangent_basis(chart)`
- `cotangent_basis(chart)`
- `orthonormal_tangent_basis(chart)`
- `orthonormal_cotangent_basis(chart)`
- `dual_basis(basis)`
- `basis_transformation_matrix(source, target)`
- `transformed_basis(basis, mapping)`

## Canonical tensor constructors

Useful symbolic tensor constructors include:

- `identity_tensor(...)`
- `kronecker_delta_tensor(...)`
- `metric_tensor(...)`
- `permutation_tensor(...)`
- `volume_form(...)`

## Indexed expressions

For abstract-index style workflows, the package exports:

- `TensorIndex`
- `IndexedTensor`
- `IndexedTensorExpr`
- `indexed(...)`
- `indices(...)`

This layer is useful when you want symbolic reasoning about slots and contractions without manually writing all component expressions.

## Normalization and equality

The package includes normalization and comparison helpers such as:

- `normalize_indexed_expression(...)`
- `to_indexed_tensor_form(...)`
- `from_indexed_tensor_form(...)`
- `indexed_equal(...)`
- `stronger_indexed_equal(...)`
- `indexed_signature(...)`

These are intended to make algebraic comparison more robust in the presence of renamed dummy indices, symmetry data, and selected special tensors.

## Rewriting and patterns

The index layer also includes pattern/rewrite support:

- `IndexedRewriteRule`
- `IndexedRewriteEngine`
- `rewrite_fixed_point(...)`
- `TensorPattern`
- `ExprPattern`
- `PatternRewriteRule`
- `rewrite_with_patterns(...)`

## Symmetry tools

The tensor core and index system include symmetry-aware tools such as:

- slot symmetrization and antisymmetrization,
- projectors,
- Young-projector-related helpers,
- dummy-index alpha renaming,
- block/direct-sum tensor constructors,
- contraction planning helpers.
