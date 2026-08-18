# Tensor core

## Motivation

Coordinate components alone are often not enough. In symbolic tensor algebra, it is useful to attach basis information, variance, and symmetry metadata directly to tensor objects. The tensor core provides that layer.

## Main objects

The tensor core revolves around objects such as:

- `TensorBasis`
- `IndexBundle`
- `TensorFrame`
- `TensorObject`
- `TensorExpr`

## Bases and frames

The package exports convenient constructors and helpers:

- `tangent_basis(chart)`
- `cotangent_basis(chart)`
- `orthonormal_tangent_basis(chart)`
- `orthonormal_cotangent_basis(chart)`
- `dual_basis(basis)`
- `basis_dimension(basis)`
- `basis_transformation_matrix(source, target)`
- `transformed_basis(basis, mapping)`

Moving between coordinate and orthonormal bases is particularly useful for orthogonal charts.

## TensorObject

A `TensorObject` is a basis-aware tensor with component data and slot metadata. The tensor core supports operations such as:

- tensor products
- contractions
- slot permutation
- symmetrization
- antisymmetrization
- raising/lowering through metric-aware wrappers
- block and direct-sum constructions

## Tensor expressions

`TensorExpr` represents composite symbolic tensor expressions. The package includes helpers for expansion, reduction, and simplification.

## Symmetry tools

The tensor core also includes symmetry-oriented helpers such as:

- symmetric and antisymmetric projectors
- Young-projector-related utilities
- irreducible canonicalization helpers

These are useful when simplifying structured algebraic identities.
