# Tensor theory and physical applications

The main tutorial now includes a beginner-friendly tensor-theory chapter before the package workflow examples.  It covers vectors, covectors, dual bases, contravariant and covariant behavior, tensors as multilinear maps, metric tensors, raising and lowering indices, tensor products, contractions, and transformation rules.

## Core ideas

A tensor is not merely an array.  Arrays store components after a basis or coordinate system has been chosen.  A tensor is the geometric or multilinear object whose components transform predictably when that choice changes.

The tutorial includes executable examples for:

- vector-covector pairings;
- unit-scaling intuition for contravariant and covariant quantities;
- dual bases and the Kronecker delta;
- non-orthogonal basis changes;
- linear maps as `(1, 1)` tensors;
- metrics as bilinear forms and vector-to-covector maps;
- metric pullbacks from Cartesian to polar coordinates;
- tensor products, elementary tensors, traces, and contractions;
- the general `(r, s)` tensor transformation rule.

## Physical examples

The tutorial also includes two physical tensor examples:

- a quadrupole moment tensor for a planar disk charge distribution;
- stress, strain, and fourth-order stiffness tensors in linear elasticity.

The tested implementations live in:

```text
src/tensoratlas/examples/tensor_theory.py
src/tensoratlas/examples/physical_tensors.py
```

and runnable scripts are available in:

```text
examples/tensor_theory_workflow.py
examples/physical_tensors_workflow.py
```
