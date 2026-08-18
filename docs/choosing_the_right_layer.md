# Choosing the right TensorAtlas layer

TensorAtlas intentionally exposes several mathematical layers. Choosing the right one keeps examples smaller and avoids unnecessary symbolic work.

## Explicit components and coordinate fields

Use `tensoratlas.core` when you have explicit components, charts, coordinate maps, scalar fields, vector fields, covector fields, tensor fields, tensor products, contractions, gradients, divergence, curl, Hessians, or Laplacians.

## Metric geometry and relativity

Use `tensoratlas.relativity` when your workflow starts from a metric and needs Christoffel symbols, Riemann curvature, Ricci curvature, scalar curvature, Einstein tensors, selected components, nonzero-component inspection, or geodesic equations. For larger metrics, prefer selected-component functions before asking for dense full tensors.

## Differential forms and frames

Use exterior-calculus helpers when the calculation is naturally written with wedge products, exterior derivatives, frames, coframes, Hodge-oriented constructions, electromagnetic forms, or Cartan structure equations.

## Abstract and indexed tensors

Use the abstract/indexed APIs when the important operation is symbolic tensor identity management: dummy-index normalization, slot symmetry, canonicalization, normal forms, and equivalence testing.

## Geometric algebra

Use `tensoratlas.geometric_algebra` for orthogonal-metric geometric algebra. The current public layer supports diagonal/orthogonal metrics and rejects non-diagonal Clifford metrics deliberately.
