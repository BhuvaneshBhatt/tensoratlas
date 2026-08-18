# Overview

TensorAtlas is a symbolic toolkit for differential geometry, coordinate calculus, component tensors, abstract-index tensor algebra, tensor-valued forms, curvature workflows, and conservative tensor canonicalization.

The preferred public surface is organized by domain: `tensoratlas.core`, `tensoratlas.relativity`, `tensoratlas.tensor_valued_forms`, `tensoratlas.geometric_algebra`, `tensoratlas.display`, and `tensoratlas.examples`. The top-level `tensoratlas` package exposes a small convenience surface for common workflows.

## Design principles

- Keep coordinate/component calculations separate from abstract tensor semantics.
- Use explicit convention metadata for curvature, Hodge operations, Clifford products, Laplacians, and density transformations.
- Prefer bounded simplification and avoid broad implicit `sympy.simplify` calls in hot paths.
- Provide selected-component and nonzero-component APIs for expensive curvature calculations.
- Make common workflows inspectable through `summary()` and `validate()` methods.
- Raise TensorAtlas-specific exceptions with actionable messages for shape, coordinate, metric, form-degree, and unsupported-geometry errors.

## Main workflows

- Standard-coordinate catalog and coordinate-map metadata.
- Coordinate-field transformations for scalars, vectors, covectors, tensors, and tensor densities.
- Coordinate-basis vector calculus, including convention-aware Hessian and Laplacian helpers.
- Component metric geometry, including Christoffel symbols, curvature components, frames, vielbeins, and spin-oriented helpers.
- Tensor-valued forms for solder forms, connection one-forms, torsion forms, curvature forms, and Cartan equations.
- Abstract tensor canonicalization with strict index validation, slot symmetries, dummy-index normalization, repeated-factor handling, and common-case zero detection.
- Differential forms, Hodge star, Lie derivative, pullback, and form inner products.
- Orthogonal-metric multivector geometric algebra.

## Project status

TensorAtlas is prepared as an initial public release. The package emphasizes correctness, explicit conventions, and tested examples. Some advanced symbolic algorithms are reference implementations intended for small and medium symbolic problems; the documentation identifies the relevant performance boundaries.
