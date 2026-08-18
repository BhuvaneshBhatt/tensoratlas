# Covariant variational calculus and geometry workflows

This layer adds practical extensions on top of the semantic and execution infrastructure.

## Covariant variational calculus

The `covariant_variational_geometry` module provides:

- `metric_density(chart, ...)`
- `metric_volume_form(chart, ...)`
- `covariant_variational_problem(lagrangian, field, chart, ...)`
- `perturb_metric_geometry(chart, perturbation_metric, ...)`

Scalar-field Lagrangians on a metric chart can be handled by folding the metric volume density into the Euler--Lagrange calculation and then dividing back out to expose the covariant form.

## Geometry features

The same module also adds:

- density descriptors (`DensityDef`)
- metric volume forms as `ExteriorFormNF`
- perturbation-order-aware metric, inverse metric, determinant, and Christoffel expansions
- coordinate-orthogonal hypersurface geometry through `coordinate_hypersurface_geometry(...)`

The hypersurface helper targets coordinate hypersurfaces for block-orthogonal metrics. In that setting it computes:

- the induced metric
- a normalized coordinate normal
- extrinsic curvature
- mean curvature
- the induced volume density and volume form

These additions provide a practical bridge toward richer geometry workflows while remaining consistent with the TensorAtlas architecture.
