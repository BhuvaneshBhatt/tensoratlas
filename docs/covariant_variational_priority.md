# Covariant variational calculus and remaining Priority B geometry layers

This layer adds two practical extensions on top of the semantic and execution infrastructure.

## Covariant variational calculus

The new `covariant_variational_geometry` module provides:

- `metric_density(chart, ...)`
- `metric_volume_form(chart, ...)`
- `covariant_variational_problem(lagrangian, field, chart, ...)`
- `perturb_metric_geometry(chart, perturbation_metric, ...)`

The current scope treats scalar-field Lagrangians on a metric chart by folding the metric volume density into the Euler--Lagrange calculation and then dividing back out to expose the covariant form.

## Remaining Priority B geometry features in current scope

The same module also adds:

- density descriptors (`DensityDef`)
- metric volume forms as `ExteriorFormNF`
- perturbation-order-aware metric / inverse metric / determinant / Christoffel expansions
- coordinate-orthogonal hypersurface geometry via `coordinate_hypersurface_geometry(...)`

The hypersurface helper currently targets coordinate hypersurfaces for block-orthogonal metrics. In that setting it computes:

- the induced metric
- a normalized coordinate normal
- extrinsic curvature
- mean curvature
- the induced volume density and volume form

These additions do not yet replace a full submanifold engine, but they provide a practical bridge toward mature geometry workflows while remaining consistent with the current TensorAtlas architecture.
