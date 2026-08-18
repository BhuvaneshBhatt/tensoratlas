# Semantic exterior/gamma operators and ordered execution layers

This layer promotes Hodge, codifferential, interior, Lie, and gamma-string objects to first-class semantic-core objects.

## Semantic operator objects

- `HodgeExpr`
- `CodifferentialExpr`
- `InteriorExpr`
- `LieExpr`
- `GammaStringExpr`

These compile into semantic-core nodes and can be evaluated through `semantic_operator_rules()`.

## Ordered execution helpers

- `metric_signature_from_chart(chart)`
- `exterior_execution_pipeline(form, ...)`
- `spin_execution_pipeline(spinor, frame, ...)`

These provide end-to-end execution-oriented workflows over the exterior and spin layers.
