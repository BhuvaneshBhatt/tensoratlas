# Tensor-valued forms, relativity, and geometric algebra

This page describes the public conventions for three high-level TensorAtlas workflows.

## Tensor-valued forms

`TensorValuedForm` represents a formal exterior form whose coefficients take values in a tensor bundle.  The exterior degree is stored separately from the tensor-value variance.  Component keys index only the tensor-value slots; each key must have the same length as the variance tuple.

Cartan helpers use the conventions

```text
T^a = dθ^a + ω^a{}_b θ^b
Ω^a{}_b = dω^a{}_b + ω^a{}_c ω^c{}_b
```

The current exterior derivative constructor is formal.  Coordinate exterior differentiation is intentionally not inferred from symbols.

## Relativity

The metric catalog includes Minkowski, the two-sphere, Schwarzschild, and FLRW metrics.  Built-in spacetime metrics use mostly-plus signature `(-,+,+,+)`.

The curvature convention is

```text
R^a{}_{bcd} = ∂_c Γ^a{}_{bd} - ∂_d Γ^a{}_{bc}
              + Γ^a{}_{ce} Γ^e{}_{bd} - Γ^a{}_{de} Γ^e{}_{bc}
R_bd = R^a{}_{bad}
R = g^{ab} R_ab
G_ab = R_ab - 1/2 g_ab R
```

Curvature functions accept `simplify=True`, `simplify=False`, or a callable simplifier.  `MetricModel` caches its inverse metric to avoid repeated symbolic inversion.

## Geometric algebra

`GeometricAlgebra` implements multivectors over an orthogonal basis.  The metric may be a sequence of diagonal entries or a diagonal matrix.  Non-diagonal matrices raise `NotImplementedError`, because arbitrary Clifford products require contraction terms that are not equivalent to sorted exterior blades.

Supported operations include geometric product, exterior product, inner product, left contraction, grade projection, reverse, grade involution, Clifford conjugation, scalar part, norm-squared, simple inverse, dual, commutator, rotor construction, rotation, reflection, and projection.
