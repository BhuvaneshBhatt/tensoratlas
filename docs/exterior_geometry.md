# Research Geometry Layer

This layer deepens the earlier breadth-oriented exterior-geometry work in four directions:

- spin connections and frame-coupled gamma calculus,
- canonicalized exterior-form normal forms and identity checks,
- fuller archive/import support for frames and bases,
- and broader benchmark/doc coverage.

## Spin geometry

The new `spin_connection(...)` helper builds a frame-attached spin connection from an existing frame and its Levi-Civita connection coefficients. It exposes both a coefficient dictionary and the underlying one-form coefficients. On top of that, `spin_covariant_components(...)` and `dirac_operator(...)` provide symbolic frame-coupled gamma calculus using the Clifford helpers introduced in the exterior/spin layer.

## Exterior-form normal forms

The new `ExteriorFormNF` data structure stores forms as canonicalized coefficient dictionaries keyed by ordered wedge blades. Canonicalization:

- sorts wedge blades into a standard order,
- tracks the induced sign,
- and annihilates repeated basis elements.

The layer also provides:

- `wedge_exterior_forms(...)`,
- `exterior_derivative_nf(...)`,
- `exterior_identity_report(...)`.

These make it practical to check identities such as $d^2 = 0$ and the graded Leibniz rule in a representation-independent way.

## Archive support

`geometry_to_data(...)` and `geometry_from_data(...)` now support serialized `TensorBasis`, `TensorFrame`, and `FrameDef` objects. Frame transforms are archived as symbolic matrices evaluated in the chart coordinate symbols and reconstructed as callable transform maps on import.

## Benchmarks

The companion benchmark file `benchmarks/bench_exterior_geometry.py` covers:

- spin-connection construction,
- Dirac-operator assembly,
- exterior-form canonicalization,
- and archive roundtrips for frame/basis data.
