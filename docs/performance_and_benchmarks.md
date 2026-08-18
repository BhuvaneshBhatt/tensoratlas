# Performance and benchmarks

TensorAtlas favors explicit symbolic objects and inspectable conventions.  Some operations, especially dense curvature tensors and tensor canonicalization with large symmetry groups, can be expensive.  Use selected-component APIs when you only need one component.

## Dense inspection helpers

The helpers `nonzero_christoffel`, `nonzero_riemann`, `nonzero_ricci`, and `nonzero_einstein` compute a dense tensor first and then filter nonzero components.  They are excellent for small tutorial metrics such as the 2-sphere, Schwarzschild, and simple FLRW models.  For larger symbolic metrics, prefer:

```python
from tensoratlas.relativity import christoffel_component, riemann_component, ricci_component, einstein_component
```

These selected-component APIs still need the metric inverse and, for some contractions, the Christoffel table, but they avoid materializing every dense higher-rank tensor component.

## Simplification controls

Most public symbolic routines accept `simplify=True`, `simplify=False`, or a callable simplifier.  Use `simplify=False` for construction and benchmarking, then simplify selected expressions explicitly for presentation.

```python
ricci = model.ricci(simplify=False)
component = model.christoffel_component(1, 0, 1, simplify=False)
```

## Benchmark scripts

The `benchmarks/` directory includes small scripts for sanity checking runtime on common workflows:

```bash
python benchmarks/benchmark_import_time.py
python benchmarks/benchmark_relativity.py
python benchmarks/benchmark_geometric_algebra.py
python benchmarks/benchmark_tensor_canonicalization.py
```

These scripts are not a replacement for rigorous performance profiling, but they catch obvious regressions in import time, curvature calculations, geometric algebra products, and canonicalization.

## Import-time expectations

The package root is intentionally lazy.  `import tensoratlas` should not import plotting libraries such as Matplotlib.  Heavy optional dependencies should be imported only by the functions that need them.
