# Benchmarks

The `benchmarks/` directory contains lightweight timing scripts for the main workloads.

- `bench_tensoratlas.py` — registry lookup, coordinate transforms, and core calculus
- `bench_geometry.py` — metric-aware gradient, Laplacian, and divergence workloads
- `bench_indexed_nf.py` — indexed normalization and equality checks
- `bench_cache_controls.py` — normalization throughput with cache stats and eviction visibility
- `bench_normalization_modes.py` — compares strict versus heuristic indexed normalization
- `bench_algebraic_zero_testing.py` — bounded algebraic canonicalization, conservative zero testing, warm-cache/cold-cache behavior, and bounded algebraic equality checks
- `bench_public_vs_tnf_matrices.py` — compares plain SymPy matrix-returning APIs with the corresponding `_tnf` matrix paths

Run any script with:

```bash
PYTHONPATH=src python benchmarks/<script_name>.py
```

## Common benchmark reporting

The benchmark scripts can use `benchmarks._common` for a shared JSON-style report schema.

- `run_case(...)` records elapsed wall time, per-iteration time, and optional metadata
- `print_report(...)` prints a consistent JSON payload
- `benchmark_report.py` aggregates representative workloads into one summary

Run the aggregate report with:

```bash
PYTHONPATH=src python benchmarks/benchmark_report.py
```

The benchmark surface now covers:
- chart and geometry workloads
- indexed canonicalization workloads
- cache behavior and normalization-mode comparisons
- bounded algebraic scalar canonicalization and zero-testing paths
