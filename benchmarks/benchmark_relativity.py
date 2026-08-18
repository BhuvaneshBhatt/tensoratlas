"""Small relativity benchmarks for TensorAtlas.

Run with:
    python benchmarks/benchmark_relativity.py
"""
from __future__ import annotations

from time import perf_counter

import sympy as sp

from tensoratlas.relativity import (
    christoffel_symbols,
    ricci_tensor,
    scalar_curvature,
    schwarzschild_metric,
    two_sphere_metric,
)


def timed(label: str, func):
    start = perf_counter()
    value = func()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.4f}s")
    return value


def main() -> None:
    R = sp.symbols("R", positive=True)
    timed("2-sphere Christoffel", lambda: christoffel_symbols(two_sphere_metric(R), simplify=False))
    timed("2-sphere scalar curvature", lambda: scalar_curvature(two_sphere_metric(R), simplify=True))
    M = sp.symbols("M", positive=True)
    timed("Schwarzschild Ricci tensor", lambda: ricci_tensor(schwarzschild_metric(M), simplify=False))


if __name__ == "__main__":
    main()
