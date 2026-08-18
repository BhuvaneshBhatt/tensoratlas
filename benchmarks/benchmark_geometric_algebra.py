"""Small geometric algebra benchmarks for TensorAtlas.

Run with:
    python benchmarks/benchmark_geometric_algebra.py
"""
from __future__ import annotations

from time import perf_counter

from tensoratlas.geometric_algebra import GeometricAlgebra


def timed(label: str, func):
    start = perf_counter()
    value = func()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.4f}s")
    return value


def main() -> None:
    ga = GeometricAlgebra.euclidean(5)
    e = ga.basis_vectors()
    A = sum(((i + 1) * e[i] for i in range(5)), ga.zero())
    B = sum(((i + 2) * e[i] for i in range(5)), ga.zero())
    timed("vector geometric product", lambda: A * B)
    timed("basis product cache reuse", lambda: [ga.basis_product(0, 1, 2, 1) for _ in range(1000)])


if __name__ == "__main__":
    main()
