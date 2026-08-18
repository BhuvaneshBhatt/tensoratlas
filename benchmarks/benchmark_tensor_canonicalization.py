"""Small tensor-canonicalization benchmark for TensorAtlas.

Run with:
    python benchmarks/benchmark_tensor_canonicalization.py
"""
from __future__ import annotations

from time import perf_counter

from tensoratlas import IndexType, TensorHead, canonicalize_tensor_expression


def timed(label: str, func):
    start = perf_counter()
    value = func()
    elapsed = perf_counter() - start
    print(f"{label}: {elapsed:.4f}s")
    return value


def main() -> None:
    space = IndexType("M", 4)
    i, j, k, l = space.indices("i j k l")
    R = TensorHead("R", (space, space, space, space), antisymmetric_pairs=((0, 1), (2, 3)), pair_exchange_symmetry=True)
    expr = R(i, j, k, l) - R(k, l, i, j)
    timed("Riemann pair-exchange canonicalization", lambda: canonicalize_tensor_expression(expr))


if __name__ == "__main__":
    main()
