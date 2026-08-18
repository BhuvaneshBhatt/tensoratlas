from __future__ import annotations

from tensoratlas import (
    IndexedNormalizationConfig,
    cache_stats,
    clear_all_caches,
    coordinate_chart,
    indexed,
    indices,
    normalize_indexed_expression,
    TensorObject,
)
from tensoratlas.tensor_algebra import kronecker_delta_tensor, metric_tensor, tensor_product
from benchmarks._common import run_case, print_report


def build_expr():
    chart = coordinate_chart("Euclidean", "Cartesian", 4)
    delta = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    g = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0, 1),)})
    i, j, k, l = indices("i^ j_ k_ l_")
    return tensor_product(indexed(delta, i, j), indexed(g, k, l))


def main() -> None:
    expr = build_expr()
    clear_all_caches()
    case = run_case(
        "cache_controls",
        lambda: [normalize_indexed_expression(expr, IndexedNormalizationConfig(normalization_mode="heuristic")) for _ in range(200)],
        metadata={"cache_stats": cache_stats()},
    )
    print_report(case)


if __name__ == "__main__":
    main()
