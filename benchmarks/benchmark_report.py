from __future__ import annotations

from tensoratlas import cache_stats
from benchmarks._common import print_report, run_case
from benchmarks.bench_normalization_modes import run_mode
from benchmarks.bench_cache_controls import build_expr as build_cache_expr
from benchmarks.bench_contraction_planner import main as planner_main
from benchmarks.bench_algebraic_zero_testing import build_cases as build_algebraic_cases
from benchmarks.bench_public_vs_tnf_matrices import build_cases as build_api_matrix_cases


def main() -> None:
    strict_case = run_case("normalization_strict", lambda: run_mode("strict"), repeat=1, metadata={"cache_stats": cache_stats()})
    heuristic_case = run_case("normalization_heuristic", lambda: run_mode("heuristic"), repeat=1, metadata={"cache_stats": cache_stats()})
    sample_case = run_case("cache_expr_build", build_cache_expr, repeat=5)
    planner_case = run_case("planner_driver", planner_main, repeat=1)
    print_report(strict_case, heuristic_case, sample_case, planner_case, *build_algebraic_cases(), *build_api_matrix_cases())


if __name__ == "__main__":
    main()
