from __future__ import annotations

import sympy as sp

from benchmarks._common import print_report, run_case
from tensoratlas import (
    algebraically_equal_basic,
    basic_root_canonicalize,
    basic_root_cache_stats,
    clear_basic_root_cache,
    possibly_zero,
)


def _cold_zero_case() -> bool:
    clear_basic_root_cache()
    expr = sp.sqrt(2) + sp.sqrt(8) - 3 * sp.sqrt(2)
    return possibly_zero(expr, emit_warning=False)


def _warm_zero_case() -> bool:
    expr = sp.sqrt(2) + sp.sqrt(8) - 3 * sp.sqrt(2)
    basic_root_canonicalize(expr)
    return possibly_zero(expr, emit_warning=False)


def _special_angle_case() -> bool:
    expr = sp.exp(sp.I * sp.pi / 3) - (sp.Rational(1, 2) + sp.sqrt(3) * sp.I / 2)
    return possibly_zero(expr, emit_warning=False)


def _algebraic_equality_case() -> bool | None:
    return algebraically_equal_basic(sp.sqrt(2) + sp.sqrt(8), 3 * sp.sqrt(2))


def _uncertain_case() -> bool:
    return possibly_zero(sp.sin(sp.Symbol("x")), emit_warning=False)


def build_cases():
    clear_basic_root_cache()
    return [
        run_case("algebraic_zero_testing_cold", _cold_zero_case, repeat=200, metadata={"cache": basic_root_cache_stats()}),
        run_case("algebraic_zero_testing_warm", _warm_zero_case, repeat=500, metadata={"cache": basic_root_cache_stats()}),
        run_case("algebraic_special_angle_zero", _special_angle_case, repeat=200),
        run_case("algebraic_equality_basic", _algebraic_equality_case, repeat=300),
        run_case("algebraic_uncertain_symbolic", _uncertain_case, repeat=200),
    ]


def main() -> None:
    print_report(*build_cases())


if __name__ == "__main__":
    main()
