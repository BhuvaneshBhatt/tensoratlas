from __future__ import annotations

from benchmarks._common import print_report, run_case
from tensoratlas import coordinate_chart, coordinate_map


def _build_objects():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    mapping = coordinate_map(polar, cart)
    return polar, mapping


def _jacobian_sympy():
    _, mapping = _build_objects()
    return mapping.jacobian()


def _jacobian_tnf():
    _, mapping = _build_objects()
    return mapping.jacobian_tnf()


def _metric_sympy():
    polar, _ = _build_objects()
    return polar.metric()


def _metric_tnf():
    polar, _ = _build_objects()
    return polar.metric_tnf()


def build_cases():
    return [
        run_case("jacobian_sympy", _jacobian_sympy, repeat=200),
        run_case("jacobian_tnf", _jacobian_tnf, repeat=200),
        run_case("metric_sympy", _metric_sympy, repeat=200),
        run_case("metric_tnf", _metric_tnf, repeat=200),
    ]


def main() -> None:
    print_report(*build_cases())


if __name__ == "__main__":
    main()
