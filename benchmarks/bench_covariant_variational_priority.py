from __future__ import annotations

import time
import sympy as sp

from tensoratlas.charts import CoordinateChart
from tensoratlas.covariant_variational_geometry import (
    covariant_variational_problem,
    perturb_metric_geometry,
    coordinate_hypersurface_geometry,
)


def _polar_chart():
    return CoordinateChart(
        metric_name="Euclidean",
        chart_name="Polar",
        dimension=2,
        coordinate_names=("r", "theta"),
        metric_func=lambda coords: sp.Matrix([[1, 0], [0, coords[0] ** 2]]),
    )


def bench_covariant_variational_priority() -> dict[str, float]:
    chart = _polar_chart()
    r, theta = chart.symbols()
    phi = sp.Function("phi")(r, theta)
    h = sp.Function("h")(r, theta)

    t0 = time.perf_counter()
    covariant_variational_problem(sp.Rational(1, 2) * sp.diff(phi, r) ** 2, phi, chart)
    t1 = time.perf_counter()
    perturb_metric_geometry(chart, sp.Matrix([[h, 0], [0, 0]]), parameter=sp.Symbol("eps"), order=1)
    t2 = time.perf_counter()
    coordinate_hypersurface_geometry(chart, 0, level=sp.Symbol("R", positive=True))
    t3 = time.perf_counter()

    return {
        "covariant_variation_seconds": t1 - t0,
        "metric_perturbation_seconds": t2 - t1,
        "hypersurface_geometry_seconds": t3 - t2,
    }


if __name__ == "__main__":
    print(bench_covariant_variational_priority())
