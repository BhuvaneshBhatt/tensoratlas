import sympy as sp

from tensoratlas.charts import CoordinateChart
from tensoratlas.covariant_variational_geometry import (
    metric_density,
    metric_volume_form,
    covariant_variational_problem,
    perturb_metric_geometry,
    coordinate_hypersurface_geometry,
)


def polar_chart():
    return CoordinateChart(
        metric_name="Euclidean",
        chart_name="Polar",
        dimension=2,
        coordinate_names=("r", "theta"),
        metric_func=lambda coords: sp.Matrix([[1, 0], [0, coords[0] ** 2]]),
    )


def test_metric_density_and_volume_form_polar():
    chart = polar_chart()
    r, theta = chart.symbols()
    dens = metric_density(chart)
    assert sp.simplify(dens.expression - r) == 0
    vol = metric_volume_form(chart)
    assert vol.dimension == 2
    assert sp.simplify(vol.terms[(0, 1)] - r) == 0


def test_covariant_variational_problem_scalar_polar_radial():
    chart = polar_chart()
    r, theta = chart.symbols()
    phi = sp.Function("phi")(r, theta)
    L = sp.Rational(1, 2) * sp.diff(phi, r) ** 2
    result = covariant_variational_problem(L, phi, chart)
    expected = -sp.diff(phi, r, 2) - sp.diff(phi, r) / r
    assert sp.simplify(result.covariant_euler - expected) == 0


def test_perturb_metric_geometry_first_order():
    chart = polar_chart()
    r, theta = chart.symbols()
    h = sp.Function("h")(r, theta)
    report = perturb_metric_geometry(chart, sp.Matrix([[h, 0], [0, 0]]), parameter=sp.Symbol("eps"), order=1)
    eps = report.parameter
    assert sp.simplify(report.expanded_metric[0, 0] - (1 + eps * h)) == 0
    assert sp.simplify(report.inverse_metric[0, 0] - (1 - eps * h)) == 0
    assert sp.simplify(report.determinant - (r**2 + eps * h * r**2)) == 0


def test_coordinate_hypersurface_geometry_circle_in_polar():
    chart = polar_chart()
    report = coordinate_hypersurface_geometry(chart, 0, level=sp.Symbol("R", positive=True))
    R = sp.Symbol("R", positive=True)
    assert report.induced_coordinates == (sp.Symbol("theta", real=True),)
    assert sp.simplify(report.induced_metric[0, 0] - R**2) == 0
    assert sp.simplify(report.extrinsic_curvature[0, 0] - R) == 0
    assert sp.simplify(report.mean_curvature - 1 / R) == 0
    assert sp.simplify(report.volume_density - R) == 0
