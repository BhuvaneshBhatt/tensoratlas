
from __future__ import annotations
import sympy as sp
from tensoratlas.charts import get_chart
from tensoratlas.basis import tangent_basis
from tensoratlas.exterior_geometry import ExteriorFormNF
from tensoratlas.geometry_components import basis_frame_transform_report, component_geometry_report, general_metric_hodge, general_metric_codifferential, general_metric_hodge_report

def test_basis_frame_transform_report_identity():
    chart = get_chart("Euclidean", "Cartesian", 2)
    tb = tangent_basis(chart)
    rep = basis_frame_transform_report(tb, tb)
    assert rep.transform_matrix == sp.eye(2)
    assert rep.inverse_matrix == sp.eye(2)

def test_component_geometry_report_flat_metric():
    chart = get_chart("Euclidean", "Cartesian", 2)
    rep = component_geometry_report(chart, include_curvature=True)
    assert rep.metric_matrix == sp.eye(2)
    assert rep.scalar_curvature == 0

def test_general_metric_hodge_diagonal_metric():
    form = ExteriorFormNF(2, {(0,): sp.Integer(1)}, basis_labels=("e0","e1"), metadata={})
    star = general_metric_hodge(form, sp.diag(2, 3))
    assert star.degree == 1
    assert (1,) in star.terms

def test_general_metric_codifferential_runs():
    x0, x1 = sp.symbols("x0 x1")
    form = ExteriorFormNF(2, {(0,): x0, (1,): x1}, basis_labels=("e0","e1"), metadata={})
    delta = general_metric_codifferential(form, sp.diag(1, 2), coords=(x0, x1))
    assert isinstance(delta.terms, dict)

def test_general_metric_hodge_report():
    form = ExteriorFormNF(2, {tuple(): sp.Integer(1)}, basis_labels=("e0","e1"), metadata={})
    rep = general_metric_hodge_report(form, sp.diag(1, 1))
    assert rep.output_degree == 2
