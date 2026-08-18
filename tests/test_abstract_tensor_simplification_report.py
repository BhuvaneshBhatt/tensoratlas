import pytest

from tensoratlas.abstract_tensor import (
    AbstractSimplificationReport,
    Index,
    IndexType,
    Metric,
    TensorHead,
    simplify_abstract,
    simplify_abstract_with_report,
)


def test_simplify_abstract_with_report_tracks_requested_stages_and_changes():
    lor = IndexType("Lor", dummy_name="L")
    i = Index("i", lor, "u")
    j = Index("j", lor, "u")
    g = Metric(lor)
    A = TensorHead("A", [lor])
    expr = g(i, -j) * A(j).to_sympy()

    simplified, report = simplify_abstract_with_report(expr, mode="metric")

    assert str(simplified.to_sympy()) == "A(i)"
    assert isinstance(report, AbstractSimplificationReport)
    assert report.requested_stages == ("structural", "metric")
    assert tuple(step.name for step in report.executed_steps) == ("structural", "metric")
    assert report.executed_steps[-1].changed is True


def test_simplify_abstract_accepts_explicit_stage_sequence():
    lor = IndexType("Lor", dummy_name="L")
    i = Index("i", lor, "u")
    j = Index("j", lor, "u")
    g = Metric(lor)
    A = TensorHead("A", [lor])
    expr = g(i, -j) * A(j).to_sympy()

    simplified = simplify_abstract(expr, mode=("metric", "structural"))
    assert str(simplified.to_sympy()) == "A(i)"


@pytest.mark.parametrize("bad_mode", ["unknown", ("structural", "oops")])
def test_simplify_abstract_rejects_unknown_modes(bad_mode):
    lor = IndexType("Lor", dummy_name="L")
    i = Index("i", lor, "u")
    A = TensorHead("A", [lor])
    expr = A(i).to_sympy()
    with pytest.raises(Exception):
        simplify_abstract(expr, mode=bad_mode)
