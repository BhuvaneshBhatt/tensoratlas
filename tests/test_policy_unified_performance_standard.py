from sympy.tensor.tensor import tensor_indices

from tensoratlas.abstract_tensor import index_type, fully_symmetric_head
from tensoratlas.performance_engineering import timed_execute_rewrite_policy
from tensoratlas.rewrite_families import (
    compose_rewrite_policies,
    override_rewrite_policy_families,
    rewrite_policy,
)
from tensoratlas.standard_tensor_library import list_standard_tensor_libraries, standard_object
from tensoratlas.unified_reduction import unified_reduce_with_trace, unified_tensor_normal_form


def test_unified_reduce_with_trace_uses_rewrite_policy_step():
    idx = index_type("M")
    S = fully_symmetric_head("S", [idx.to_sympy(), idx.to_sympy()])
    a, b = tensor_indices("a b", idx.to_sympy())
    expr = S(-b, -a)
    policy = rewrite_policy("abstract_only_metric", layer="abstract", families=("metric_delta",), mode="full", diagnostics=True)
    reduced, trace = unified_reduce_with_trace(expr, policy=policy)
    assert trace.steps[0].name == "rewrite_policy"
    assert trace.steps[0].details["policy"] == "abstract_only_metric"
    assert str(getattr(reduced, "expr", reduced)) == str(S(-a, -b))
    nf = unified_tensor_normal_form(expr, policy=policy)
    assert nf.layer == "abstract"


def test_policy_composition_and_family_override_surface():
    p1 = rewrite_policy("p1", layer="abstract", families=("linearity",), mode="full")
    p2 = rewrite_policy("p2", layer="abstract", families=("metric_delta",), mode="full", family_overrides={"metric_delta": {"max_passes": 2}})
    composed = compose_rewrite_policies("combo", p1, p2)
    assert composed.families == ("linearity", "metric_delta")
    assert composed.family_overrides["metric_delta"]["max_passes"] == 2
    overridden = override_rewrite_policy_families(composed, append=("multiterm",), family_overrides={"multiterm": {"dimension": 4}})
    assert "multiterm" in overridden.families
    assert overridden.family_overrides["multiterm"]["dimension"] == 4


def test_standard_library_expands_beyond_riemannian_bundle():
    names = list_standard_tensor_libraries()
    assert "riemannian_geometry" in names
    assert "lorentzian_geometry" in names
    assert "symplectic_geometry" in names
    assert standard_object("metric", library="lorentzian_geometry").name == "eta"


def test_timed_execute_rewrite_policy_reports_family_samples():
    idx = index_type("M")
    S = fully_symmetric_head("S", [idx.to_sympy(), idx.to_sympy()])
    a, b = tensor_indices("a b", idx.to_sympy())
    expr = S(-b, -a)
    policy = rewrite_policy("combo", layer="abstract", families=("linearity", "metric_delta"), mode="full", diagnostics=True)
    _, report = timed_execute_rewrite_policy(expr, policy, use_cache=False)
    assert report.family_samples
    assert tuple(sample.name for sample in report.family_samples) == ("linearity", "metric_delta")
