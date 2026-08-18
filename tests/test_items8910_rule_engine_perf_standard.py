from sympy.tensor.tensor import tensor_indices
from tensoratlas.rewrite_families import (
    RewritePolicy,
    apply_registered_rewrite_pipeline,
    execute_rewrite_policy,
    list_rewrite_policies,
    rewrite_context,
    rewrite_policy,
    select_rewrite_policy,
)
from tensoratlas.performance_engineering import (
    clear_reducer_caches,
    get_reducer_cache_registry,
    timed_execute_rewrite_policy,
)
from tensoratlas.standard_tensor_library import (
    get_standard_tensor_library,
    list_standard_tensor_libraries,
    standard_identity_library,
    standard_object,
)
from tensoratlas.abstract_tensor import index_type, fully_symmetric_head, tensor_head


def test_rewrite_policy_selection_and_execution_trace():
    idx = index_type("M")
    S = fully_symmetric_head("S", [idx.to_sympy(), idx.to_sympy()])
    a, b = tensor_indices("a b", idx.to_sympy())
    expr = S(-b, -a)
    policy = select_rewrite_policy(layer="abstract")
    result, trace = execute_rewrite_policy(expr, policy, context=rewrite_context(layer="abstract"), with_trace=True)
    assert str(getattr(result, "expr", result)) == str(S(-a, -b))
    assert trace.policy.layer == "abstract"
    assert len(trace.steps) >= 1


def test_registered_pipeline_uses_default_policy():
    idx = index_type("M")
    S = fully_symmetric_head("S", [idx.to_sympy(), idx.to_sympy()])
    a, b = tensor_indices("a b", idx.to_sympy())
    result, trace = apply_registered_rewrite_pipeline(S(-b, -a), layer="abstract", with_trace=True)
    assert str(getattr(result, "expr", result)) == str(S(-a, -b))
    assert trace.policy.name.startswith("abstract")


def test_timed_execute_rewrite_policy_uses_cache():
    clear_reducer_caches()
    idx = index_type("M")
    S = fully_symmetric_head("S", [idx.to_sympy(), idx.to_sympy()])
    a, b = tensor_indices("a b", idx.to_sympy())
    expr = S(-b, -a)
    policy = rewrite_policy("quick", layer="abstract", families=("metric_delta",), mode="full", diagnostics=True)
    (_, report1) = timed_execute_rewrite_policy(expr, policy)
    (_, report2) = timed_execute_rewrite_policy(expr, policy)
    assert report1.samples[0].cache_hit is False
    assert report2.samples[0].cache_hit is True
    stats = get_reducer_cache_registry().stats()
    assert "policy" in stats


def test_standard_library_objects_and_identities():
    names = list_standard_tensor_libraries()
    assert "riemannian_geometry" in names
    lib = get_standard_tensor_library("riemannian_geometry")
    assert "metric" in lib.objects
    metric_obj = standard_object("metric")
    assert metric_obj.name == "g"
    identity_lib = standard_identity_library("full")
    assert identity_lib.name == "full"
