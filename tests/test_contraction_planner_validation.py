from tensoratlas import (
    IndexedNormalizationConfig,
    build_contraction_graph,
    build_contraction_plan,
    coordinate_chart,
    indexed,
    indices,
    TensorObject,
)
from tensoratlas.tensor_algebra import kronecker_delta_tensor, metric_tensor, tensor_product
from tensoratlas.tensorform_planning import _estimate_factor_cost


def _planner_expr(shared=True):
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    delta = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    metric = TensorObject.from_tensor_field(metric_tensor(chart, "ll"), name="g", symmetry_metadata={"symmetric": ((0, 1),)})
    if shared:
        i, j, k, l = indices("i^ j_ j^ k_")
        return tensor_product(indexed(delta, i, j), indexed(delta, k, l))
    i, j, k, l = indices("i^ j_ k_ l_")
    return tensor_product(indexed(delta, i, j), indexed(metric, k, l))


def test_contraction_graph_is_symmetric():
    expr = _planner_expr(shared=True)
    graph = build_contraction_graph(expr)
    for i, nbrs in graph.items():
        for j, weight in nbrs.items():
            assert graph[j][i] == weight


def test_contraction_plan_is_deterministic():
    expr = _planner_expr(shared=True)
    p1 = build_contraction_plan(expr, config=IndexedNormalizationConfig())
    p2 = build_contraction_plan(expr, config=IndexedNormalizationConfig())
    names1 = tuple(getattr(getattr(f, "tensor", None), "name", "") for f in p1.ordered_factors)
    names2 = tuple(getattr(getattr(f, "tensor", None), "name", "") for f in p2.ordered_factors)
    assert names1 == names2
    assert p1.estimated_cost == p2.estimated_cost


def test_contraction_plan_cost_uses_unique_edges_only():
    expr = _planner_expr(shared=True)
    plan = build_contraction_plan(expr)
    graph = build_contraction_graph(expr)
    unique_edge_cost = sum(weight for i, nbrs in graph.items() for j, weight in nbrs.items() if i < j)
    factor_cost = sum(_estimate_factor_cost(f) for f in plan.ordered_factors)
    assert plan.estimated_cost == factor_cost + unique_edge_cost


def test_single_factor_graph_has_no_self_edges():
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    delta = TensorObject.from_tensor_field(kronecker_delta_tensor(chart), name="δ")
    i, j = indices("i^ i_")
    factor = indexed(delta, i, j)
    graph = build_contraction_graph([factor])
    assert graph == {0: {}}


def test_disconnected_factors_have_zero_edge_cost():
    expr = _planner_expr(shared=False)
    plan = build_contraction_plan(expr)
    graph = build_contraction_graph(expr)
    assert all(not nbrs for nbrs in graph.values())
    factor_cost = sum(_estimate_factor_cost(f) for f in plan.ordered_factors)
    assert plan.estimated_cost == factor_cost
