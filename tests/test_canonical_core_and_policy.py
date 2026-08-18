import sympy as sp
from sympy.tensor.tensor import tensor_indices

from tensoratlas.abstract_tensor import (
    canonical_tensor_normal_form,
    fully_symmetric_head,
    index_type,
    build_operator_tree,
    collect_covariant_derivatives,
    compose_operator_trees,
    decompose_irreducible,
)
from tensoratlas.rewrite_families import (
    apply_rewrite_families,
    rewrite_policy,
    execute_rewrite_policy,
)


def test_canonical_tensor_normal_form_is_authoritative_under_slot_permutation():
    V = index_type("M")
    sym = V.to_sympy()
    S = fully_symmetric_head("Sauth", [sym, sym])
    a, b = tensor_indices("a b", sym)
    left = S(a, b)
    right = S(b, a)
    assert canonical_tensor_normal_form(left) == canonical_tensor_normal_form(right)


def test_explicit_family_request_wins_within_exclusive_mode_group():
    V = index_type("M")
    sym = V.to_sympy()
    S = fully_symmetric_head("Srw", [sym, sym])
    a, b = tensor_indices("a b", sym)
    expr = S(a, b)
    _, diagnostics = apply_rewrite_families(
        expr,
        families=("all", "metric_delta"),
        layer="abstract",
        mode="full",
        with_diagnostics=True,
    )
    assert diagnostics.applied_families == ("metric_delta",)
    assert ("all", "metric_delta") in diagnostics.mutually_exclusive_resolutions


def test_rewrite_trace_iterations_match_diagnostics():
    V = index_type("M")
    sym = V.to_sympy()
    S = fully_symmetric_head("Strace", [sym, sym])
    a, b = tensor_indices("a b", sym)
    expr = S(a, b)
    policy = rewrite_policy("abstract_metric_only", layer="abstract", families=("metric_delta",), mode="full")
    _, trace = execute_rewrite_policy(expr, policy, with_trace=True)
    assert trace.steps
    step = trace.steps[0]
    assert step.iterations == trace.diagnostics.iterations[step.family]


def test_nested_operator_tree_collection_and_composition():
    V = index_type("M")
    sym = V.to_sympy()
    S = fully_symmetric_head("Sop", [sym, sym])
    a, b = tensor_indices("a b", sym)
    c, d = tensor_indices("c d", sym)
    inner = build_operator_tree(S(a, b), (c,))
    outer = build_operator_tree(S(a, b), (d,))
    composed = compose_operator_trees(outer, inner)
    collected = collect_covariant_derivatives(composed)
    assert len(collected) == 1
    assert collected[0] == composed


def test_irreducible_decomposition_preserves_projector_count_and_canonicalizes_components():
    V = index_type("M")
    sym = V.to_sympy()
    T = fully_symmetric_head("Sirr", [sym, sym, sym, sym])
    a, b, c, d = tensor_indices("a b c d", sym)
    report = decompose_irreducible(T(a, b, c, d), [((0, 1), (2, 3)), ((0, 1), (2, 3))])
    assert len(report.projectors) == 2
    assert len(report.components) == 2
    assert report.components[0].expr == report.components[1].expr
