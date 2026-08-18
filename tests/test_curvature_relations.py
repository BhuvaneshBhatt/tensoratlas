
from __future__ import annotations

from tensoratlas.curvature_relations import (
    DEFAULT_CURVATURE_ORIENTATION_POLICY,
    CURVATURE_EXECUTABLE_RULES,
    Riemann,
    Ricci,
    executable_curvature_reduce,
    CurvatureExpr,
)

def test_curvature_executable_rules_exist():
    assert len(CURVATURE_EXECUTABLE_RULES) >= 4
    assert CURVATURE_EXECUTABLE_RULES[0].metadata["terminating"] is True

def test_riemann_contracts_to_ricci():
    rep = executable_curvature_reduce(Riemann(4))
    assert any("riemann" in name.lower() for name in rep.applied_rules)
    assert len(rep.reduced_terms) >= 1

def test_ricci_contracts_or_decomposes():
    rep = executable_curvature_reduce(Ricci(4))
    assert len(rep.reduced_terms) >= 1
    assert rep.orientation_policy == DEFAULT_CURVATURE_ORIENTATION_POLICY.name

def test_reduced_terms_are_curvature_objects():
    rep = executable_curvature_reduce([(1, Riemann(4)), (1, Riemann(4))])
    coeff, term = rep.reduced_terms[0]
    assert coeff != 0
    assert hasattr(term, "family") or isinstance(term, CurvatureExpr)

def test_orientation_policy_present():
    rep = executable_curvature_reduce(Riemann(3))
    assert rep.metadata["preferred_targets"] == DEFAULT_CURVATURE_ORIENTATION_POLICY.preferred_targets
