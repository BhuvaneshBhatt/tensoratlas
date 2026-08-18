from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from .canonical_keys import structural_key


@dataclass(frozen=True)
class ReconstructionDiagnostics:
    factor_count: int
    contracted_name_count: int
    free_name_count: int
    repeated_free_occurrences: int
    assignment_size: int


def _slot_identity(slot: Any) -> tuple[Any, ...]:
    return tuple(slot) if isinstance(slot, tuple) else (slot,)


def _slot_sort_key(slot_id: tuple[Any, ...]):
    return tuple(structural_key(x) for x in slot_id)


def _occurrence_plan(term) -> dict[tuple[int, int], tuple[str, str]]:
    occurrences: dict[tuple[Any, ...], dict[str, list[tuple[int, int]]]] = defaultdict(lambda: {"u": [], "l": []})
    for factor_pos, factor in enumerate(getattr(term, "factors", tuple())):
        slots = getattr(factor, "typed_slots", tuple())
        variances = getattr(factor, "variance_spec", tuple())
        for slot_pos, (slot, variance) in enumerate(zip(slots, variances)):
            occurrences[_slot_identity(slot)][variance].append((factor_pos, slot_pos))

    assignment: dict[tuple[int, int], tuple[str, str]] = {}
    next_contracted = 0
    next_free = 0
    for slot_id in sorted(occurrences, key=_slot_sort_key):
        uppers = deque(occurrences[slot_id]["u"])
        lowers = deque(occurrences[slot_id]["l"])
        pair_count = min(len(uppers), len(lowers))
        for _ in range(pair_count):
            name = f"d{next_contracted}"
            next_contracted += 1
            up_occ = uppers.popleft()
            low_occ = lowers.popleft()
            assignment[up_occ] = (name, "u")
            assignment[low_occ] = (name, "l")
        while uppers:
            occ = uppers.popleft()
            name = f"f{next_free}"
            next_free += 1
            assignment[occ] = (name, "u")
        while lowers:
            occ = lowers.popleft()
            name = f"f{next_free}"
            next_free += 1
            assignment[occ] = (name, "l")
    return assignment


def reconstruction_diagnostics(term) -> ReconstructionDiagnostics:
    assignment = _occurrence_plan(term)
    names = [name for name, _ in assignment.values()]
    contracted = {name for name in names if name.startswith("d")}
    free = {name for name in names if name.startswith("f")}
    repeated_free = len([name for name in names if name.startswith("f")]) - len(free)
    return ReconstructionDiagnostics(
        factor_count=len(getattr(term, "factors", tuple())),
        contracted_name_count=len(contracted),
        free_name_count=len(free),
        repeated_free_occurrences=max(repeated_free, 0),
        assignment_size=len(assignment),
    )


def tnf_term_to_expr(term, *, factor_converter, tensor_index_cls, indexed_tensor_cls, indexed_expr_cls, scalar_field_cls, scalar_cleanup):
    factors = getattr(term, "factors", tuple())
    if not factors:
        return scalar_field_cls(None, scalar_cleanup(getattr(term, "scalar", 0)))

    assignment = _occurrence_plan(term)
    rendered = []
    for factor_pos, factor in enumerate(factors):
        tensor = factor_converter(factor)
        indices = []
        for slot_pos, variance in enumerate(getattr(factor, "variance_spec", tuple())):
            name, actual_variance = assignment[(factor_pos, slot_pos)]
            indices.append(tensor_index_cls(name, actual_variance))
        rendered.append(indexed_tensor_cls(tensor, tuple(indices)))

    expr = rendered[0]
    for factor in rendered[1:]:
        expr = indexed_expr_cls("tensor_product", (expr, factor))

    scalar = scalar_cleanup(getattr(term, "scalar", 1))
    if scalar == 1:
        return expr
    if scalar == 0:
        return scalar_field_cls(None, 0)
    return indexed_expr_cls("tensor_product", (scalar_field_cls(None, scalar), expr))
