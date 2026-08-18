"""Differential-form examples."""

from __future__ import annotations

from tensoratlas.differential_forms_frame import basis_one_form, wedge_forms


def differential_forms_workflow() -> dict[str, object]:
    """Build a simple two-form from basis one-forms."""
    dx = basis_one_form("dx")
    dy = basis_one_form("dy")
    return {"dx": dx, "dy": dy, "area_form": wedge_forms(dx, dy)}
