"""Static checks that the public demo notebook contains executable workflows."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "tensoratlas_demo.ipynb"


REQUIRED_MARKDOWN_PHRASES = [
    "Coordinate systems and field transformations",
    "Vector calculus as tensor calculus",
    "Differential forms",
    "Electromagnetism with formal forms",
    "Tensor-valued forms and Cartan calculus",
    "Metric geometry: the 2-sphere",
    "Relativity workflow: Schwarzschild vacuum check",
    "Relativity workflow: FLRW cosmology",
    "Abstract tensor canonicalization",
    "Geometric algebra with orthogonal metrics",
]


REQUIRED_CODE_PHRASES = [
    "transform_scalar_field",
    "transform_vector_field",
    "coordinate_gradient",
    "coordinate_curl",
    "differential_forms_workflow()",
    "electromagnetic_workflow()",
    "cartan_structure_workflow()",
    "two_sphere_workflow()",
    "schwarzschild_workflow()",
    "flrw_workflow()",
    "geometric_algebra_workflow()",
    "canonicalization_workflow()",
    "riemann_component",
    "einstein_component",
]


def test_demo_notebook_is_merged_tutorial_with_workflows():
    data = json.loads(NOTEBOOK.read_text())
    assert len(data["cells"]) >= 50
    assert sum(cell["cell_type"] == "code" for cell in data["cells"]) >= 30

    markdown_text = "\n".join(
        "".join(cell.get("source", "")) for cell in data["cells"] if cell["cell_type"] == "markdown"
    )
    code_text = "\n".join(
        "".join(cell.get("source", "")) for cell in data["cells"] if cell["cell_type"] == "code"
    )

    for phrase in REQUIRED_MARKDOWN_PHRASES:
        assert phrase in markdown_text
    for phrase in REQUIRED_CODE_PHRASES:
        assert phrase in code_text


def test_only_primary_public_notebook_is_shipped():
    notebooks = sorted(path.name for path in (ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks == ["tensoratlas_demo.ipynb"]
