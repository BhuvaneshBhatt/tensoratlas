from tensoratlas import __public_api__
from tensoratlas.indexed_pipeline import INDEXED_NORMALIZATION_STAGES, IndexedNormalizationPlan


def test_public_api_curated_surface_contains_core_entry_points():
    required = {
        "coordinate_chart",
        "coordinate_map",
        "transform_coordinates",
        "transform_field",
        "gradient",
        "divergence",
        "curl",
        "laplacian",
        "ScalarField",
        "VectorField",
        "TensorField",
    }
    assert required.issubset(set(__public_api__))


def test_indexed_normalization_plan_is_documented_and_ordered():
    assert INDEXED_NORMALIZATION_STAGES[0] == "expand_expression"
    assert INDEXED_NORMALIZATION_STAGES[-1] == "rebuild_expression"
    plan = IndexedNormalizationPlan()
    assert "alpha_rename_dummies" in plan.describe()
