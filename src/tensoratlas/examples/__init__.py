"""Executable example workflows used by tutorials and tests."""

from .canonicalization import canonicalization_workflow
from .cartan_geometry import cartan_structure_workflow
from .coordinate_calculus import coordinate_calculus_workflow
from .differential_forms import differential_forms_workflow
from .electromagnetism import electromagnetic_workflow
from .geometric_algebra import geometric_algebra_workflow
from .relativity import flrw_workflow, schwarzschild_workflow, two_sphere_workflow
from .tensor_theory import (
    basis_change_example,
    covariant_contravariant_scaling_example,
    dual_basis_example,
    metric_pullback_example,
    metric_raising_lowering_example,
    multilinear_metric_example,
    notation_table,
    tensor_product_contraction_example,
    tensor_theory_workflow,
    vector_covector_pairing_example,
)
from .physical_tensors import (
    physical_tensor_workflow,
    quadrupole_moment_disk_example,
    stress_strain_stiffness_example,
)

__all__ = [
    "canonicalization_workflow",
    "cartan_structure_workflow",
    "coordinate_calculus_workflow",
    "differential_forms_workflow",
    "electromagnetic_workflow",
    "geometric_algebra_workflow",
    "two_sphere_workflow",
    "schwarzschild_workflow",
    "flrw_workflow",
    "notation_table",
    "vector_covector_pairing_example",
    "covariant_contravariant_scaling_example",
    "dual_basis_example",
    "basis_change_example",
    "multilinear_metric_example",
    "metric_raising_lowering_example",
    "metric_pullback_example",
    "tensor_product_contraction_example",
    "tensor_theory_workflow",
    "quadrupole_moment_disk_example",
    "stress_strain_stiffness_example",
    "physical_tensor_workflow",
]
from .usability import (
    coordinate_map_summary_example,
    forms_and_ga_usability_example,
    relativity_usability_example,
    tensor_array_usability_example,
    usability_workflow_examples,
)

from .visualizations import (
    plot_basis_change,
    plot_contravariant_covariant_scaling,
    plot_covector_level_sets,
    plot_dual_basis,
    plot_metric_unit_ellipse,
    plot_raising_lowering,
    plot_tensor_product_heatmap,
    plot_contraction_diagram,
    plot_vector_field_transformation,
    plot_differential_form_visual,
    plot_pullback_grid,
    plot_curvature_on_sphere,
    plot_sphere_geodesic,
    plot_stress_element,
    plot_strain_deformation,
    plot_quadrupole_density,
    plot_rotor_rotation,
    plot_canonicalization_tree,
    visualization_workflow,
)

__all__ += [
    "coordinate_map_summary_example",
    "forms_and_ga_usability_example",
    "relativity_usability_example",
    "tensor_array_usability_example",
    "usability_workflow_examples",
    "plot_basis_change",
    "plot_contravariant_covariant_scaling",
    "plot_covector_level_sets",
    "plot_dual_basis",
    "plot_metric_unit_ellipse",
    "plot_raising_lowering",
    "plot_tensor_product_heatmap",
    "plot_contraction_diagram",
    "plot_vector_field_transformation",
    "plot_differential_form_visual",
    "plot_pullback_grid",
    "plot_curvature_on_sphere",
    "plot_sphere_geodesic",
    "plot_stress_element",
    "plot_strain_deformation",
    "plot_quadrupole_density",
    "plot_rotor_rotation",
    "plot_canonicalization_tree",
    "visualization_workflow",
]
