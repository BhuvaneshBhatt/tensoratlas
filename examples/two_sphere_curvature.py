"""Compute selected curvature quantities for the two-sphere.

Run from the repository root with:

    python examples/two_sphere_curvature.py
"""

from tensoratlas.examples import two_sphere_workflow

if __name__ == "__main__":
    result = two_sphere_workflow()
    print("Metric:", result["metric"])
    print("Scalar curvature residual R - 2/R^2:", result["scalar_curvature"])
    print("Gamma^theta_{phi phi}:", result["gamma_theta_phiphi"])
