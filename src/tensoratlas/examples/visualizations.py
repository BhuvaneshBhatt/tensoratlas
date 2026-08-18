"""Visualization examples for tensor theory and tensor calculus.

The functions in this module are intentionally lightweight.  They return
Matplotlib ``Figure`` objects and import plotting dependencies lazily so that
``import tensoratlas`` remains fast and does not load Matplotlib.
"""

from __future__ import annotations

from math import pi
from typing import Any


def _plt_np():
    """Import plotting dependencies lazily."""

    import numpy as np
    import matplotlib.pyplot as plt

    return plt, np


def _finish_2d(ax: Any, title: str, *, equal: bool = True) -> None:
    ax.set_title(title)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    if equal:
        ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.4)


def plot_basis_change():
    """Plot one geometric vector with standard and skew basis components."""

    plt, np = _plt_np()
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    b1 = np.array([1.0, 0.0])
    b2 = np.array([0.6, 1.1])
    vector = np.array([2.2, 1.4])
    skew_components = np.linalg.solve(np.column_stack([b1, b2]), vector)

    for basis_vec, label in [(e1, "e1"), (e2, "e2")]:
        ax.arrow(0, 0, basis_vec[0], basis_vec[1], head_width=0.05, length_includes_head=True)
        ax.text(*(basis_vec * 1.1), label)
    for basis_vec, label in [(b1, "b1"), (b2, "b2")]:
        ax.arrow(0, 0, basis_vec[0], basis_vec[1], head_width=0.05, length_includes_head=True, linestyle="--")
        ax.text(*(basis_vec * 1.15), label)
    ax.arrow(0, 0, vector[0], vector[1], head_width=0.08, length_includes_head=True, linewidth=2)
    ax.text(vector[0], vector[1] + 0.1, f"v = {tuple(vector)}")
    ax.text(0.05, -0.55, f"skew components ≈ ({skew_components[0]:.2f}, {skew_components[1]:.2f})")
    ax.set_xlim(-0.6, 3.0)
    ax.set_ylim(-0.8, 2.4)
    _finish_2d(ax, "Same vector, different basis components")
    return fig


def plot_contravariant_covariant_scaling():
    """Visualize component scaling for vectors and rates under unit changes."""

    plt, np = _plt_np()
    units = ["km", "m", "mm"]
    vector_components = [1, 1000, 1_000_000]
    rate_components = [60, 0.06, 0.00006]
    x = np.arange(len(units))

    fig, ax1 = plt.subplots(figsize=(6.0, 4.0))
    ax1.plot(x, vector_components, marker="o", label="length components")
    ax1.set_yscale("log")
    ax1.set_ylabel("fixed length: numerical component")
    ax1.set_xticks(x, units)

    ax2 = ax1.twinx()
    ax2.plot(x, rate_components, marker="s", linestyle="--", label="rate components")
    ax2.set_yscale("log")
    ax2.set_ylabel("fixed rate: numerical component")
    ax1.set_title("Contravariant and covariant scaling under unit changes")
    ax1.grid(True, linewidth=0.3, alpha=0.4)
    return fig


def plot_covector_level_sets(a: float = 2.0, b: float = 1.0):
    """Plot level sets of the covector alpha(x, y) = a*x + b*y."""

    plt, np = _plt_np()
    xs = np.linspace(-3, 3, 120)
    ys = np.linspace(-3, 3, 120)
    X, Y = np.meshgrid(xs, ys)
    Z = a * X + b * Y

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.contour(X, Y, Z, levels=np.arange(-6, 7, 1), linewidths=0.8)
    vectors = [np.array([1.5, 0.5]), np.array([0.5, 1.8]), np.array([-1.2, 1.0])]
    for vec in vectors:
        value = a * vec[0] + b * vec[1]
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.08, length_includes_head=True)
        ax.text(vec[0], vec[1], f"α(v)={value:.1f}")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    _finish_2d(ax, "Covectors as level-set counters")
    return fig


def plot_dual_basis():
    """Visualize a skew basis and the associated dual covector level sets."""

    plt, np = _plt_np()
    basis = np.array([[1.0, 0.6], [0.0, 1.2]])
    dual_rows = np.linalg.inv(basis)
    xs = np.linspace(-2, 2, 120)
    ys = np.linspace(-2, 2, 120)
    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for i, row in enumerate(dual_rows):
        Z = row[0] * X + row[1] * Y
        ax.contour(X, Y, Z, levels=[-1, 0, 1], linewidths=0.8, linestyles="--")
        ax.text(-1.9, 1.6 - 0.25 * i, f"ε^{i+1}(e_j)=δ^{i+1}_j")
    for i in range(2):
        vec = basis[:, i]
        ax.arrow(0, 0, vec[0], vec[1], head_width=0.06, length_includes_head=True)
        ax.text(*(vec * 1.1), f"e{i+1}")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    _finish_2d(ax, "Dual basis level sets")
    return fig


def plot_metric_unit_ellipse(metric=((3.0, 1.0), (1.0, 2.0))):
    """Plot the unit set v^T g v = 1 for a positive-definite metric."""

    plt, np = _plt_np()
    g = np.array(metric, dtype=float)
    angles = np.linspace(0, 2 * np.pi, 300)
    directions = np.vstack([np.cos(angles), np.sin(angles)])
    denom = np.einsum("ij,ji->i", directions.T @ g, directions)
    radii = 1.0 / np.sqrt(denom)
    points = directions * radii

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot(points[0], points[1])
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    _finish_2d(ax, "Metric unit set: vᵀgv = 1")
    return fig


def plot_raising_lowering():
    """Show a vector and covector level sets associated by a metric."""

    plt, np = _plt_np()
    g = np.array([[1.0, 0.0], [0.0, 3.0]])
    v = np.array([1.5, 0.8])
    cov = g @ v
    xs = np.linspace(-2, 2, 120)
    ys = np.linspace(-2, 2, 120)
    X, Y = np.meshgrid(xs, ys)
    Z = cov[0] * X + cov[1] * Y

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.contour(X, Y, Z, levels=np.arange(-6, 7, 1), linewidths=0.8)
    ax.arrow(0, 0, v[0], v[1], head_width=0.08, length_includes_head=True, linewidth=2)
    ax.text(v[0], v[1], "v")
    ax.text(-1.9, -1.8, "covector v♭ = g(v, ·) shown by level sets")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    _finish_2d(ax, "Raising/lowering: vector to covector")
    return fig


def plot_tensor_product_heatmap():
    """Plot an outer product as a heatmap."""

    plt, np = _plt_np()
    v = np.array([1, 2, -1])
    alpha = np.array([3, -2, 1])
    outer = np.outer(v, alpha)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    image = ax.imshow(outer, aspect="auto")
    for i in range(outer.shape[0]):
        for j in range(outer.shape[1]):
            ax.text(j, i, str(outer[i, j]), ha="center", va="center")
    ax.set_title("Elementary tensor v ⊗ α")
    ax.set_xlabel("covector slot")
    ax.set_ylabel("vector slot")
    fig.colorbar(image, ax=ax, shrink=0.8)
    return fig


def plot_contraction_diagram():
    """Draw a small slot-wiring diagram for contraction."""

    plt, _ = _plt_np()
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    ax.axis("off")
    ax.text(0.1, 0.65, r"$A^i{}_j$", fontsize=16)
    ax.text(0.33, 0.65, r"$B^j{}_k$", fontsize=16)
    ax.annotate("", xy=(0.31, 0.64), xytext=(0.23, 0.64), arrowprops={"arrowstyle": "<->"})
    ax.text(0.22, 0.25, "contract repeated j-slot", fontsize=10)
    ax.annotate("", xy=(0.62, 0.65), xytext=(0.43, 0.65), arrowprops={"arrowstyle": "->"})
    ax.text(0.68, 0.65, r"$C^i{}_k$", fontsize=16)
    ax.text(0.68, 0.35, r"$C^i{}_k=A^i{}_jB^j{}_k$", fontsize=12)
    ax.set_title("Tensor contraction as wiring one upper slot to one lower slot")
    return fig


def plot_vector_field_transformation():
    """Plot a radial vector field with polar basis directions."""

    plt, np = _plt_np()
    xs = np.linspace(-2, 2, 9)
    ys = np.linspace(-2, 2, 9)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X**2 + Y**2)
    U = np.divide(X, R, out=np.zeros_like(X), where=R > 0)
    V = np.divide(Y, R, out=np.zeros_like(Y), where=R > 0)

    fig, ax = plt.subplots(figsize=(5.3, 5.0))
    ax.quiver(X, Y, U, V, angles="xy")
    circle = plt.Circle((0, 0), 1.0, fill=False, linestyle="--")
    ax.add_patch(circle)
    ax.text(0.2, 1.1, "radial direction e_r changes with position")
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    _finish_2d(ax, "Vector field and rotating polar basis")
    return fig


def plot_differential_form_visual():
    """Visualize a 1-form and the oriented area element dx wedge dy."""

    plt, np = _plt_np()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    xs = np.linspace(-2, 2, 100)
    ys = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(xs, ys)
    axes[0].contour(X, Y, X + 2 * Y, levels=np.arange(-5, 6), linewidths=0.8)
    axes[0].set_title("1-form: level sets")
    axes[0].set_aspect("equal")
    square_x = [0, 1, 1, 0, 0]
    square_y = [0, 0, 1, 1, 0]
    axes[1].plot(square_x, square_y)
    axes[1].arrow(0.5, 0.1, 0.3, 0, head_width=0.05, length_includes_head=True)
    axes[1].arrow(0.9, 0.5, 0, 0.3, head_width=0.05, length_includes_head=True)
    axes[1].text(0.15, 0.5, r"$dx\wedge dy$", fontsize=14)
    axes[1].set_title("2-form: oriented area")
    axes[1].set_aspect("equal")
    for ax in axes:
        ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_pullback_grid():
    """Plot a rectangular parameter grid and its polar map into the plane."""

    plt, np = _plt_np()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    rs = np.linspace(0.3, 2.0, 7)
    thetas = np.linspace(0, 1.5 * np.pi, 10)
    for r in rs:
        theta_vals = np.linspace(0, 1.5 * np.pi, 100)
        axes[0].plot(theta_vals, np.full_like(theta_vals, r), linewidth=0.8)
        axes[1].plot(r * np.cos(theta_vals), r * np.sin(theta_vals), linewidth=0.8)
    for theta in thetas:
        r_vals = np.linspace(0.3, 2.0, 100)
        axes[0].plot(np.full_like(r_vals, theta), r_vals, linewidth=0.8)
        axes[1].plot(r_vals * np.cos(theta), r_vals * np.sin(theta), linewidth=0.8)
    axes[0].set_title("parameter grid (θ, r)")
    axes[1].set_title("mapped grid in Cartesian plane")
    axes[1].set_aspect("equal")
    for ax in axes:
        ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_curvature_on_sphere():
    """Draw a sphere with a small triangle suggesting parallel transport."""

    plt, np = _plt_np()
    fig = plt.figure(figsize=(5.5, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(X, Y, Z, linewidth=0.3, alpha=0.4)
    pts = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=2)
    ax.set_title("Curvature: transport around a spherical triangle")
    ax.set_box_aspect((1, 1, 1))
    return fig


def plot_sphere_geodesic():
    """Plot a great circle and a latitude circle on the unit sphere."""

    plt, np = _plt_np()
    fig = plt.figure(figsize=(5.5, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    u = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(u), np.sin(u), 0 * u, label="great circle")
    phi = np.pi / 4
    ax.plot(np.cos(u) * np.sin(phi), np.sin(u) * np.sin(phi), np.cos(phi) + 0 * u, linestyle="--", label="latitude circle")
    ax.legend(loc="upper left")
    ax.set_title("Geodesic intuition on the sphere")
    ax.set_box_aspect((1, 1, 1))
    return fig


def plot_stress_element():
    """Draw normal and shear stresses on a square material element."""

    plt, _ = _plt_np()
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    square = plt.Rectangle((-1, -1), 2, 2, fill=False, linewidth=2)
    ax.add_patch(square)
    arrows = [((1, 0), (0.7, 0), r"$\sigma_{xx}$"), ((0, 1), (0, 0.7), r"$\sigma_{yy}$"), ((1, 0.6), (0, 0.45), r"$\sigma_{xy}$"), ((0.6, 1), (0.45, 0), r"$\sigma_{yx}$")]
    for start, delta, label in arrows:
        ax.arrow(start[0], start[1], delta[0], delta[1], head_width=0.06, length_includes_head=True)
        ax.text(start[0] + delta[0], start[1] + delta[1], label)
    ax.set_xlim(-1.6, 1.9)
    ax.set_ylim(-1.6, 1.9)
    _finish_2d(ax, "Stress tensor components on a material element")
    return fig


def plot_strain_deformation():
    """Compare undeformed, normal-strain, and shear-strain squares."""

    plt, np = _plt_np()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    transforms = [np.eye(2), np.array([[1.25, 0], [0, 0.85]]), np.array([[1, 0.35], [0, 1]])]
    titles = ["original", "normal strain", "shear strain"]
    for ax, transform, title in zip(axes, transforms, titles):
        pts = square @ transform.T
        ax.plot(pts[:, 0], pts[:, 1])
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-0.2, 1.6)
        ax.set_ylim(-0.2, 1.4)
        ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_quadrupole_density():
    """Plot the disk density proportional to x^2 - y^2."""

    plt, np = _plt_np()
    xs = np.linspace(-1, 1, 200)
    ys = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(xs, ys)
    mask = X**2 + Y**2 <= 1
    Z = np.where(mask, X**2 - Y**2, np.nan)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    image = ax.imshow(Z, extent=(-1, 1, -1, 1), origin="lower")
    ax.set_title(r"Quadrupole-like density $\rho\propto x^2-y^2$")
    ax.set_aspect("equal")
    fig.colorbar(image, ax=ax, shrink=0.8)
    return fig


def plot_rotor_rotation(angle: float = pi / 4):
    """Visualize a 2D rotor-style rotation of a vector."""

    plt, np = _plt_np()
    v = np.array([1.4, 0.4])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    w = rot @ v
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    circle = plt.Circle((0, 0), np.linalg.norm(v), fill=False, linestyle="--")
    ax.add_patch(circle)
    ax.arrow(0, 0, v[0], v[1], head_width=0.06, length_includes_head=True, label="v")
    ax.arrow(0, 0, w[0], w[1], head_width=0.06, length_includes_head=True, linestyle="--")
    ax.text(v[0], v[1], "v")
    ax.text(w[0], w[1], "RvR⁻¹")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    _finish_2d(ax, "Geometric algebra rotor intuition")
    return fig


def plot_canonicalization_tree():
    """Draw two equivalent tensor-expression trees converging to one canonical form."""

    plt, _ = _plt_np()
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.axis("off")
    nodes = {
        "expr1": (0.15, 0.75),
        "expr2": (0.15, 0.25),
        "rename": (0.45, 0.75),
        "sym": (0.45, 0.25),
        "canon": (0.75, 0.5),
    }
    labels = {
        "expr1": r"$R^a{}_{bad}$",
        "expr2": r"$-R^a{}_{bda}$",
        "rename": "dummy rename",
        "sym": "antisymmetry",
        "canon": "canonical form",
    }
    for key, (x, y) in nodes.items():
        ax.text(x, y, labels[key], ha="center", va="center", bbox={"boxstyle": "round", "fc": "white"})
    for start, end in [("expr1", "rename"), ("expr2", "sym"), ("rename", "canon"), ("sym", "canon")]:
        ax.annotate("", xy=nodes[end], xytext=nodes[start], arrowprops={"arrowstyle": "->"})
    ax.set_title("Canonicalization: different expressions, one normal form")
    return fig


_VISUALIZATION_FUNCTIONS = [
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
]


def visualization_workflow():
    """Return the names of all tutorial visualization functions."""

    return [func.__name__ for func in _VISUALIZATION_FUNCTIONS]


__all__ = [func.__name__ for func in _VISUALIZATION_FUNCTIONS] + ["visualization_workflow"]
