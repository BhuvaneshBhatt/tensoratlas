"""Physical tensor examples for the public tutorial."""

from __future__ import annotations

import sympy as sp


def quadrupole_moment_disk_example() -> dict[str, object]:
    """Compute a planar quadrupole moment for a disk charge distribution.

    The density is rho0 * (x**2 - y**2) on a disk of radius R in the z=0 plane.
    The angular/radial integrals are simple enough to present symbolically, and
    the function returns the evaluated tensor rather than re-integrating every
    time the tutorial notebook is executed.
    """
    rho0, radius = sp.symbols("rho0 R", positive=True)
    r, theta = sp.symbols("r theta", positive=True)
    x = r * sp.cos(theta)
    y = r * sp.sin(theta)
    rho = rho0 * (x**2 - y**2)
    q_xx = sp.pi * rho0 * radius**6 / 4
    q_yy = -sp.pi * rho0 * radius**6 / 4
    q_xy = sp.Integer(0)
    q_zz = sp.Integer(0)
    tensor = sp.Matrix([[q_xx, q_xy, 0], [q_xy, q_yy, 0], [0, 0, q_zz]])
    return {
        "density": rho,
        "quadrupole_tensor": tensor,
        "trace": sp.simplify(sp.trace(tensor)),
        "integration_note": "Computed from polar integrals with dA = r dr dtheta.",
    }


def stress_strain_stiffness_example() -> dict[str, object]:
    """Show the 2D isotropic linear-elastic stress-strain relation."""
    lam, mu = sp.symbols("lambda mu")
    eps11, eps22, eps12 = sp.symbols("epsilon_11 epsilon_22 epsilon_12")
    strain = sp.Matrix([[eps11, eps12], [eps12, eps22]])
    identity = sp.eye(2)
    stress = lam * sp.trace(strain) * identity + 2 * mu * strain

    # Explicit fourth-order stiffness tensor C_ijkl for sigma_ij = C_ijkl epsilon_kl.
    delta = lambda i, j: sp.Integer(1) if i == j else sp.Integer(0)
    stiffness = {}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    stiffness[(i, j, k, l)] = lam * delta(i, j) * delta(k, l) + mu * (delta(i, k) * delta(j, l) + delta(i, l) * delta(j, k))
    reconstructed = sp.Matrix(
        [
            [sum(stiffness[(i, j, k, l)] * strain[k, l] for k in range(2) for l in range(2)) for j in range(2)]
            for i in range(2)
        ]
    )
    return {
        "strain": strain,
        "stress": sp.simplify(stress),
        "stiffness_components": stiffness,
        "reconstruction_residual": sp.simplify(reconstructed - stress),
    }


def physical_tensor_workflow() -> dict[str, object]:
    """Return physical tensor examples used by the tutorial."""
    return {"quadrupole": quadrupole_moment_disk_example(), "elasticity": stress_strain_stiffness_example()}
