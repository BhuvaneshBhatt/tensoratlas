from __future__ import annotations

import sympy as sp

from .charts import get_chart
from .mappings import CoordinateMap, register_map


def register_builtin_maps() -> None:
    # 2D Cartesian <-> Polar
    cart2 = get_chart("Euclidean", "Cartesian", 2)
    polar2 = get_chart("Euclidean", "Polar", 2)
    elliptic2 = get_chart("Euclidean", "Elliptic", 2)
    parabolic2 = get_chart("Euclidean", "Parabolic", 2)
    bipolar2 = get_chart("Euclidean", "Bipolar", 2)

    register_map(CoordinateMap(
        source=cart2,
        target=polar2,
        mapping_exprs_func=lambda coords: (
            sp.sqrt(coords[0]**2 + coords[1]**2),
            sp.atan2(coords[1], coords[0]),
        ),
        inverse_exprs_func=lambda coords: (
            coords[0] * sp.cos(coords[1]),
            coords[0] * sp.sin(coords[1]),
        ),
        metadata={"standard_name": "Cartesian->Polar"},
    ))

    register_map(CoordinateMap(
        source=polar2,
        target=cart2,
        mapping_exprs_func=lambda coords: (
            coords[0] * sp.cos(coords[1]),
            coords[0] * sp.sin(coords[1]),
        ),
        inverse_exprs_func=lambda coords: (
            sp.sqrt(coords[0]**2 + coords[1]**2),
            sp.atan2(coords[1], coords[0]),
        ),
        metadata={"standard_name": "Polar->Cartesian"},
    ))

    register_map(CoordinateMap(
        source=elliptic2,
        target=cart2,
        mapping_exprs_func=lambda coords: (
            elliptic2.parameters()["a"] * sp.cosh(coords[0]) * sp.cos(coords[1]),
            elliptic2.parameters()["a"] * sp.sinh(coords[0]) * sp.sin(coords[1]),
        ),
        inverse_exprs_func=lambda coords: (
            sp.acosh((sp.sqrt((coords[0] + elliptic2.parameters()["a"])**2 + coords[1]**2) + sp.sqrt((coords[0] - elliptic2.parameters()["a"])**2 + coords[1]**2)) / (2 * elliptic2.parameters()["a"])),
            sp.acos((sp.sqrt((coords[0] + elliptic2.parameters()["a"])**2 + coords[1]**2) - sp.sqrt((coords[0] - elliptic2.parameters()["a"])**2 + coords[1]**2)) / (2 * elliptic2.parameters()["a"])),
        ),
        metadata={"standard_name": "Elliptic->Cartesian"},
    ))

    register_map(CoordinateMap(
        source=cart2,
        target=elliptic2,
        mapping_exprs_func=lambda coords: (
            sp.acosh((sp.sqrt((coords[0] + elliptic2.parameters()["a"])**2 + coords[1]**2) + sp.sqrt((coords[0] - elliptic2.parameters()["a"])**2 + coords[1]**2)) / (2 * elliptic2.parameters()["a"])),
            sp.acos((sp.sqrt((coords[0] + elliptic2.parameters()["a"])**2 + coords[1]**2) - sp.sqrt((coords[0] - elliptic2.parameters()["a"])**2 + coords[1]**2)) / (2 * elliptic2.parameters()["a"])),
        ),
        inverse_exprs_func=lambda coords: (
            elliptic2.parameters()["a"] * sp.cosh(coords[0]) * sp.cos(coords[1]),
            elliptic2.parameters()["a"] * sp.sinh(coords[0]) * sp.sin(coords[1]),
        ),
        metadata={"standard_name": "Cartesian->Elliptic"},
    ))


    register_map(CoordinateMap(
        source=parabolic2,
        target=cart2,
        mapping_exprs_func=lambda coords: (
            (coords[0]**2 - coords[1]**2) / 2,
            coords[0] * coords[1],
        ),
        inverse_exprs_func=lambda coords: (
            sp.sign(coords[1]) * sp.sqrt(sp.sqrt(coords[0]**2 + coords[1]**2) + coords[0]),
            sp.sqrt(sp.sqrt(coords[0]**2 + coords[1]**2) - coords[0]),
        ),
        metadata={"standard_name": "Parabolic->Cartesian", "symbolic_inverse_kind": "principal_branch"},
    ))

    register_map(CoordinateMap(
        source=cart2,
        target=parabolic2,
        mapping_exprs_func=lambda coords: (
            sp.sign(coords[1]) * sp.sqrt(sp.sqrt(coords[0]**2 + coords[1]**2) + coords[0]),
            sp.sqrt(sp.sqrt(coords[0]**2 + coords[1]**2) - coords[0]),
        ),
        inverse_exprs_func=lambda coords: (
            (coords[0]**2 - coords[1]**2) / 2,
            coords[0] * coords[1],
        ),
        metadata={"standard_name": "Cartesian->Parabolic", "symbolic_inverse_kind": "principal_branch"},
    ))


    register_map(CoordinateMap(
        source=bipolar2,
        target=cart2,
        mapping_exprs_func=lambda coords: _bipolar_to_cartesian(coords, bipolar2.parameters()["a"]),
        inverse_exprs_func=lambda coords: _cartesian_to_bipolar(coords, bipolar2.parameters()["a"]),
        metadata={"standard_name": "Bipolar->Cartesian"},
    ))

    register_map(CoordinateMap(
        source=cart2,
        target=bipolar2,
        mapping_exprs_func=lambda coords: _cartesian_to_bipolar(coords, bipolar2.parameters()["a"]),
        inverse_exprs_func=lambda coords: _bipolar_to_cartesian(coords, bipolar2.parameters()["a"]),
        metadata={"standard_name": "Cartesian->Bipolar"},
    ))

    # 3D common systems
    cart3 = get_chart("Euclidean", "Cartesian", 3)
    cyl3 = get_chart("Euclidean", "Cylindrical", 3)
    sph3 = get_chart("Euclidean", "Spherical", 3)
    parab3 = get_chart("Euclidean", "Paraboloidal", 3)
    prolate3 = get_chart("Euclidean", "ProlateSpheroidal", 3)
    oblate3 = get_chart("Euclidean", "OblateSpheroidal", 3)
    bisph3 = get_chart("Euclidean", "Bispherical", 3)
    tor3 = get_chart("Euclidean", "Toroidal", 3)
    parabcyl3 = get_chart("Euclidean", "ParabolicCylindrical", 3)
    ellipticcyl3 = get_chart("Euclidean", "EllipticCylindrical", 3)
    conical3 = get_chart("Euclidean", "Conical", 3)
    ellipsoidal3 = get_chart("Euclidean", "Ellipsoidal", 3)

    register_map(CoordinateMap(
        source=cyl3,
        target=cart3,
        mapping_exprs_func=lambda coords: (
            coords[0] * sp.cos(coords[1]),
            coords[0] * sp.sin(coords[1]),
            coords[2],
        ),
        inverse_exprs_func=lambda coords: (
            sp.sqrt(coords[0]**2 + coords[1]**2),
            sp.atan2(coords[1], coords[0]),
            coords[2],
        ),
        metadata={"standard_name": "Cylindrical->Cartesian"},
    ))

    register_map(CoordinateMap(
        source=cart3,
        target=cyl3,
        mapping_exprs_func=lambda coords: (
            sp.sqrt(coords[0]**2 + coords[1]**2),
            sp.atan2(coords[1], coords[0]),
            coords[2],
        ),
        inverse_exprs_func=lambda coords: (
            coords[0] * sp.cos(coords[1]),
            coords[0] * sp.sin(coords[1]),
            coords[2],
        ),
        metadata={"standard_name": "Cartesian->Cylindrical"},
    ))

    register_map(CoordinateMap(
        source=sph3,
        target=cart3,
        mapping_exprs_func=lambda coords: (
            coords[0] * sp.cos(coords[2]) * sp.sin(coords[1]),
            coords[0] * sp.sin(coords[2]) * sp.sin(coords[1]),
            coords[0] * sp.cos(coords[1]),
        ),
        inverse_exprs_func=lambda coords: (
            sp.sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2),
            sp.atan2(sp.sqrt(coords[0]**2 + coords[1]**2), coords[2]),
            sp.atan2(coords[1], coords[0]),
        ),
        metadata={"standard_name": "Spherical->Cartesian"},
    ))

    register_map(CoordinateMap(
        source=cart3,
        target=sph3,
        mapping_exprs_func=lambda coords: (
            sp.sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2),
            sp.atan2(sp.sqrt(coords[0]**2 + coords[1]**2), coords[2]),
            sp.atan2(coords[1], coords[0]),
        ),
        inverse_exprs_func=lambda coords: (
            coords[0] * sp.cos(coords[2]) * sp.sin(coords[1]),
            coords[0] * sp.sin(coords[2]) * sp.sin(coords[1]),
            coords[0] * sp.cos(coords[1]),
        ),
        metadata={"standard_name": "Cartesian->Spherical"},
    ))

    register_map(CoordinateMap(
        source=cyl3,
        target=sph3,
        mapping_exprs_func=lambda coords: (
            sp.sqrt(coords[0]**2 + coords[2]**2),
            sp.atan2(coords[0], coords[2]),
            coords[1],
        ),
        inverse_exprs_func=lambda coords: (
            coords[0] * sp.sin(coords[1]),
            coords[2],
            coords[0] * sp.cos(coords[1]),
        ),
        metadata={"standard_name": "Cylindrical->Spherical"},
    ))

    register_map(CoordinateMap(
        source=sph3,
        target=cyl3,
        mapping_exprs_func=lambda coords: (
            coords[0] * sp.sin(coords[1]),
            coords[2],
            coords[0] * sp.cos(coords[1]),
        ),
        inverse_exprs_func=lambda coords: (
            sp.sqrt(coords[0]**2 + coords[2]**2),
            sp.atan2(coords[0], coords[2]),
            coords[1],
        ),
        metadata={"standard_name": "Spherical->Cylindrical"},
    ))

    register_map(CoordinateMap(
        source=parab3,
        target=cart3,
        mapping_exprs_func=lambda coords: (
            coords[0] * coords[1] * sp.cos(coords[2]),
            coords[0] * coords[1] * sp.sin(coords[2]),
            (coords[0]**2 - coords[1]**2) / 2,
        ),
        inverse_exprs_func=lambda coords: (
            sp.sqrt(sp.sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2) + coords[2]),
            sp.sqrt(sp.sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2) - coords[2]),
            sp.atan2(coords[1], coords[0]),
        ),
        metadata={"standard_name": "Paraboloidal->Cartesian"},
    ))

    register_map(CoordinateMap(
        source=cart3,
        target=parab3,
        mapping_exprs_func=lambda coords: (
            sp.sqrt(sp.sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2) + coords[2]),
            sp.sqrt(sp.sqrt(coords[0]**2 + coords[1]**2 + coords[2]**2) - coords[2]),
            sp.atan2(coords[1], coords[0]),
        ),
        inverse_exprs_func=lambda coords: (
            coords[0] * coords[1] * sp.cos(coords[2]),
            coords[0] * coords[1] * sp.sin(coords[2]),
            (coords[0]**2 - coords[1]**2) / 2,
        ),
        metadata={"standard_name": "Cartesian->Paraboloidal"},
    ))

    register_map(CoordinateMap(
        source=prolate3,
        target=cart3,
        mapping_exprs_func=lambda coords: (
            prolate3.parameters()["a"] * sp.sinh(coords[0]) * sp.sin(coords[1]) * sp.cos(coords[2]),
            prolate3.parameters()["a"] * sp.sinh(coords[0]) * sp.sin(coords[1]) * sp.sin(coords[2]),
            prolate3.parameters()["a"] * sp.cosh(coords[0]) * sp.cos(coords[1]),
        ),
        inverse_exprs_func=lambda coords: (
            sp.acosh((sp.sqrt(coords[0]**2 + coords[1]**2 + (coords[2] + prolate3.parameters()["a"])**2) + sp.sqrt(coords[0]**2 + coords[1]**2 + (coords[2] - prolate3.parameters()["a"])**2)) / (2 * prolate3.parameters()["a"])),
            sp.acos((sp.sqrt(coords[0]**2 + coords[1]**2 + (coords[2] + prolate3.parameters()["a"])**2) - sp.sqrt(coords[0]**2 + coords[1]**2 + (coords[2] - prolate3.parameters()["a"])**2)) / (2 * prolate3.parameters()["a"])),
            sp.atan2(coords[1], coords[0]),
        ),
        metadata={"standard_name": "ProlateSpheroidal->Cartesian"},
    ))

    register_map(CoordinateMap(
        source=cart3,
        target=prolate3,
        mapping_exprs_func=lambda coords: (
            sp.acosh((sp.sqrt(coords[0]**2 + coords[1]**2 + (coords[2] + prolate3.parameters()["a"])**2) + sp.sqrt(coords[0]**2 + coords[1]**2 + (coords[2] - prolate3.parameters()["a"])**2)) / (2 * prolate3.parameters()["a"])),
            sp.acos((sp.sqrt(coords[0]**2 + coords[1]**2 + (coords[2] + prolate3.parameters()["a"])**2) - sp.sqrt(coords[0]**2 + coords[1]**2 + (coords[2] - prolate3.parameters()["a"])**2)) / (2 * prolate3.parameters()["a"])),
            sp.atan2(coords[1], coords[0]),
        ),
        inverse_exprs_func=lambda coords: (
            prolate3.parameters()["a"] * sp.sinh(coords[0]) * sp.sin(coords[1]) * sp.cos(coords[2]),
            prolate3.parameters()["a"] * sp.sinh(coords[0]) * sp.sin(coords[1]) * sp.sin(coords[2]),
            prolate3.parameters()["a"] * sp.cosh(coords[0]) * sp.cos(coords[1]),
        ),
        metadata={"standard_name": "Cartesian->ProlateSpheroidal"},
    ))

    register_map(CoordinateMap(
        source=oblate3,
        target=cart3,
        mapping_exprs_func=lambda coords: (
            oblate3.parameters()["a"] * sp.cosh(coords[0]) * sp.cos(coords[1]) * sp.cos(coords[2]),
            oblate3.parameters()["a"] * sp.cosh(coords[0]) * sp.cos(coords[1]) * sp.sin(coords[2]),
            oblate3.parameters()["a"] * sp.sinh(coords[0]) * sp.sin(coords[1]),
        ),
        inverse_exprs_func=lambda coords: _cartesian_to_oblate(coords, oblate3.parameters()["a"]),
        metadata={"standard_name": "OblateSpheroidal->Cartesian"},
    ))

    register_map(CoordinateMap(
        source=cart3,
        target=oblate3,
        mapping_exprs_func=lambda coords: _cartesian_to_oblate(coords, oblate3.parameters()["a"]),
        inverse_exprs_func=lambda coords: (
            oblate3.parameters()["a"] * sp.cosh(coords[0]) * sp.cos(coords[1]) * sp.cos(coords[2]),
            oblate3.parameters()["a"] * sp.cosh(coords[0]) * sp.cos(coords[1]) * sp.sin(coords[2]),
            oblate3.parameters()["a"] * sp.sinh(coords[0]) * sp.sin(coords[1]),
        ),
        metadata={"standard_name": "Cartesian->OblateSpheroidal"},
    ))

    register_map(CoordinateMap(
        source=bisph3,
        target=cart3,
        mapping_exprs_func=lambda coords: _bispherical_to_cartesian(coords, bisph3.parameters()["a"]),
        inverse_exprs_func=lambda coords: _cartesian_to_bispherical(coords, bisph3.parameters()["a"]),
        metadata={"standard_name": "Bispherical->Cartesian"},
    ))

    register_map(CoordinateMap(
        source=cart3,
        target=bisph3,
        mapping_exprs_func=lambda coords: _cartesian_to_bispherical(coords, bisph3.parameters()["a"]),
        inverse_exprs_func=lambda coords: _bispherical_to_cartesian(coords, bisph3.parameters()["a"]),
        metadata={"standard_name": "Cartesian->Bispherical"},
    ))

    register_map(CoordinateMap(
        source=tor3,
        target=cart3,
        mapping_exprs_func=lambda coords: _toroidal_to_cartesian(coords, tor3.parameters()["a"]),
        inverse_exprs_func=lambda coords: _cartesian_to_toroidal(coords, tor3.parameters()["a"]),
        metadata={"standard_name": "Toroidal->Cartesian"},
    ))

    register_map(CoordinateMap(
        source=cart3,
        target=tor3,
        mapping_exprs_func=lambda coords: _cartesian_to_toroidal(coords, tor3.parameters()["a"]),
        inverse_exprs_func=lambda coords: _toroidal_to_cartesian(coords, tor3.parameters()["a"]),
        metadata={"standard_name": "Cartesian->Toroidal"},
    ))



    register_map(CoordinateMap(
        source=parabcyl3,
        target=cart3,
        mapping_exprs_func=lambda coords: _parabolic_cylindrical_to_cartesian(coords),
        inverse_exprs_func=lambda coords: _cartesian_to_parabolic_cylindrical(coords),
        metadata={"standard_name": "ParabolicCylindrical->Cartesian"},
    ))

    register_map(CoordinateMap(
        source=cart3,
        target=parabcyl3,
        mapping_exprs_func=lambda coords: _cartesian_to_parabolic_cylindrical(coords),
        inverse_exprs_func=lambda coords: _parabolic_cylindrical_to_cartesian(coords),
        metadata={"standard_name": "Cartesian->ParabolicCylindrical"},
    ))

    register_map(CoordinateMap(
        source=ellipticcyl3,
        target=cart3,
        mapping_exprs_func=lambda coords: _elliptic_cylindrical_to_cartesian(coords, ellipticcyl3.parameters()["a"]),
        inverse_exprs_func=lambda coords: _cartesian_to_elliptic_cylindrical(coords, ellipticcyl3.parameters()["a"]),
        metadata={"standard_name": "EllipticCylindrical->Cartesian"},
    ))

    register_map(CoordinateMap(
        source=cart3,
        target=ellipticcyl3,
        mapping_exprs_func=lambda coords: _cartesian_to_elliptic_cylindrical(coords, ellipticcyl3.parameters()["a"]),
        inverse_exprs_func=lambda coords: _elliptic_cylindrical_to_cartesian(coords, ellipticcyl3.parameters()["a"]),
        metadata={"standard_name": "Cartesian->EllipticCylindrical"},
    ))

    register_map(CoordinateMap(
        source=conical3,
        target=cart3,
        mapping_exprs_func=lambda coords: _conical_to_cartesian(coords, conical3.parameters()["b"], conical3.parameters()["c"]),
        inverse_exprs_func=lambda coords: _cartesian_to_conical(coords, conical3.parameters()["b"], conical3.parameters()["c"]),
        metadata={"standard_name": "Conical->Cartesian", "branch": "principal-octant symbolic inverse via quadratic roots", "symbolic_inverse_kind": "root_based", "branch_assumptions": sp.And(sp.Q.real(sp.Symbol("x")), sp.Q.real(sp.Symbol("y")), sp.Q.real(sp.Symbol("z")), sp.Ge(sp.Symbol("x"), 0), sp.Ge(sp.Symbol("y"), 0), sp.Ge(sp.Symbol("z"), 0))},
    ))

    register_map(CoordinateMap(
        source=ellipsoidal3,
        target=cart3,
        mapping_exprs_func=lambda coords: _ellipsoidal_to_cartesian(coords, ellipsoidal3.parameters()["a"], ellipsoidal3.parameters()["b"], ellipsoidal3.parameters()["c"]),
        inverse_exprs_func=lambda coords: _cartesian_to_ellipsoidal(coords, ellipsoidal3.parameters()["a"], ellipsoidal3.parameters()["b"], ellipsoidal3.parameters()["c"]),
        metadata={"standard_name": "Ellipsoidal->Cartesian", "branch": "principal-octant symbolic inverse via cubic roots", "symbolic_inverse_kind": "root_based", "branch_assumptions": sp.And(sp.Q.real(sp.Symbol("x")), sp.Q.real(sp.Symbol("y")), sp.Q.real(sp.Symbol("z")), sp.Ge(sp.Symbol("x"), 0), sp.Ge(sp.Symbol("y"), 0), sp.Ge(sp.Symbol("z"), 0))},
    ))


    register_map(CoordinateMap(
        source=cart3,
        target=conical3,
        mapping_exprs_func=lambda coords: _cartesian_to_conical(coords, conical3.parameters()["b"], conical3.parameters()["c"]),
        inverse_exprs_func=lambda coords: _conical_to_cartesian(coords, conical3.parameters()["b"], conical3.parameters()["c"]),
        metadata={"standard_name": "Cartesian->Conical", "branch": "principal-octant symbolic inverse via quadratic roots", "symbolic_inverse_kind": "root_based", "branch_assumptions": sp.And(sp.Q.real(sp.Symbol("x")), sp.Q.real(sp.Symbol("y")), sp.Q.real(sp.Symbol("z")), sp.Ge(sp.Symbol("x"), 0), sp.Ge(sp.Symbol("y"), 0), sp.Ge(sp.Symbol("z"), 0))},
    ))

    register_map(CoordinateMap(
        source=cart3,
        target=ellipsoidal3,
        mapping_exprs_func=lambda coords: _cartesian_to_ellipsoidal(coords, ellipsoidal3.parameters()["a"], ellipsoidal3.parameters()["b"], ellipsoidal3.parameters()["c"]),
        inverse_exprs_func=lambda coords: _ellipsoidal_to_cartesian(coords, ellipsoidal3.parameters()["a"], ellipsoidal3.parameters()["b"], ellipsoidal3.parameters()["c"]),
        metadata={"standard_name": "Cartesian->Ellipsoidal", "branch": "principal-octant symbolic inverse via cubic roots", "symbolic_inverse_kind": "root_based", "branch_assumptions": sp.And(sp.Q.real(sp.Symbol("x")), sp.Q.real(sp.Symbol("y")), sp.Q.real(sp.Symbol("z")), sp.Ge(sp.Symbol("x"), 0), sp.Ge(sp.Symbol("y"), 0), sp.Ge(sp.Symbol("z"), 0))},
    ))


    mink4 = get_chart("Minkowski", "Cartesian", 4)
    rind4 = get_chart("Rindler", "Standard", 4)

    register_map(CoordinateMap(
        source=rind4,
        target=mink4,
        mapping_exprs_func=lambda coords: (
            coords[0] * sp.sinh(coords[1]),
            coords[0] * sp.cosh(coords[1]),
            coords[2],
            coords[3],
        ),
        inverse_exprs_func=lambda coords: (
            sp.sqrt(coords[1]**2 - coords[0]**2),
            sp.atanh(coords[0] / coords[1]),
            coords[2],
            coords[3],
        ),
        metadata={"standard_name": "Rindler->Minkowski", "symbolic_inverse_kind": "principal_branch"},
    ))

    register_map(CoordinateMap(
        source=mink4,
        target=rind4,
        mapping_exprs_func=lambda coords: (
            sp.sqrt(coords[1]**2 - coords[0]**2),
            sp.atanh(coords[0] / coords[1]),
            coords[2],
            coords[3],
        ),
        inverse_exprs_func=lambda coords: (
            coords[0] * sp.sinh(coords[1]),
            coords[0] * sp.cosh(coords[1]),
            coords[2],
            coords[3],
        ),
        metadata={"standard_name": "Minkowski->Rindler", "symbolic_inverse_kind": "principal_branch"},
    ))

    register_map(CoordinateMap(
        source=get_chart("Minkowski", "Cylindrical", 4),
        target=mink4,
        mapping_exprs_func=lambda coords: (
            coords[0],
            coords[1] * sp.cos(coords[2]),
            coords[1] * sp.sin(coords[2]),
            coords[3],
        ),
        inverse_exprs_func=lambda coords: (
            coords[0],
            sp.sqrt(coords[1]**2 + coords[2]**2),
            sp.atan2(coords[2], coords[1]),
            coords[3],
        ),
        metadata={"standard_name": "MinkowskiCylindrical->MinkowskiCartesian", "symbolic_inverse_kind": "principal_branch"},
    ))

    register_map(CoordinateMap(
        source=mink4,
        target=get_chart("Minkowski", "Cylindrical", 4),
        mapping_exprs_func=lambda coords: (
            coords[0],
            sp.sqrt(coords[1]**2 + coords[2]**2),
            sp.atan2(coords[2], coords[1]),
            coords[3],
        ),
        inverse_exprs_func=lambda coords: (
            coords[0],
            coords[1] * sp.cos(coords[2]),
            coords[1] * sp.sin(coords[2]),
            coords[3],
        ),
        metadata={"standard_name": "MinkowskiCartesian->MinkowskiCylindrical", "symbolic_inverse_kind": "principal_branch"},
    ))

    register_map(CoordinateMap(
        source=get_chart("Minkowski", "LightCone", 4),
        target=mink4,
        mapping_exprs_func=lambda coords: (
            (coords[0] + coords[1]) / 2,
            (coords[1] - coords[0]) / 2,
            coords[2],
            coords[3],
        ),
        inverse_exprs_func=lambda coords: (
            coords[0] - coords[1],
            coords[0] + coords[1],
            coords[2],
            coords[3],
        ),
        metadata={"standard_name": "LightCone->Minkowski", "symbolic_inverse_kind": "exact"},
    ))

    register_map(CoordinateMap(
        source=mink4,
        target=get_chart("Minkowski", "LightCone", 4),
        mapping_exprs_func=lambda coords: (
            coords[0] - coords[1],
            coords[0] + coords[1],
            coords[2],
            coords[3],
        ),
        inverse_exprs_func=lambda coords: (
            (coords[0] + coords[1]) / 2,
            (coords[1] - coords[0]) / 2,
            coords[2],
            coords[3],
        ),
        metadata={"standard_name": "Minkowski->LightCone", "symbolic_inverse_kind": "exact"},
    ))


def _bispherical_to_cartesian(coords, a):
    sigma, tau, phi = coords
    denom = sp.cosh(tau) - sp.cos(sigma)
    return (
        a * sp.sin(sigma) * sp.cos(phi) / denom,
        a * sp.sin(sigma) * sp.sin(phi) / denom,
        a * sp.sinh(tau) / denom,
    )


def _cartesian_to_bispherical(coords, a):
    x, y, z = coords
    rho = sp.sqrt(x**2 + y**2)
    rsq = x**2 + y**2 + z**2
    return (
        sp.atan2(2 * a * rho, rsq - a**2),
        sp.atanh(2 * a * z / (rsq + a**2)),
        sp.atan2(y, x),
    )


def _toroidal_to_cartesian(coords, a):
    tau, sigma, phi = coords
    denom = sp.cosh(tau) - sp.cos(sigma)
    return (
        a * sp.sinh(tau) * sp.cos(phi) / denom,
        a * sp.sinh(tau) * sp.sin(phi) / denom,
        a * sp.sin(sigma) / denom,
    )


def _cartesian_to_toroidal(coords, a):
    x, y, z = coords
    rho = sp.sqrt(x**2 + y**2)
    rsq = x**2 + y**2 + z**2
    return (
        sp.acosh((rsq + a**2) / (2 * a * rho)),
        sp.atan2(2 * a * z, rsq - a**2),
        sp.atan2(y, x),
    )


def _cartesian_to_oblate(coords, a):
    x, y, z = coords
    rho2 = x**2 + y**2
    alpha = rho2 / a**2 + z**2 / a**2 - 1
    beta = sp.sqrt(alpha**2 + 4 * z**2 / a**2)
    s = sp.simplify((alpha + beta) / 2)
    mu = sp.asinh(sp.sqrt(s))
    sin_nu = sp.simplify(z / (a * sp.sinh(mu)))
    cos_nu = sp.simplify(sp.sqrt(rho2) / (a * sp.cosh(mu)))
    nu = sp.atan2(sin_nu, cos_nu)
    phi = sp.atan2(y, x)
    return (sp.simplify(mu), sp.simplify(nu), phi)



def _bipolar_to_cartesian(coords, a):
    sigma, tau = coords
    denom = sp.cosh(tau) - sp.cos(sigma)
    return (
        a * sp.sinh(tau) / denom,
        a * sp.sin(sigma) / denom,
    )


def _cartesian_to_bipolar(coords, a):
    x, y = coords
    rsq = x**2 + y**2
    return (
        sp.atan2(2 * a * y, rsq - a**2),
        sp.atanh(2 * a * x / (rsq + a**2)),
    )


def _parabolic_cylindrical_to_cartesian(coords):
    u, v, z = coords
    return (
        (u**2 - v**2) / 2,
        u * v,
        z,
    )


def _cartesian_to_parabolic_cylindrical(coords):
    x, y, z = coords
    rho = sp.sqrt(x**2 + y**2)
    u = sp.sqrt(rho + x)
    v = sp.sign(y) * sp.sqrt(rho - x)
    return (u, v, z)


def _elliptic_cylindrical_to_cartesian(coords, a):
    mu, nu, z = coords
    return (
        a * sp.cosh(mu) * sp.cos(nu),
        a * sp.sinh(mu) * sp.sin(nu),
        z,
    )


def _cartesian_to_elliptic_cylindrical(coords, a):
    x, y, z = coords
    return (
        sp.acosh((sp.sqrt((x + a)**2 + y**2) + sp.sqrt((x - a)**2 + y**2)) / (2 * a)),
        sp.acos((sp.sqrt((x + a)**2 + y**2) - sp.sqrt((x - a)**2 + y**2)) / (2 * a)),
        z,
    )


def _conical_to_cartesian(coords, b, c):
    r, mu, nu = coords
    denom = b**2 - c**2
    return (
        r * mu * nu / (b * c),
        r * sp.sqrt((mu**2 - b**2) * (b**2 - nu**2) / (b**2 * denom)),
        r * sp.sqrt((mu**2 - c**2) * (nu**2 - c**2) / (c**2 * denom)),
    )


def _ellipsoidal_to_cartesian(coords, a, b, c):
    lam, mu, nu = coords
    A, B, C = a**2, b**2, c**2
    return (
        sp.sqrt(((lam - A) * (mu - A) * (nu - A)) / ((A - B) * (A - C))),
        sp.sqrt(((lam - B) * (mu - B) * (nu - B)) / ((B - A) * (B - C))),
        sp.sqrt(((lam - C) * (mu - C) * (nu - C)) / ((C - A) * (C - B))),
    )


def _sorted_roots_exact(poly, symbol):
    poly = sp.Poly(sp.expand(poly), symbol)
    return tuple(sp.RootOf(poly, idx) for idx in range(poly.degree()))


def _cartesian_to_conical(coords, b, c):
    x, y, z = coords
    r = sp.sqrt(x**2 + y**2 + z**2)
    t = sp.Symbol("t", real=True)
    poly = sp.expand(x**2 * (t - b**2) * (t - c**2) + y**2 * t * (t - c**2) + z**2 * t * (t - b**2))
    roots = _sorted_roots_exact(poly, t)
    if len(roots) != 2:
        roots = (sp.RootOf(poly, 0), sp.RootOf(poly, 1))
    small, large = roots[0], roots[1]
    return (
        r,
        sp.sqrt(sp.simplify(large)),
        sp.sqrt(sp.simplify(small)),
    )


def _cartesian_to_ellipsoidal(coords, a, b, c):
    x, y, z = coords
    s = sp.Symbol("s", real=True)
    A, B, C = a**2, b**2, c**2
    poly = sp.expand(
        x**2 * (s - B) * (s - C)
        + y**2 * (s - A) * (s - C)
        + z**2 * (s - A) * (s - B)
        - (s - A) * (s - B) * (s - C)
    )
    roots = sp.solve(sp.Eq(poly, 0), s)
    roots = tuple(roots) if len(roots) == 3 else _sorted_roots_exact(poly, s)
    return (
        sp.simplify(roots[2]),
        sp.simplify(roots[1]),
        sp.simplify(roots[0]),
    )
