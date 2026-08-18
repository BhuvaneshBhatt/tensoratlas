from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .declarations import (
    CurvatureConventionPolicy,
    DeclarationRegistry,
    IndexVariance,
    TensorSymmetryDeclaration,
)
from .semantic_ir import TensorExpr, canonical_ir_key, covariant_derivative_ir, ir_node
from .tensor_expr_canonicalization import (
    TensorExprCanonicalizationReport,
    canonicalize_tensor_expr,
)


_CURVATURE_NAMES = {
    "Riemann": "Riemann",
    "Ricci": "Ricci",
    "ScalarCurvature": "ScalarCurvature",
    "Weyl": "Weyl",
    "Schouten": "Schouten",
    "Einstein": "Einstein",
}


@dataclass(frozen=True)
class ConnectionProfile:
    name: str
    kind: str
    metric: str | None
    torsion_free: bool
    metric_compatible: bool
    has_torsion: bool
    has_nonmetricity: bool
    curvature_policy: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeometryCanonicalizationReport:
    original: TensorExpr
    reduced: TensorExpr
    canonical_report: TensorExprCanonicalizationReport
    applied_identities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def canonical(self) -> TensorExpr:
        return self.canonical_report.canonical

    @property
    def canonical_key(self) -> tuple[Any, ...]:
        return self.canonical_report.canonical_key


@dataclass(frozen=True)
class ConventionConversionReport:
    original: TensorExpr
    converted: TensorExpr
    source_policy: str
    target_policy: str
    sign_factor: int
    index_permutation: tuple[int, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry / declarations


def connection_profile(registry: DeclarationRegistry, connection: str) -> ConnectionProfile:
    conn = registry.require_connection(connection)
    torsion_free = conn.is_torsion_free()
    metric_compatible = conn.is_metric_compatible()
    if conn.metric and torsion_free and metric_compatible:
        kind = "levi_civita"
    elif not torsion_free and not metric_compatible:
        kind = "torsionful_nonmetric_affine"
    elif not torsion_free:
        kind = "torsionful_affine"
    elif not metric_compatible:
        kind = "nonmetric_affine"
    else:
        kind = "affine"
    return ConnectionProfile(
        name=connection,
        kind=kind,
        metric=conn.metric,
        torsion_free=torsion_free,
        metric_compatible=metric_compatible,
        has_torsion=not torsion_free,
        has_nonmetricity=bool(conn.metric and not metric_compatible),
        curvature_policy=conn.curvature_policy,
        metadata={"bundle": conn.bundle, "manifold": conn.manifold},
    )


def _tensor_name(family: str, connection: str) -> str:
    return f"{family}[{connection}]"


def _has_tensor(registry: DeclarationRegistry, name: str) -> bool:
    return name in registry.tensors or name in registry.metrics


def ensure_connection_curvature_declarations(registry: DeclarationRegistry, connection: str) -> DeclarationRegistry:
    """Declare standard curvature/torsion/nonmetricity tensor heads for a connection.

    The function is deliberately additive and idempotent: existing declarations
    are left untouched.  It gives the central TensorExpr canonicalizer enough
    slot-symmetry and variance data to handle curvature expressions without
    curvature-specific ad hoc keys.
    """

    conn = registry.require_connection(connection)
    bundle = conn.bundle
    cov = IndexVariance.COVARIANT
    con = IndexVariance.CONTRAVARIANT
    reg = registry

    riem = _tensor_name("Riemann", connection)
    if not _has_tensor(reg, riem):
        reg = reg.declare_tensor(
            riem,
            (bundle, bundle, bundle, bundle),
            (con, cov, cov, cov),
            symmetries=(TensorSymmetryDeclaration("antisymmetric", (2, 3)),),
            role="riemann_curvature",
        )

    ricci = _tensor_name("Ricci", connection)
    if not _has_tensor(reg, ricci):
        syms = (TensorSymmetryDeclaration("symmetric", (0, 1)),) if conn.is_torsion_free() and conn.is_metric_compatible() else ()
        reg = reg.declare_tensor(ricci, (bundle, bundle), (cov, cov), symmetries=syms, role="ricci_curvature")

    for family, role in (("Weyl", "weyl_curvature"),):
        name = _tensor_name(family, connection)
        if not _has_tensor(reg, name):
            reg = reg.declare_tensor(
                name,
                (bundle, bundle, bundle, bundle),
                (con, cov, cov, cov),
                symmetries=(TensorSymmetryDeclaration("antisymmetric", (2, 3)),),
                role=role,
            )

    for family, role in (("Schouten", "schouten_tensor"), ("Einstein", "einstein_tensor")):
        name = _tensor_name(family, connection)
        if not _has_tensor(reg, name):
            syms = (TensorSymmetryDeclaration("symmetric", (0, 1)),) if conn.is_torsion_free() and conn.is_metric_compatible() else ()
            reg = reg.declare_tensor(name, (bundle, bundle), (cov, cov), symmetries=syms, role=role)

    torsion = _tensor_name("Torsion", connection)
    if not _has_tensor(reg, torsion):
        reg = reg.declare_tensor(
            torsion,
            (bundle, bundle, bundle),
            (con, cov, cov),
            symmetries=(TensorSymmetryDeclaration("antisymmetric", (1, 2)),),
            role="torsion",
        )

    if conn.metric is not None:
        nonmetricity = _tensor_name("Nonmetricity", connection)
        if not _has_tensor(reg, nonmetricity):
            reg = reg.declare_tensor(
                nonmetricity,
                (bundle, bundle, bundle),
                (cov, cov, cov),
                symmetries=(TensorSymmetryDeclaration("symmetric", (1, 2)),),
                role="nonmetricity",
            )

    return reg


# ---------------------------------------------------------------------------
# TensorExpr constructors


def _dimension(registry: DeclarationRegistry, connection: str) -> int:
    conn = registry.require_connection(connection)
    return registry.require_manifold(conn.manifold).dimension


def _policy(registry: DeclarationRegistry, connection: str) -> CurvatureConventionPolicy:
    conn = registry.require_connection(connection)
    return registry.curvature_policies[conn.curvature_policy]


def curvature_tensor_expr(family: str, registry: DeclarationRegistry, connection: str, indices: Sequence[Any] = ()) -> TensorExpr:
    reg = ensure_connection_curvature_declarations(registry, connection)
    dim = _dimension(reg, connection)
    policy = _policy(reg, connection)
    if family == "ScalarCurvature":
        return ir_node(
            "curvature_scalar",
            payload=_tensor_name(family, connection),
            family=family,
            connection=connection,
            dimension=dim,
            convention=policy.name,
            provenance={"origin": "connection_curvature"},
        )
    return reg.tensor_expr(_tensor_name(family, connection), tuple(indices)).with_metadata(
        family=family,
        connection=connection,
        dimension=dim,
        convention=policy.name,
        curvature_role=family,
    )


def riemann_tensor_expr(registry: DeclarationRegistry, connection: str, indices: Sequence[Any]) -> TensorExpr:
    return curvature_tensor_expr("Riemann", registry, connection, indices)


def ricci_tensor_expr(registry: DeclarationRegistry, connection: str, indices: Sequence[Any]) -> TensorExpr:
    return curvature_tensor_expr("Ricci", registry, connection, indices)


def scalar_curvature_tensor_expr(registry: DeclarationRegistry, connection: str) -> TensorExpr:
    return curvature_tensor_expr("ScalarCurvature", registry, connection, ())


def weyl_tensor_expr(registry: DeclarationRegistry, connection: str, indices: Sequence[Any]) -> TensorExpr:
    return curvature_tensor_expr("Weyl", registry, connection, indices)


def schouten_tensor_expr(registry: DeclarationRegistry, connection: str, indices: Sequence[Any]) -> TensorExpr:
    return curvature_tensor_expr("Schouten", registry, connection, indices)


def einstein_tensor_expr(registry: DeclarationRegistry, connection: str, indices: Sequence[Any]) -> TensorExpr:
    return curvature_tensor_expr("Einstein", registry, connection, indices)


def torsion_tensor_expr(registry: DeclarationRegistry, connection: str, indices: Sequence[Any]) -> TensorExpr:
    reg = ensure_connection_curvature_declarations(registry, connection)
    return reg.tensor_expr(_tensor_name("Torsion", connection), tuple(indices)).with_metadata(
        family="Torsion",
        connection=connection,
        dimension=_dimension(reg, connection),
    )


def nonmetricity_tensor_expr(registry: DeclarationRegistry, connection: str, indices: Sequence[Any]) -> TensorExpr:
    reg = ensure_connection_curvature_declarations(registry, connection)
    return reg.tensor_expr(_tensor_name("Nonmetricity", connection), tuple(indices)).with_metadata(
        family="Nonmetricity",
        connection=connection,
        dimension=_dimension(reg, connection),
    )


def covariant_derivative_tensor_expr(registry: DeclarationRegistry, connection: str, derivative_index: Any, operand: TensorExpr) -> TensorExpr:
    return covariant_derivative_ir(operand, index=derivative_index, connection=connection)


# ---------------------------------------------------------------------------
# Identities and executable reductions


def first_bianchi_identity_ir(registry: DeclarationRegistry, connection: str, indices: Sequence[Any]) -> TensorExpr:
    """Return the algebraic first-Bianchi cyclic sum R^a_{[bcd]}."""

    a, b, c, d = tuple(indices)
    return ir_node(
        "add",
        riemann_tensor_expr(registry, connection, (a, b, c, d)),
        riemann_tensor_expr(registry, connection, (a, c, d, b)),
        riemann_tensor_expr(registry, connection, (a, d, b, c)),
        identity="first_bianchi",
        connection=connection,
        provenance={"origin": "connection_curvature"},
    )


def second_bianchi_identity_ir(registry: DeclarationRegistry, connection: str, indices: Sequence[Any]) -> TensorExpr:
    """Return the differential second-Bianchi cyclic sum ∇_[e R^a_|b|cd]."""

    e, a, b, c, d = tuple(indices)
    return ir_node(
        "add",
        covariant_derivative_tensor_expr(registry, connection, e, riemann_tensor_expr(registry, connection, (a, b, c, d))),
        covariant_derivative_tensor_expr(registry, connection, c, riemann_tensor_expr(registry, connection, (a, b, d, e))),
        covariant_derivative_tensor_expr(registry, connection, d, riemann_tensor_expr(registry, connection, (a, b, e, c))),
        identity="second_bianchi",
        connection=connection,
        provenance={"origin": "connection_curvature"},
    )


def torsion_identity_ir(registry: DeclarationRegistry, connection: str, indices: Sequence[Any]) -> TensorExpr:
    return torsion_tensor_expr(registry, connection, indices).with_metadata(identity="torsion_free" if registry.require_connection(connection).is_torsion_free() else "torsion")


def nonmetricity_identity_ir(registry: DeclarationRegistry, connection: str, indices: Sequence[Any]) -> TensorExpr:
    return nonmetricity_tensor_expr(registry, connection, indices).with_metadata(identity="metric_compatibility" if registry.require_connection(connection).is_metric_compatible() else "nonmetricity")


def covariant_derivative_commutator_ir(
    registry: DeclarationRegistry,
    connection: str,
    left_index: Any,
    right_index: Any,
    operand: TensorExpr,
) -> TensorExpr:
    conn = registry.require_connection(connection)
    policy = registry.curvature_policies[conn.curvature_policy]
    return ir_node(
        "covariant_derivative_commutator",
        operand,
        payload=f"[{connection}_{left_index},{connection}_{right_index}]",
        left_index=left_index,
        right_index=right_index,
        connection=connection,
        curvature="Riemann",
        sign=policy.commutator_sign,
        torsion_term=not conn.is_torsion_free(),
        nonmetricity_term=not conn.is_metric_compatible(),
        convention=policy.name,
        provenance={"origin": "connection_curvature"},
    )


def expand_covariant_derivative_commutator(ir: TensorExpr, registry: DeclarationRegistry) -> TensorExpr:
    if ir.kind != "covariant_derivative_commutator" or not ir.children:
        return ir
    connection = str(ir.metadata["connection"])
    left = ir.metadata["left_index"]
    right = ir.metadata["right_index"]
    sign = sp.Integer(ir.metadata.get("sign", 1))
    operand = ir.children[0]
    terms: list[TensorExpr] = []
    # For this executable symbolic layer, represent the curvature action by a
    # curvature-action node; bundle-specific index insertion is handled by later
    # abstract/indexed adapters.
    terms.append(
        ir_node(
            "curvature_action",
            riemann_tensor_expr(registry, connection, ("_out", "_in", left, right)),
            operand,
            coefficient=sign,
            connection=connection,
            provenance={"origin": "connection_curvature"},
        )
    )
    if ir.metadata.get("torsion_term"):
        terms.append(
            ir_node(
                "torsion_commutator_term",
                torsion_tensor_expr(registry, connection, ("_k", left, right)),
                covariant_derivative_tensor_expr(registry, connection, "_k", operand),
                coefficient=-sign,
                connection=connection,
                provenance={"origin": "connection_curvature"},
            )
        )
    if ir.metadata.get("nonmetricity_term"):
        terms.append(
            ir_node(
                "nonmetricity_commutator_marker",
                operand,
                coefficient=sign,
                connection=connection,
                provenance={"origin": "connection_curvature"},
            )
        )
    return ir_node("add", *terms, connection=connection, expanded_from="covariant_derivative_commutator")


def curvature_decomposition_ir(registry: DeclarationRegistry, connection: str, indices: Sequence[Any]) -> TensorExpr:
    """Return a dimension-aware symbolic Riemann decomposition.

    The result is intentionally a TensorExpr expression rather than a rendered
    formula so it can be passed through the central canonicalizer.  The payload
    records the standard formula family:
      n >= 4: Riemann = Weyl + Schouten wedge metric
      n == 3: Weyl vanishes; Riemann is Schouten/metric part
      n == 2: Riemann is scalar/metric part
    """

    dim = _dimension(registry, connection)
    a, b, c, d = tuple(indices)
    if dim <= 2:
        return ir_node(
            "curvature_decomposition",
            scalar_curvature_tensor_expr(registry, connection),
            payload="riemann_scalar_metric_decomposition",
            family="Riemann",
            target="ScalarCurvatureMetricPart",
            dimension=dim,
            connection=connection,
        )
    schouten_part = ir_node(
        "schouten_metric_wedge",
        schouten_tensor_expr(registry, connection, (b, d)),
        schouten_tensor_expr(registry, connection, (b, c)),
        payload="g_wedge_P",
        dimension=dim,
        connection=connection,
    )
    if dim == 3:
        return ir_node(
            "curvature_decomposition",
            schouten_part,
            payload="riemann_schouten_decomposition",
            family="Riemann",
            target="SchoutenMetricPart",
            dimension=dim,
            connection=connection,
        )
    return ir_node(
        "curvature_decomposition",
        weyl_tensor_expr(registry, connection, (a, b, c, d)),
        schouten_part,
        payload="riemann_weyl_schouten_decomposition",
        family="Riemann",
        target="WeylPlusSchoutenMetricPart",
        dimension=dim,
        connection=connection,
    )


def _is_curvature_family(ir: TensorExpr, family: str) -> bool:
    return ir.metadata.get("family") == family or ir.metadata.get("curvature_role") == family


def _same_cyclic_riemann_terms(children: Sequence[TensorExpr]) -> bool:
    if len(children) != 3 or not all(_is_curvature_family(ch, "Riemann") for ch in children):
        return False
    inds = [tuple(ch.metadata.get("indices", ())) for ch in children]
    if any(len(x) != 4 for x in inds):
        return False
    a, b, c, d = inds[0]
    want = {(a, b, c, d), (a, c, d, b), (a, d, b, c)}
    return set(inds) == want


def _same_second_bianchi_terms(children: Sequence[TensorExpr]) -> bool:
    if len(children) != 3 or not all(ch.kind == "covariant_derivative" and ch.children for ch in children):
        return False
    inners = [ch.children[0] for ch in children]
    return all(_is_curvature_family(inner, "Riemann") for inner in inners)


def reduce_connection_curvature_identities(ir: TensorExpr, registry: DeclarationRegistry) -> tuple[TensorExpr, tuple[str, ...]]:
    """Apply executable connection/curvature identities before canonicalization."""

    applied: list[str] = []

    def rec(node: TensorExpr) -> TensorExpr:
        children = tuple(rec(ch) for ch in node.children)
        cur = TensorExpr(node.kind, node.payload, children, dict(node.metadata), node.provenance)
        connection = cur.metadata.get("connection")
        if isinstance(connection, str) and connection in registry.connections:
            conn = registry.connections[connection]
            dim = _dimension(registry, connection)
            if _is_curvature_family(cur, "Weyl") and dim < 4:
                applied.append("dimension_dependent_weyl_zero")
                return ir_node("zero", family="Weyl", dimension=dim, connection=connection)
            if _is_curvature_family(cur, "Torsion") and conn.is_torsion_free():
                applied.append("torsion_free_connection")
                return ir_node("zero", family="Torsion", connection=connection)
            if _is_curvature_family(cur, "Nonmetricity") and conn.is_metric_compatible():
                applied.append("metric_compatible_connection")
                return ir_node("zero", family="Nonmetricity", connection=connection)
        if cur.kind == "add":
            if cur.metadata.get("identity") == "first_bianchi" or _same_cyclic_riemann_terms(cur.children):
                conn_name = str(cur.metadata.get("connection", cur.children[0].metadata.get("connection", "")))
                if conn_name in registry.connections and registry.connections[conn_name].is_torsion_free():
                    applied.append("first_bianchi")
                    return ir_node("zero", identity="first_bianchi", connection=conn_name)
            if cur.metadata.get("identity") == "second_bianchi" or _same_second_bianchi_terms(cur.children):
                conn_name = str(cur.metadata.get("connection", cur.children[0].metadata.get("connection", "")))
                if conn_name in registry.connections:
                    applied.append("second_bianchi")
                    return ir_node("zero", identity="second_bianchi", connection=conn_name)
        if cur.kind == "covariant_derivative" and cur.children:
            child = cur.children[0]
            if child.kind == "indexed_tensor" and child.payload in registry.metrics:
                conn_name = cur.metadata.get("connection")
                if isinstance(conn_name, str) and conn_name in registry.connections and registry.connections[conn_name].is_metric_compatible():
                    applied.append("metric_compatibility_covariant_derivative")
                    return ir_node("zero", identity="metric_compatibility", connection=conn_name)
        if cur.kind == "covariant_derivative_commutator":
            applied.append("covariant_derivative_commutator")
            return rec(expand_covariant_derivative_commutator(cur, registry))
        return cur

    reduced = rec(ir)
    return reduced, tuple(dict.fromkeys(applied))


def _registry_with_needed_curvature_declarations(registry: DeclarationRegistry, ir: TensorExpr) -> DeclarationRegistry:
    reg = registry
    seen: set[str] = set()

    def visit(node: TensorExpr) -> None:
        conn = node.metadata.get("connection")
        if isinstance(conn, str) and conn in registry.connections and conn not in seen:
            seen.add(conn)
        for child in node.children:
            visit(child)

    visit(ir)
    for conn in seen or set(registry.connections):
        reg = ensure_connection_curvature_declarations(reg, conn)
    return reg


def canonicalize_geometry_ir(obj: TensorExpr, registry: DeclarationRegistry) -> GeometryCanonicalizationReport:
    original = obj
    working_registry = _registry_with_needed_curvature_declarations(registry, obj)
    reduced, identities = reduce_connection_curvature_identities(obj, working_registry)
    working_registry = _registry_with_needed_curvature_declarations(working_registry, reduced)
    canonical = canonicalize_tensor_expr(reduced, registry=working_registry)
    return GeometryCanonicalizationReport(original, reduced, canonical, identities)


# ---------------------------------------------------------------------------
# Convention-aware sign/order conversion


def convert_curvature_convention_ir(
    ir: TensorExpr,
    *,
    source: CurvatureConventionPolicy,
    target: CurvatureConventionPolicy,
) -> ConventionConversionReport:
    md = dict(ir.metadata)
    indices = tuple(md.get("indices", ()))
    sign_factor = 1
    if source.commutator_sign != target.commutator_sign or source.sign != target.sign:
        sign_factor = -1
        md["coefficient"] = sp.sympify(md.get("coefficient", 1)) * sign_factor
    permutation: tuple[int, ...] | None = None
    if indices and source.riemann_slot_order != target.riemann_slot_order and len(indices) == len(source.riemann_slot_order) == len(target.riemann_slot_order):
        source_positions = {slot: pos for pos, slot in enumerate(source.riemann_slot_order)}
        permutation = tuple(source_positions[slot] for slot in target.riemann_slot_order)
        md["indices"] = tuple(indices[pos] for pos in permutation)
        md["riemann_slot_order"] = target.riemann_slot_order
    md["convention"] = target.name
    converted = TensorExpr(ir.kind, ir.payload, ir.children, md, ir.provenance.append("curvature_convention_conversion", source="connection_curvature", before_key=canonical_ir_key(ir), after_key=None, source_policy=source.name, target_policy=target.name))
    return ConventionConversionReport(ir, converted, source.name, target.name, sign_factor, permutation)


__all__ = [
    "ConnectionProfile",
    "GeometryCanonicalizationReport",
    "ConventionConversionReport",
    "connection_profile",
    "ensure_connection_curvature_declarations",
    "curvature_tensor_expr",
    "riemann_tensor_expr",
    "ricci_tensor_expr",
    "scalar_curvature_tensor_expr",
    "weyl_tensor_expr",
    "schouten_tensor_expr",
    "einstein_tensor_expr",
    "torsion_tensor_expr",
    "nonmetricity_tensor_expr",
    "covariant_derivative_tensor_expr",
    "first_bianchi_identity_ir",
    "second_bianchi_identity_ir",
    "torsion_identity_ir",
    "nonmetricity_identity_ir",
    "covariant_derivative_commutator_ir",
    "expand_covariant_derivative_commutator",
    "curvature_decomposition_ir",
    "reduce_connection_curvature_identities",
    "canonicalize_geometry_ir",
    "convert_curvature_convention_ir",
]
