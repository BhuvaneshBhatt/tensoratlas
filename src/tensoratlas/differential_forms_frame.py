from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

import sympy as sp

from .semantic_ir import TensorExpr, ir_node, scalar_ir, symbol_ir, gamma_object_ir, canonical_ir_key, to_tensor_expr


def _degree(expr: TensorExpr) -> int:
    if expr.kind == "zero":
        return int(expr.metadata.get("degree", 0))
    if expr.kind == "form:basis":
        return 1
    if expr.kind == "form:wedge":
        return int(expr.metadata.get("degree", sum(_degree(ch) for ch in expr.children)))
    return int(expr.metadata.get("degree", 0))


def _basis_labels(expr: TensorExpr) -> tuple[Any, ...]:
    if expr.kind == "form:basis":
        return (expr.payload,)
    if expr.kind == "form:wedge":
        labels: list[Any] = []
        for ch in expr.children:
            labels.extend(_basis_labels(ch))
        return tuple(labels)
    return tuple(expr.metadata.get("basis_labels", ()))


def _permutation_parity(values: Sequence[Any], key=repr) -> int:
    ordered = sorted(range(len(values)), key=lambda i: key(values[i]))
    pos = {old: new for new, old in enumerate(ordered)}
    perm = [pos[i] for i in range(len(values))]
    inversions = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inversions += 1
    return -1 if inversions % 2 else 1


def _zero_form(degree: int = 0) -> TensorExpr:
    return ir_node("zero", degree=degree, provenance={"origin": "differential_forms"})


def basis_one_form(label: Any, *, frame: str | None = None) -> TensorExpr:
    """Create a coframe basis one-form."""
    return ir_node(
        "form:basis",
        payload=str(label),
        degree=1,
        frame=frame,
        basis_labels=(str(label),),
        provenance={"origin": "coframe"},
    )


def form_expr(name: str, degree: int, *, frame: str | None = None, components: Any = None) -> TensorExpr:
    return ir_node(
        "form:abstract",
        payload=str(name),
        degree=int(degree),
        frame=frame,
        components=components,
        provenance={"origin": "differential_form"},
    )


def canonicalize_wedge(expr: TensorExpr) -> TensorExpr:
    """Canonicalize a wedge product with graded signs and duplicate one-form annihilation."""
    expr = to_tensor_expr(expr)
    if expr.kind != "form:wedge" and expr.kind != "wedge":
        return expr
    factors: list[TensorExpr] = []
    sign = sp.Integer(expr.metadata.get("coefficient", 1))
    for child in expr.children:
        child = canonicalize_wedge(child)
        if child.kind == "zero":
            return _zero_form(_degree(expr))
        if child.kind in {"form:wedge", "wedge"}:
            sign *= sp.Integer(child.metadata.get("coefficient", 1))
            factors.extend(child.children)
        else:
            factors.append(child)
    if not factors:
        return scalar_ir(sign)
    one_labels = []
    for f in factors:
        if _degree(f) % 2 == 1:
            labels = _basis_labels(f)
            if len(labels) == 1:
                one_labels.append(labels[0])
    if len(one_labels) != len(set(one_labels)):
        return _zero_form(sum(_degree(f) for f in factors))
    # Graded ordering: swapping degrees p and q contributes (-1)^(pq).
    decorated = list(factors)
    swaps = 0
    for i in range(len(decorated)):
        for j in range(i + 1, len(decorated)):
            if repr(canonical_ir_key(decorated[i])) > repr(canonical_ir_key(decorated[j])):
                swaps += _degree(decorated[i]) * _degree(decorated[j])
    ordered = tuple(sorted(decorated, key=lambda f: repr(canonical_ir_key(f))))
    if swaps % 2:
        sign = -sign
    if len(ordered) == 1 and sign == 1:
        return ordered[0]
    return ir_node(
        "form:wedge",
        *ordered,
        degree=sum(_degree(f) for f in ordered),
        coefficient=sp.simplify(sign),
        basis_labels=tuple(label for f in ordered for label in _basis_labels(f)),
        provenance={"origin": "wedge_canonicalization"},
    )


def wedge_forms(*forms: Any) -> TensorExpr:
    return canonicalize_wedge(ir_node("form:wedge", *(to_tensor_expr(f) for f in forms), provenance={"origin": "wedge"}))


@dataclass(frozen=True)
class FrameCalculusPolicy:
    dimension: int
    signature: tuple[int, ...] | None = None
    orientation: str = "positive"
    frame: str = "e"
    coframe: str = "theta"

    @property
    def metric_sign_count(self) -> int:
        if self.signature is None:
            return 0
        return sum(1 for s in self.signature if int(s) < 0)

    @property
    def orientation_sign(self) -> int:
        return -1 if self.orientation.lower() in {"negative", "reversed", "-"} else 1


def hodge_star_form(expr: Any, *, policy: FrameCalculusPolicy) -> TensorExpr:
    expr = canonicalize_wedge(to_tensor_expr(expr))
    labels = _basis_labels(expr)
    degree = _degree(expr)
    all_labels = tuple(f"{policy.coframe}{i}" for i in range(policy.dimension))
    if not labels:
        complement = all_labels
    else:
        complement = tuple(x for x in all_labels if x not in labels)
    if len(set(labels)) != len(labels):
        return _zero_form(policy.dimension - degree)
    combined = labels + complement
    sign = _permutation_parity(combined) * policy.orientation_sign
    # Pseudo-Riemannian Hodge: include one metric-sign factor for each timelike basis vector in the input slot.
    if policy.signature is not None:
        label_to_sign = {all_labels[i]: int(policy.signature[i]) for i in range(min(policy.dimension, len(policy.signature)))}
        for label in labels:
            sign *= label_to_sign.get(label, 1)
    result = wedge_forms(*(basis_one_form(l, frame=policy.coframe) for l in complement)) if complement else scalar_ir(1)
    return result.with_metadata(coefficient=sp.simplify(sign * sp.sympify(result.metadata.get("coefficient", 1))), hodge_dual_of=canonical_ir_key(expr), degree=policy.dimension - degree)


def exterior_covariant_derivative(expr: Any, *, connection: str = "CD") -> TensorExpr:
    expr = to_tensor_expr(expr)
    return ir_node("form:exterior_covariant_derivative", expr, connection=connection, degree=_degree(expr) + 1, provenance={"origin": "exterior_covariant_derivative"})


def frame_vector(label: Any, *, frame: str = "e") -> TensorExpr:
    return ir_node("frame:vector", payload=str(label), frame=frame, provenance={"origin": "frame"})


def coframe_one_form(label: Any, *, coframe: str = "theta") -> TensorExpr:
    return basis_one_form(f"{coframe}{label}", frame=coframe)


def frame_to_coframe(expr: Any, *, frame: str = "e", coframe: str = "theta") -> TensorExpr:
    expr = to_tensor_expr(expr)
    if expr.kind == "frame:vector":
        label = str(expr.payload)
        if label.startswith(frame):
            label = label[len(frame):]
        return coframe_one_form(label, coframe=coframe).with_metadata(converted_from="frame")
    return ir_node("frame:to_coframe", expr, frame=frame, coframe=coframe, provenance={"origin": "frame_conversion"})


def coframe_to_frame(expr: Any, *, frame: str = "e", coframe: str = "theta") -> TensorExpr:
    expr = to_tensor_expr(expr)
    if expr.kind == "form:basis":
        label = str(expr.payload)
        if label.startswith(coframe):
            label = label[len(coframe):]
        return frame_vector(label, frame=frame).with_metadata(converted_from="coframe")
    return ir_node("coframe:to_frame", expr, frame=frame, coframe=coframe, provenance={"origin": "frame_conversion"})


def connection_one_form(connection: str, upper: Any, lower: Any, *, coframe: str = "theta") -> TensorExpr:
    return ir_node("connection:one_form", payload=str(connection), upper=str(upper), lower=str(lower), degree=1, coframe=coframe, provenance={"origin": "connection_one_form"})


def spin_connection_one_form(name: str = "omega", *, upper: Any = "a", lower: Any = "b", coframe: str = "theta") -> TensorExpr:
    return ir_node("spin:connection_one_form", payload=str(name), upper=str(upper), lower=str(lower), degree=1, coframe=coframe, antisymmetric_pair=(str(upper), str(lower)), provenance={"origin": "spin_connection"})


def torsion_two_form(connection: str, index: Any, *, coframe: str = "theta") -> TensorExpr:
    return ir_node("torsion:two_form", payload=str(connection), index=str(index), degree=2, coframe=coframe, provenance={"origin": "torsion_two_form"})


def curvature_two_form(connection: str, upper: Any, lower: Any, *, coframe: str = "theta") -> TensorExpr:
    return ir_node("curvature:two_form", payload=str(connection), upper=str(upper), lower=str(lower), degree=2, coframe=coframe, provenance={"origin": "curvature_two_form"})


def first_cartan_structure_equation(index: Any, *, connection: str = "omega", coframe: str = "theta") -> TensorExpr:
    theta = coframe_one_form(index, coframe=coframe)
    terms = [exterior_covariant_derivative(theta, connection="d")]
    for j in (0, 1):
        terms.append(wedge_forms(connection_one_form(connection, index, j, coframe=coframe), coframe_one_form(j, coframe=coframe)))
    return ir_node("cartan:first_structure_equation", *terms, equals=torsion_two_form(connection, index, coframe=coframe), provenance={"origin": "cartan"})


def second_cartan_structure_equation(upper: Any, lower: Any, *, connection: str = "omega", coframe: str = "theta") -> TensorExpr:
    omega = connection_one_form(connection, upper, lower, coframe=coframe)
    terms = [exterior_covariant_derivative(omega, connection="d")]
    for j in (0, 1):
        terms.append(wedge_forms(connection_one_form(connection, upper, j, coframe=coframe), connection_one_form(connection, j, lower, coframe=coframe)))
    return ir_node("cartan:second_structure_equation", *terms, equals=curvature_two_form(connection, upper, lower, coframe=coframe), provenance={"origin": "cartan"})


def gamma_product(*gammas: Any, dimension: int | None = None, signature: Any = None) -> TensorExpr:
    return ir_node("gamma:product", *(to_tensor_expr(g) for g in gammas), dimension=dimension, signature=signature, provenance={"origin": "gamma_product"})


def simplify_gamma_product(expr: Any, *, metric: str = "g") -> TensorExpr:
    expr = to_tensor_expr(expr)
    if expr.kind != "gamma:product":
        return expr
    children = list(expr.children)
    out: list[TensorExpr] = []
    contractions: list[TensorExpr] = []
    i = 0
    while i < len(children):
        current = children[i]
        if i + 1 < len(children):
            nxt = children[i + 1]
            if current.kind == "gamma_object" and nxt.kind == "gamma_object":
                a = tuple(current.metadata.get("indices", ()))
                b = tuple(nxt.metadata.get("indices", ()))
                if a and b and a[0] == b[0]:
                    contractions.append(ir_node("metric:trace", payload=metric, index=a[0], provenance={"origin": "clifford_simplification"}))
                    i += 2
                    continue
        out.append(current)
        i += 1
    if not out and len(contractions) == 1:
        return contractions[0]
    return ir_node("gamma:product", *(tuple(out) + tuple(contractions)), simplified=True, metric=metric, provenance={"origin": "clifford_simplification"})


def gamma_anticommutator_expr(index_a: Any, index_b: Any, *, metric: str = "g", dimension: int | None = None, signature: Any = None) -> TensorExpr:
    ga = gamma_object_ir("gamma", indices=(index_a,), dimension=dimension, signature=signature)
    gb = gamma_object_ir("gamma", indices=(index_b,), dimension=dimension, signature=signature)
    return ir_node("gamma:anticommutator", ga, gb, equals=ir_node("metric:component", payload=metric, indices=(index_a, index_b), coefficient=2), provenance={"origin": "clifford_relation"})


def dirac_operator_expr(spinor: Any, *, gamma_indices: Sequence[Any] = ("a",), connection: str = "spinCD", dimension: int | None = None, signature: Any = None) -> TensorExpr:
    spinor_expr = to_tensor_expr(spinor) if not isinstance(spinor, str) else ir_node("spinor:field", payload=spinor, provenance={"origin": "spinor"})
    terms = []
    for idx in gamma_indices:
        gamma = gamma_object_ir("gamma", indices=(idx,), dimension=dimension, signature=signature)
        deriv = ir_node("spin:covariant_derivative", spinor_expr, index=idx, connection=connection, provenance={"origin": "dirac_operator"})
        terms.append(ir_node("mul", gamma, deriv, provenance={"origin": "dirac_operator"}))
    return ir_node("dirac:operator", *terms, connection=connection, dimension=dimension, signature=signature, provenance={"origin": "dirac_operator"})


def lichnerowicz_example(spinor: str = "psi", *, scalar_curvature: str = "R", connection: str = "spinCD") -> TensorExpr:
    psi = ir_node("spinor:field", payload=spinor, provenance={"origin": "spinor"})
    lap = ir_node("spin:laplacian", psi, connection=connection, provenance={"origin": "dirac_example"})
    curv = ir_node("mul", scalar_ir(sp.Rational(1, 4)), ir_node("curvature:scalar_action", psi, payload=scalar_curvature, provenance={"origin": "dirac_example"}))
    return ir_node("dirac:lichnerowicz", lap, curv, identity="D^2 psi = nabla^*nabla psi + R psi/4", provenance={"origin": "dirac_example"})


__all__ = [
    "FrameCalculusPolicy",
    "basis_one_form",
    "form_expr",
    "canonicalize_wedge",
    "wedge_forms",
    "hodge_star_form",
    "exterior_covariant_derivative",
    "frame_vector",
    "coframe_one_form",
    "frame_to_coframe",
    "coframe_to_frame",
    "connection_one_form",
    "spin_connection_one_form",
    "torsion_two_form",
    "curvature_two_form",
    "first_cartan_structure_equation",
    "second_cartan_structure_equation",
    "gamma_product",
    "simplify_gamma_product",
    "gamma_anticommutator_expr",
    "dirac_operator_expr",
    "lichnerowicz_example",
]
