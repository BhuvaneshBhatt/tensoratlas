from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp

from .semantic_ir import TensorExpr, IRProvenance, canonical_ir_key, ir_node, to_tensor_expr
try:  # declarations are optional to keep this module import-cheap and usable alone
    from .declarations import DeclarationRegistry, TensorSymmetryDeclaration
except Exception:  # pragma: no cover - defensive for partial imports
    DeclarationRegistry = Any  # type: ignore
    TensorSymmetryDeclaration = Any  # type: ignore


@dataclass(frozen=True)
class CanonicalizationStep:
    rule: str
    before_key: tuple[Any, ...]
    after_key: tuple[Any, ...]
    detail: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfluenceDiagnostic:
    issue: str
    severity: str = "info"
    path_a: tuple[str, ...] = ()
    path_b: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorExprCanonicalizationReport:
    original: TensorExpr
    canonical: TensorExpr
    canonical_key: tuple[Any, ...]
    steps: tuple[CanonicalizationStep, ...] = ()
    diagnostics: tuple[ConfluenceDiagnostic, ...] = ()

    @property
    def changed(self) -> bool:
        return canonical_ir_key(self.original) != self.canonical_key


@dataclass(frozen=True)
class CanonicalizationPolicy:
    rename_dummies: bool = True
    apply_slot_symmetry: bool = True
    apply_identities: bool = True
    normalize_metric_delta_epsilon: bool = True
    order_products: bool = True
    order_covariant_derivatives: bool = True
    check_confluence: bool = True
    max_passes: int = 8


@dataclass(frozen=True)
class SlotPermutationRule:
    """A signed generator acting on tensor slots."""

    permutation: tuple[int, ...]
    sign: int = 1
    source: str = "declaration"

    def __post_init__(self) -> None:
        if self.sign not in {-1, 1}:
            raise ValueError("slot-permutation signs must be ±1")
        if sorted(self.permutation) != list(range(len(self.permutation))):
            raise ValueError("slot permutation must be a permutation of 0..n-1")


@dataclass(frozen=True)
class SlotOrbitElement:
    permutation: tuple[int, ...]
    sign: int


@dataclass(frozen=True)
class SlotSymmetryGroup:
    """Finite signed permutation group for one tensor head.

    Declared monoterm symmetries and Young row/column generators are closed
    under composition.  Slot canonicalization chooses the signed minimum over
    the whole orbit rather than doing a local one-shot slot sort.
    """

    rank: int
    generators: tuple[SlotPermutationRule, ...] = ()

    @staticmethod
    def identity(rank: int) -> "SlotSymmetryGroup":
        return SlotSymmetryGroup(rank, ())

    def orbit(self) -> tuple[SlotOrbitElement, ...]:
        identity = tuple(range(self.rank))
        seen: dict[tuple[int, ...], int] = {identity: 1}
        queue: deque[tuple[tuple[int, ...], int]] = deque([(identity, 1)])
        while queue:
            perm, sign = queue.popleft()
            for gen in self.generators:
                if len(gen.permutation) != self.rank:
                    continue
                composed = tuple(perm[i] for i in gen.permutation)
                new_sign = sign * gen.sign
                old_sign = seen.get(composed)
                if old_sign is None:
                    seen[composed] = new_sign
                    queue.append((composed, new_sign))
                elif old_sign != new_sign:
                    return (SlotOrbitElement(identity, 0),)
        return tuple(SlotOrbitElement(p, s) for p, s in sorted(seen.items(), key=lambda item: repr(item[0])))

    def canonicalize_indices(self, indices: Sequence[Any]) -> tuple[tuple[Any, ...], int, tuple[int, ...], bool]:
        if len(indices) != self.rank:
            return tuple(indices), 1, tuple(range(len(indices))), False
        best_indices: tuple[Any, ...] | None = None
        best_sign = 1
        best_perm: tuple[int, ...] = tuple(range(self.rank))
        zero = False
        for elem in self.orbit():
            if elem.sign == 0:
                zero = True
                continue
            candidate = tuple(indices[i] for i in elem.permutation)
            key = tuple(repr((_index_name(i), _index_variance(i))) for i in candidate)
            if best_indices is None or key < tuple(repr((_index_name(i), _index_variance(i))) for i in best_indices):
                best_indices = candidate
                best_sign = elem.sign
                best_perm = elem.permutation
        if best_indices is None:
            return tuple(indices), 1, tuple(range(self.rank)), True
        labels = tuple((_index_name(i), _index_variance(i)) for i in indices)
        for elem in self.orbit():
            if elem.sign < 0 and tuple(labels[i] for i in elem.permutation) == labels:
                zero = True
                break
        return best_indices, best_sign, best_perm, zero


@dataclass(frozen=True)
class YoungProjector:
    """Young-tableau style row symmetrizer / column antisymmetrizer."""

    rows: tuple[tuple[int, ...], ...] = ()
    columns: tuple[tuple[int, ...], ...] = ()

    def generators(self, rank: int) -> tuple[SlotPermutationRule, ...]:
        gens: list[SlotPermutationRule] = []
        for row in self.rows:
            for a, b in zip(row, row[1:]):
                perm = list(range(rank)); perm[a], perm[b] = perm[b], perm[a]
                gens.append(SlotPermutationRule(tuple(perm), 1, "young_row"))
        for col in self.columns:
            for a, b in zip(col, col[1:]):
                perm = list(range(rank)); perm[a], perm[b] = perm[b], perm[a]
                gens.append(SlotPermutationRule(tuple(perm), -1, "young_column"))
        return tuple(gens)

    def project_terms(self, tensor: TensorExpr) -> tuple[tuple[int, TensorExpr], ...]:
        indices = tuple(tensor.metadata.get("indices", ()))
        rank = len(indices)

        def subgroup(groups: tuple[tuple[int, ...], ...], sign_for_swap: int) -> tuple[SlotOrbitElement, ...]:
            gens: list[SlotPermutationRule] = []
            for group in groups:
                for a, b in zip(group, group[1:]):
                    perm = list(range(rank)); perm[a], perm[b] = perm[b], perm[a]
                    gens.append(SlotPermutationRule(tuple(perm), sign_for_swap))
            return SlotSymmetryGroup(rank, tuple(gens)).orbit()

        row_orbit = subgroup(self.rows, 1) or (SlotOrbitElement(tuple(range(rank)), 1),)
        col_orbit = subgroup(self.columns, -1) or (SlotOrbitElement(tuple(range(rank)), 1),)
        terms: list[tuple[int, TensorExpr]] = []
        for col in col_orbit:
            for row in row_orbit:
                if col.sign == 0 or row.sign == 0:
                    continue
                perm = tuple(row.permutation[col.permutation[i]] for i in range(rank))
                md = dict(tensor.metadata)
                md["indices"] = tuple(indices[i] for i in perm)
                terms.append((col.sign * row.sign, TensorExpr(tensor.kind, tensor.payload, tensor.children, md, tensor.provenance)))
        return tuple(terms)


@dataclass(frozen=True)
class YoungSymmetryRule:
    tensor: str
    rows: tuple[tuple[int, ...], ...] = ()
    columns: tuple[tuple[int, ...], ...] = ()

    def projector(self) -> YoungProjector:
        return YoungProjector(self.rows, self.columns)

    def to_slot_symmetries(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        rules: list[tuple[str, tuple[int, ...]]] = []
        for row in self.rows:
            if len(row) > 1:
                rules.append(("symmetric", row))
        for col in self.columns:
            if len(col) > 1:
                rules.append(("antisymmetric", col))
        return tuple(rules)


@dataclass(frozen=True)
class LinearIdentityTerm:
    coefficient: sp.Expr
    node: TensorExpr


@dataclass(frozen=True)
class LinearIdentityRule:
    """Homogeneous linear identity over TensorExpr terms."""

    name: str
    terms: tuple[LinearIdentityTerm, ...]
    dimension: int | None = None
    replacement: TensorExpr = field(default_factory=lambda: ir_node("zero"))

    def applicable(self, ir: TensorExpr, registry: DeclarationRegistry | None) -> bool:
        if self.dimension is not None:
            dim = _registry_dimension(registry) or ir.metadata.get("dimension")
            if dim != self.dimension:
                return False
        if ir.kind not in {"add", "indexed_expr:add", "curvature_linear_combo"}:
            return False
        if len(ir.children) != len(self.terms):
            return False
        have = Counter(canonical_ir_key(ch) for ch in ir.children)
        want = Counter(canonical_ir_key(term.node) for term in self.terms)
        return have == want

    def apply(self, ir: TensorExpr) -> TensorExpr:
        return self.replacement.with_provenance(ir.provenance.append(self.name, source="linear_identity", before_key=canonical_ir_key(ir), after_key=canonical_ir_key(self.replacement)))


@dataclass(frozen=True)
class IdentityRule:
    name: str
    source_kind: str
    target: TensorExpr | None = None
    dimension: int | None = None
    antisymmetric_slots: tuple[int, ...] = ()
    replacement_kind: str | None = None

    def applicable(self, ir: TensorExpr, registry: DeclarationRegistry | None) -> bool:
        if ir.kind != self.source_kind:
            return False
        if self.dimension is None:
            return True
        dim = _registry_dimension(registry) or ir.metadata.get("dimension")
        return dim == self.dimension

    def apply(self, ir: TensorExpr, registry: DeclarationRegistry | None) -> TensorExpr:
        if self.target is not None:
            return self.target
        if self.replacement_kind == "zero":
            return ir_node("zero", provenance={"origin": "identity_rule", "rule": self.name})
        return ir

def _registry_dimension(registry: DeclarationRegistry | None) -> int | None:
    if registry is None:
        return None
    if getattr(registry, "manifolds", None):
        return next(iter(registry.manifolds.values())).dimension
    return None


def _with_step(steps: list[CanonicalizationStep], before: TensorExpr, after: TensorExpr, rule: str, detail: str = "", **metadata: Any) -> None:
    before_key = canonical_ir_key(before)
    after_key = canonical_ir_key(after)
    if before_key != after_key:
        steps.append(CanonicalizationStep(rule, before_key, after_key, detail, metadata))


def _index_name(index: Any) -> str:
    if isinstance(index, tuple) and index:
        return str(index[0])
    return str(getattr(index, "name", index))


def _index_variance(index: Any, fallback: str = "") -> str:
    if isinstance(index, tuple) and len(index) > 1:
        return str(index[1])
    return str(getattr(index, "variance", fallback))


def _index_family_bundle(name: str, registry: DeclarationRegistry | None) -> tuple[str | None, str | None]:
    if registry is None:
        return None, None
    for family_name, family in registry.index_families.items():
        if name in family.symbols or name.startswith(family.dummy_prefix or family.name):
            return family_name, family.bundle
    return None, None


def _tensor_decl(registry: DeclarationRegistry | None, name: str):
    if registry is None:
        return None
    if name in registry.tensors:
        return registry.tensors[name]
    if name in registry.metrics:
        return registry.metrics[name].to_tensor_declaration()
    return None


def _walk(ir: TensorExpr):
    yield ir
    for ch in ir.children:
        yield from _walk(ch)


def _collect_index_occurrences(ir: TensorExpr, registry: DeclarationRegistry | None) -> dict[str, list[tuple[str | None, str | None, str]]]:
    occurrences: dict[str, list[tuple[str | None, str | None, str]]] = defaultdict(list)
    for node in _walk(ir):
        indices = node.metadata.get("indices", ())
        tensor_name = node.metadata.get("tensor_name", node.payload if node.kind == "indexed_tensor" else None)
        decl = _tensor_decl(registry, str(tensor_name)) if tensor_name is not None else None
        variance_spec = tuple(str(v) for v in getattr(decl, "variance", ())) or tuple(node.metadata.get("variance_spec", ""))
        for pos, idx in enumerate(indices):
            name = _index_name(idx)
            family, bundle = _index_family_bundle(name, registry)
            var = _index_variance(idx, variance_spec[pos] if pos < len(variance_spec) else "")
            occurrences[name].append((family, bundle, var))
    return occurrences


def _rename_index_value(index: Any, new_name: str) -> Any:
    if isinstance(index, tuple):
        if len(index) == 1:
            return (new_name,)
        return (new_name,) + tuple(index[1:])
    if hasattr(index, "__class__") and hasattr(index, "variance"):
        return (new_name, getattr(index, "variance"))
    return new_name


def _rename_dummies(ir: TensorExpr, registry: DeclarationRegistry | None) -> TensorExpr:
    # Sums contain repeated free-index labels across addends; those are not dummy
    # contractions.  Rename independently inside each additive branch.
    if ir.kind in {"add", "indexed_expr:add", "curvature_linear_combo"}:
        return TensorExpr(ir.kind, ir.payload, tuple(_rename_dummies(ch, registry) for ch in ir.children), dict(ir.metadata), ir.provenance)
    occurrences = _collect_index_occurrences(ir, registry)
    dummy_names = []
    for name, occ in occurrences.items():
        if len(occ) < 2:
            continue
        variances = {v for _, _, v in occ if v}
        if not variances or ("u" in variances and "l" in variances) or len(occ) == 2:
            dummy_names.append(name)
    grouped: dict[tuple[str | None, str | None], list[str]] = defaultdict(list)
    for name in dummy_names:
        family, bundle, _ = occurrences[name][0]
        grouped[(family, bundle)].append(name)
    rename: dict[str, str] = {}
    for (family, bundle), names in sorted(grouped.items(), key=repr):
        prefix = None
        if registry is not None and family in registry.index_families:
            prefix = registry.index_families[family].dummy_prefix
        prefix = prefix or (family or bundle or "d")
        for number, name in enumerate(sorted(names), start=1):
            rename[name] = f"{prefix}{number}"

    def rec(node: TensorExpr) -> TensorExpr:
        children = tuple(rec(ch) for ch in node.children)
        md = dict(node.metadata)
        if "indices" in md:
            md["indices"] = tuple(_rename_index_value(idx, rename.get(_index_name(idx), _index_name(idx))) for idx in md["indices"])
        if "index" in md and _index_name(md["index"]) in rename:
            md["index"] = _rename_index_value(md["index"], rename[_index_name(md["index"])])
        if "left_index" in md and _index_name(md["left_index"]) in rename:
            md["left_index"] = _rename_index_value(md["left_index"], rename[_index_name(md["left_index"])])
        if "right_index" in md and _index_name(md["right_index"]) in rename:
            md["right_index"] = _rename_index_value(md["right_index"], rename[_index_name(md["right_index"])])
        return TensorExpr(node.kind, node.payload, children, md, node.provenance)

    return rec(ir)


def _slot_rules_for_tensor(registry: DeclarationRegistry | None, tensor_name: str, young_rules: Sequence[YoungSymmetryRule]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    rules: list[tuple[str, tuple[int, ...]]] = []
    decl = _tensor_decl(registry, tensor_name)
    if decl is not None:
        for sym in decl.symmetries:
            rules.append((sym.kind, tuple(sym.slots)))
    for young in young_rules:
        if young.tensor == tensor_name:
            rules.extend(young.to_slot_symmetries())
    return tuple(rules)


def _swap_generator(rank: int, a: int, b: int, sign: int, source: str) -> SlotPermutationRule:
    perm = list(range(rank))
    perm[a], perm[b] = perm[b], perm[a]
    return SlotPermutationRule(tuple(perm), sign, source)


def _slot_group_for_tensor(registry: DeclarationRegistry | None, tensor_name: str, rank: int, young_rules: Sequence[YoungSymmetryRule]) -> SlotSymmetryGroup:
    generators: list[SlotPermutationRule] = []
    for kind, slots in _slot_rules_for_tensor(registry, tensor_name, young_rules):
        slots = tuple(slots)
        if len(slots) < 2 or any(pos >= rank or pos < 0 for pos in slots):
            continue
        if kind in {"symmetric", "row_symmetric"}:
            generators.extend(_swap_generator(rank, a, b, 1, kind) for a, b in zip(slots, slots[1:]))
        elif kind in {"antisymmetric", "column_antisymmetric"}:
            generators.extend(_swap_generator(rank, a, b, -1, kind) for a, b in zip(slots, slots[1:]))
        elif kind in {"cyclic", "riemann_pair_exchange"} and len(slots) > 2:
            perm = list(range(rank))
            rotated = slots[1:] + slots[:1]
            for dst, src in zip(slots, rotated):
                perm[dst] = src
            generators.append(SlotPermutationRule(tuple(perm), 1, kind))
    for young in young_rules:
        if young.tensor == tensor_name:
            generators.extend(young.projector().generators(rank))
    return SlotSymmetryGroup(rank, tuple(generators))


def _canonicalize_slot_symmetries(ir: TensorExpr, registry: DeclarationRegistry | None, young_rules: Sequence[YoungSymmetryRule]) -> TensorExpr:
    children = tuple(_canonicalize_slot_symmetries(ch, registry, young_rules) for ch in ir.children)
    md = dict(ir.metadata)
    if ir.kind == "indexed_tensor" and "indices" in md:
        tensor_name = str(md.get("tensor_name", ir.payload))
        indices = tuple(md["indices"])
        group = _slot_group_for_tensor(registry, tensor_name, len(indices), young_rules)
        canon_indices, sign, permutation, zero = group.canonicalize_indices(indices)
        if zero:
            return ir_node("zero", provenance={"origin": "slot_symmetry_group", "tensor": tensor_name})
        md["indices"] = canon_indices
        if permutation != tuple(range(len(indices))):
            md["slot_permutation"] = permutation
            md["slot_symmetry_group_size"] = len(group.orbit())
        if sign != 1:
            md["coefficient"] = sp.sympify(md.get("coefficient", 1)) * sign
            md["slot_symmetry_sign"] = int(md.get("slot_symmetry_sign", 1)) * sign
    return TensorExpr(ir.kind, ir.payload, children, md, ir.provenance)


def _normalize_metric_delta_epsilon(ir: TensorExpr, registry: DeclarationRegistry | None) -> TensorExpr:
    children = tuple(_normalize_metric_delta_epsilon(ch, registry) for ch in ir.children)
    node = TensorExpr(ir.kind, ir.payload, children, dict(ir.metadata), ir.provenance)
    if node.kind not in {"mul", "indexed_expr:mul", "indexed_expr:tensor_product", "contract"}:
        return node
    factors = list(children)
    if not factors:
        return node
    def tensor_name(f: TensorExpr) -> str | None:
        return str(f.metadata.get("tensor_name", f.payload)) if f.kind == "indexed_tensor" else None
    changed = True
    while changed:
        changed = False
        for i, left in enumerate(list(factors)):
            lname = tensor_name(left)
            if lname not in {"delta", "KroneckerDelta", "δ"} or len(left.metadata.get("indices", ())) != 2:
                continue
            a, b = left.metadata["indices"]
            a_name, b_name = _index_name(a), _index_name(b)
            for j, right in enumerate(list(factors)):
                if i == j or right.kind != "indexed_tensor":
                    continue
                indices = tuple(right.metadata.get("indices", ()))
                if any(_index_name(idx) == b_name for idx in indices):
                    new_indices = tuple(_rename_index_value(idx, a_name) if _index_name(idx) == b_name else idx for idx in indices)
                    md = dict(right.metadata); md["indices"] = new_indices
                    factors[j] = TensorExpr(right.kind, right.payload, right.children, md, right.provenance)
                    factors.pop(i)
                    changed = True
                    break
            if changed:
                break
    # canonical factor order after contractions
    return TensorExpr(node.kind, node.payload, tuple(sorted(factors, key=lambda f: repr(canonical_ir_key(f)))), dict(node.metadata), node.provenance)


def _normalize_products_and_sums(ir: TensorExpr) -> TensorExpr:
    children = tuple(_normalize_products_and_sums(ch) for ch in ir.children)
    md = dict(ir.metadata)
    if ir.kind in {"add", "mul", "indexed_expr:add", "indexed_expr:mul", "indexed_expr:tensor_product", "contract", "wedge"}:
        flat: list[TensorExpr] = []
        for ch in children:
            if ch.kind == ir.kind:
                flat.extend(ch.children)
            else:
                flat.append(ch)
        if any(ch.kind == "zero" for ch in flat) and ir.kind in {"mul", "indexed_expr:mul", "indexed_expr:tensor_product", "contract"}:
            return ir_node("zero")
        if ir.kind in {"add", "indexed_expr:add"}:
            flat = [ch for ch in flat if ch.kind != "zero"]
            if not flat:
                return ir_node("zero")
        return TensorExpr(ir.kind, ir.payload, tuple(sorted(flat, key=lambda f: repr(canonical_ir_key(f)))), md, ir.provenance)
    return TensorExpr(ir.kind, ir.payload, children, md, ir.provenance)


def _normalize_covariant_derivatives(ir: TensorExpr, registry: DeclarationRegistry | None) -> TensorExpr:
    children = tuple(_normalize_covariant_derivatives(ch, registry) for ch in ir.children)
    node = TensorExpr(ir.kind, ir.payload, children, dict(ir.metadata), ir.provenance)
    if node.kind != "covariant_derivative" or len(node.children) != 1:
        return node
    inner = node.children[0]
    if inner.kind != "covariant_derivative" or len(inner.children) != 1:
        return node
    left = node.metadata.get("index")
    right = inner.metadata.get("index")
    if left is None or right is None:
        return node
    if repr(_index_name(left)) <= repr(_index_name(right)):
        return node
    # Sort commuting derivatives.  If the registry has a matching commutation rule, retain a diagnostic marker
    # because the actual curvature commutator expansion belongs to a rewrite family, not this ordering pass.
    outer_md = dict(node.metadata); inner_md = dict(inner.metadata)
    outer_md["index"], inner_md["index"] = right, left
    swapped_inner = TensorExpr(inner.kind, inner.payload, inner.children, inner_md, inner.provenance)
    swapped = TensorExpr(node.kind, node.payload, (swapped_inner,), outer_md, node.provenance)
    if registry is not None and registry.commutation_rules:
        swapped = swapped.with_metadata(commutation_ordered=True, commutation_rule=next(iter(registry.commutation_rules)))
    return swapped


def _identity_rules_default(registry: DeclarationRegistry | None) -> tuple[IdentityRule, ...]:
    dim = _registry_dimension(registry)
    rules = [
        IdentityRule("riemann_two_dim_weyl_zero", "curvature", dimension=2, replacement_kind="zero"),
    ]
    if dim is not None and dim < 4:
        rules.append(IdentityRule("low_dim_weyl_zero", "curvature:weyl", dimension=dim, replacement_kind="zero"))
    return tuple(rules)


def _apply_identity_rules(ir: TensorExpr, registry: DeclarationRegistry | None, rules: Sequence[IdentityRule | LinearIdentityRule]) -> TensorExpr:
    children = tuple(_apply_identity_rules(ch, registry, rules) for ch in ir.children)
    node = TensorExpr(ir.kind, ir.payload, children, dict(ir.metadata), ir.provenance)
    for rule in rules:
        if isinstance(rule, LinearIdentityRule):
            if rule.applicable(node, registry):
                return rule.apply(node)
            continue
        if rule.applicable(node, registry):
            candidate = rule.apply(node, registry)
            if candidate is not node:
                return candidate.with_provenance(node.provenance.append(rule.name, source="identity_canonicalization", before_key=canonical_ir_key(node), after_key=canonical_ir_key(candidate)))
    return node


def _confluence_diagnostics(original: TensorExpr, canonical: TensorExpr, registry: DeclarationRegistry | None) -> tuple[ConfluenceDiagnostic, ...]:
    diagnostics: list[ConfluenceDiagnostic] = []
    # Idempotence is the minimum practical confluence check for a canonicalizer.
    second = _normalize_products_and_sums(canonical)
    if canonical_ir_key(second) != canonical_ir_key(canonical):
        diagnostics.append(ConfluenceDiagnostic("canonicalization_not_idempotent", "warning", ("first",), ("second",)))
    for node in _walk(original):
        if node.kind == "indexed_tensor":
            name = str(node.metadata.get("tensor_name", node.payload))
            rules = _slot_rules_for_tensor(registry, name, ())
            if len(rules) > 1:
                diagnostics.append(ConfluenceDiagnostic("overlapping_slot_symmetries", "info", tuple(r[0] for r in rules), metadata={"tensor": name}))
    return tuple(diagnostics)


def canonicalize_tensor_expr(
    obj: Any,
    *,
    registry: DeclarationRegistry | None = None,
    policy: CanonicalizationPolicy | None = None,
    identity_rules: Sequence[IdentityRule | LinearIdentityRule] = (),
    young_rules: Sequence[YoungSymmetryRule] = (),
) -> TensorExprCanonicalizationReport:
    policy = policy or CanonicalizationPolicy()
    original = to_tensor_expr(obj)
    current = original
    steps: list[CanonicalizationStep] = []
    all_rules = tuple(identity_rules) + (_identity_rules_default(registry) if policy.apply_identities else ())

    for _ in range(policy.max_passes):
        before_pass = current
        if policy.rename_dummies:
            new = _rename_dummies(current, registry); _with_step(steps, current, new, "dummy_index_renaming", "bundle-aware dummy renaming"); current = new
        if policy.apply_slot_symmetry:
            new = _canonicalize_slot_symmetries(current, registry, young_rules); _with_step(steps, current, new, "slot_symmetry", "monoterm/Young slot symmetry"); current = new
        if policy.normalize_metric_delta_epsilon:
            new = _normalize_metric_delta_epsilon(current, registry); _with_step(steps, current, new, "metric_delta_epsilon", "metric/delta/epsilon contraction normalization"); current = new
        if policy.order_covariant_derivatives:
            new = _normalize_covariant_derivatives(current, registry); _with_step(steps, current, new, "covariant_derivative_ordering", "canonical derivative order"); current = new
        if policy.apply_identities and all_rules:
            new = _apply_identity_rules(current, registry, all_rules); _with_step(steps, current, new, "identity_rules", "monoterm/multiterm/dimension identities"); current = new
        if policy.order_products:
            new = _normalize_products_and_sums(current); _with_step(steps, current, new, "product_ordering", "canonical product/sum ordering"); current = new
        if canonical_ir_key(before_pass) == canonical_ir_key(current):
            break

    provenance = IRProvenance(origin=current.provenance.origin, steps=current.provenance.steps, metadata=dict(current.provenance.metadata))
    for step in steps:
        provenance = provenance.append(step.rule, source="tensor_expr_canonicalization", before_key=step.before_key, after_key=step.after_key, detail=step.detail)
    current = current.with_provenance(provenance)
    diagnostics = _confluence_diagnostics(original, current, registry) if policy.check_confluence else ()
    return TensorExprCanonicalizationReport(original, current, canonical_ir_key(current), tuple(steps), diagnostics)


def canonical_tensor_expr_key(obj: Any, *, registry: DeclarationRegistry | None = None, **kwargs: Any) -> tuple[Any, ...]:
    return canonicalize_tensor_expr(obj, registry=registry, **kwargs).canonical_key
