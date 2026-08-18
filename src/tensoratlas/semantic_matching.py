
from __future__ import annotations

from functools import lru_cache
from typing import Any
from itertools import product, permutations

import sympy as sp

from .canonical_keys import structural_key


def _safe_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def canonical_match_object(obj: Any) -> Any:
    cls = type(obj).__name__

    if cls in {"ExteriorFormNF", "HodgeExpr", "CodifferentialExpr", "InteriorExpr", "LieExpr"}:
        from .semantic_core import semantic_normalize_object
        return semantic_normalize_object(obj)

    if cls == "GammaStringExpr":
        from .semantic_ops import evaluate_semantic_operator
        return sp.expand(evaluate_semantic_operator(obj))

    if cls in {"IndexedTensor", "IndexedTensorExpr"}:
        from .tensor_indices import alpha_rename_dummies, canonical_indexed_form
        renamed = alpha_rename_dummies(obj, prefix="d")
        return canonical_indexed_form(renamed)

    if cls in {"AbstractTensorExpr", "AbstractTensor", "TensorMonomial"}:
        try:
            from .abstract_tensor import canonical_tensor_expression
            norm = canonical_tensor_expression(obj)
            if hasattr(norm, "expr"):
                return norm.expr
        except Exception:
            pass

    if isinstance(obj, sp.Basic):
        try:
            return sp.expand(obj)
        except Exception:
            return obj

    return obj


def _coerce_basic(obj: Any) -> Any:
    try:
        return sp.sympify(obj)
    except Exception:
        return obj


def semantic_match_key(obj: Any) -> tuple[Any, ...]:
    canon = canonical_match_object(_coerce_basic(obj))
    cls = type(canon).__name__
    if cls in {"ExteriorFormNF", "HodgeExpr", "CodifferentialExpr", "InteriorExpr", "LieExpr", "GammaStringExpr"}:
        from .semantic_core import compile_semantic_node, semantic_node_fingerprint
        return semantic_node_fingerprint(compile_semantic_node(canon))
    if cls in {"IndexedTensor", "IndexedTensorExpr"}:
        from .semantic_core import compile_semantic_node
        node = compile_semantic_node(canon)
        rsigs = indexed_identity_rewrite_signatures(node)
        return min(rsigs, key=repr) if rsigs else indexed_graph_family_signature(node)
    if isinstance(canon, (sp.Basic, int, float)):
        try:
            return ("sympy", sp.srepr(sp.sympify(canon)))
        except Exception:
            pass
    return ("structural", structural_key(canon))


def semantic_equivalent_objects(left: Any, right: Any) -> bool:
    lcls = type(left).__name__
    rcls = type(right).__name__
    if lcls in {"IndexedTensor", "IndexedTensorExpr", "SemanticNode"} or rcls in {"IndexedTensor", "IndexedTensorExpr", "SemanticNode"}:
        try:
            return indexed_graph_equivalent(left, right)
        except Exception:
            pass
    return semantic_match_key(left) == semantic_match_key(right)



def _perm_sign(perm: tuple[int, ...]) -> int:
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inv += 1
    return -1 if inv % 2 else 1



def indexed_tensor_orbit_specs(obj: Any, *, max_states: int = 64) -> tuple[tuple[tuple[int, ...], int], ...]:
    cls = type(obj).__name__
    if cls != 'IndexedTensor':
        return ((tuple(range(len(getattr(obj, 'indices', tuple())))), 1),)
    n = len(obj.indices)
    md = getattr(getattr(obj, 'tensor', None), 'symmetry_metadata', {}) or {}
    groups = []

    def _all_perms(grp, antisym=False):
        vals = []
        for perm in permutations(grp):
            sign = _perm_sign(tuple(grp.index(p) for p in perm)) if antisym else 1
            vals.append((perm, sign))
        return tuple(vals)

    for key in ('symmetric', 'antisymmetric'):
        for group in md.get(key, ()):
            grp = tuple(int(i) for i in group)
            groups.append((grp, _all_perms(grp, antisym=(key == 'antisymmetric'))))

    for group in md.get('cyclic', ()):
        grp = tuple(int(i) for i in group)
        perms = []
        m = len(grp)
        for shift in range(m):
            perm = tuple(grp[(i + shift) % m] for i in range(m))
            perms.append((perm, 1))
        groups.append((grp, tuple(perms)))

    for key, sign0 in (('pair_symmetric', 1), ('pair_antisymmetric', -1)):
        for pair_group in md.get(key, ()):
            # expected shape: ((a,b),(c,d),...)
            pair_group = tuple(tuple(int(i) for i in grp) for grp in pair_group)
            if len(pair_group) < 2:
                continue
            perms = []
            m = len(pair_group)
            for perm_idx in permutations(range(m)):
                perm = []
                sign = 1
                if sign0 == -1:
                    sign = _perm_sign(tuple(perm_idx))
                for dest_group, src_i in zip(pair_group, perm_idx):
                    perm.extend(pair_group[src_i])
                flat_dest = tuple(i for grp in pair_group for i in grp)
                perms.append((tuple(perm), sign))
                groups.append((flat_dest, tuple(perms)))
                break  # appended below outside loop
            # rebuild correctly once
            groups.pop()
            flat_dest = tuple(i for grp in pair_group for i in grp)
            perms = []
            for perm_idx in permutations(range(m)):
                src_flat = []
                sign = _perm_sign(tuple(perm_idx)) if sign0 == -1 else 1
                for src_i in perm_idx:
                    src_flat.extend(pair_group[src_i])
                perms.append((tuple(src_flat), sign))
            groups.append((flat_dest, tuple(perms)))

    if not groups:
        return ((tuple(range(n)), 1),)
    out = []
    for combo in product(*[g[1] for g in groups]):
        perm = list(range(n))
        sign = 1
        for (grp, _), (arr, sgn) in zip(groups, combo):
            for dest, src in zip(grp, arr):
                perm[dest] = src
            sign *= sgn
        item = (tuple(perm), int(sign))
        if item not in out:
            out.append(item)
        if len(out) >= max_states:
            break
    if not out:
        out = [(tuple(range(n)), 1)]
    return tuple(out)

def _unique_orbit_nodes(items: list[tuple[Any, int]], *, max_states: int = 128) -> tuple[tuple[Any, int], ...]:
    from .semantic_core import semantic_node_fingerprint
    seen = set()
    out = []
    for node, sign in items:
        key = (semantic_node_fingerprint(node), int(sign))
        if key in seen:
            continue
        seen.add(key)
        out.append((node, int(sign)))
        if len(out) >= max_states:
            break
    return tuple(out)




def _normalize_symmetry_group(group: Any) -> Any:
    if isinstance(group, (tuple, list)) and group and all(not isinstance(x, (tuple, list)) for x in group):
        vals = tuple(int(x) for x in group)
        if not vals:
            return vals
        rots = [vals[i:] + vals[:i] for i in range(len(vals))]
        rev = vals[::-1]
        rots_rev = [rev[i:] + rev[:i] for i in range(len(rev))]
        return min(rots + rots_rev)
    if isinstance(group, (tuple, list)):
        norm = tuple(_normalize_symmetry_group(g) for g in group)
        try:
            return tuple(sorted(norm))
        except Exception:
            return norm
    return group


def _collect_index_name_stats(node: Any, stats: dict[tuple[str, Any], dict[str, Any]], order: list[tuple[str, Any]]) -> None:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode):
        return
    if node.kind == "tensor_index" and node.children:
        name = str(getattr(node.children[0], "value", ""))
        variance, bundle = node.value if isinstance(node.value, tuple) and len(node.value) == 2 else (None, None)
        key = (name, bundle)
        slot = stats.setdefault(key, {"variances": set(), "count": 0})
        slot["variances"].add(variance)
        slot["count"] += 1
        if key not in order:
            order.append(key)
    for child in node.children:
        _collect_index_name_stats(child, stats, order)

def indexed_tree_dummy_normalize(node: Any) -> Any:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode):
        return node
    stats = {}
    order = []
    _collect_index_name_stats(node, stats, order)
    rename = {}
    counter = 0
    for key in order:
        variances = stats[key]["variances"]
        if 'u' in variances and 'l' in variances:
            rename[key] = f"d{counter}"
            counter += 1

    def rec(cur):
        if not isinstance(cur, SemanticNode):
            return cur
        if cur.kind == "tensor_index" and cur.children:
            variance, bundle = cur.value if isinstance(cur.value, tuple) and len(cur.value) == 2 else (None, None)
            old_name = str(getattr(cur.children[0], "value", ""))
            new_name = rename.get((old_name, bundle), old_name)
            return SemanticNode(
                cur.kind,
                value=cur.value,
                children=(SemanticNode("atom", value=sp.Symbol(new_name), metadata=dict(cur.children[0].metadata)),),
                metadata=dict(cur.metadata),
            )
        return SemanticNode(cur.kind, value=cur.value, children=tuple(rec(ch) for ch in cur.children), metadata=dict(cur.metadata))
    return rec(node)

def indexed_expression_orbit_nodes(node: Any, *, max_states: int = 128) -> tuple[tuple[Any, int], ...]:
    from .semantic_core import SemanticNode

    if not isinstance(node, SemanticNode):
        return tuple()

    if node.kind == "indexed_tensor":
        original = node.metadata.get("_original_obj")
        variants = []
        if original is None:
            return ((node, 1),)
        for perm, sign in indexed_tensor_orbit_specs(original, max_states=max_states):
            perm_children = tuple(node.children[i] for i in perm)
            variants.append(
                (
                    SemanticNode(
                        node.kind,
                        value=node.value,
                        children=perm_children,
                        metadata=dict(node.metadata, _orbit_sign=int(sign)),
                    ),
                    int(sign),
                )
            )
        normed = [(indexed_tree_dummy_normalize(vnode), sign) for vnode, sign in (variants or [(node, 1)])]
        return _unique_orbit_nodes(normed, max_states=max_states)

    if node.kind in {"indexed_add", "indexed_tensor_product", "indexed_expr"}:
        child_variants = []
        for child in node.children:
            if isinstance(child, SemanticNode) and str(child.kind).startswith("indexed"):
                vars_for_child = indexed_expression_orbit_nodes(child, max_states=max_states)
                child_variants.append(vars_for_child or ((child, 1),))
            else:
                child_variants.append(((child, 1),))
        out = []
        total = 1
        for variants in child_variants:
            total *= max(1, len(variants))
            if total > max_states:
                break
        from itertools import product as _product
        for combo in _product(*child_variants):
            children = tuple(item[0] for item in combo)
            sign = 1
            for _, sgn in combo:
                sign *= int(sgn)
            out.append(
                (
                    SemanticNode(
                        node.kind,
                        value=node.value,
                        children=children,
                        metadata=dict(node.metadata, _orbit_sign=int(sign)),
                    ),
                    int(sign),
                )
            )
            if len(out) >= max_states:
                break
        normed = [(indexed_tree_dummy_normalize(vnode), sign) for vnode, sign in (out or [(node, 1)])]
        return _unique_orbit_nodes(normed, max_states=max_states)

    return ((node, 1),)


def _indexed_tree_refinement_data(node: Any):
    from .semantic_core import SemanticNode

    leaves = []
    class_occ = {}

    def visit(cur, path=()):
        if not isinstance(cur, SemanticNode):
            return
        if cur.kind == "indexed_tensor":
            tensor_sig = (
                cur.value,
                cur.metadata.get("variance_spec"),
            )
            leaf_id = len(leaves)
            slots = []
            for slot, child in enumerate(cur.children):
                if isinstance(child, SemanticNode) and child.kind == "tensor_index" and child.children:
                    variance, bundle = child.value if isinstance(child.value, tuple) and len(child.value) == 2 else (None, None)
                    name = str(getattr(child.children[0], "value", ""))
                    cls_key = (name, bundle)
                    class_occ.setdefault(cls_key, []).append((leaf_id, slot, variance, bundle))
                    slots.append((variance, bundle, cls_key))
            leaves.append({"tensor_sig": tensor_sig, "slots": slots, "path": path})
            return
        for i, ch in enumerate(cur.children):
            visit(ch, path + (cur.kind, i))

    visit(node)
    return leaves, class_occ

def indexed_contraction_graph_signature(node: Any, *, rounds: int = 4) -> tuple[Any, ...]:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode):
        return ("non_semantic", structural_key(node))
    node = indexed_tree_dummy_normalize(node)
    leaves, class_occ = _indexed_tree_refinement_data(node)
    if not leaves:
        return ("indexed_empty", node.kind, node.value, tuple())

    leaf_labels = [("leaf", leaf["tensor_sig"]) for leaf in leaves]
    class_labels = {
        cls: ("class", tuple(sorted((occ[2], occ[3]) for occ in occs)), len(occs))
        for cls, occs in class_occ.items()
    }

    for _ in range(rounds):
        new_leaf_labels = []
        for leaf_id, leaf in enumerate(leaves):
            slot_desc = []
            for slot, (variance, bundle, cls_key) in enumerate(leaf["slots"]):
                slot_desc.append((slot, variance, bundle, class_labels.get(cls_key)))
            new_leaf_labels.append(("leaf", leaf["tensor_sig"], tuple(slot_desc)))
        leaf_labels = new_leaf_labels

        new_class_labels = {}
        for cls, occs in class_occ.items():
            inc = []
            for leaf_id, slot, variance, bundle in occs:
                inc.append((leaf_labels[leaf_id], slot, variance, bundle))
            new_class_labels[cls] = ("class", tuple(sorted(inc)))
        class_labels = new_class_labels

    leaf_sigs = []
    for leaf_id, leaf in enumerate(leaves):
        slot_desc = []
        for slot, (variance, bundle, cls_key) in enumerate(leaf["slots"]):
            slot_desc.append((slot, variance, bundle, class_labels.get(cls_key)))
        leaf_sigs.append((leaf["tensor_sig"], tuple(slot_desc)))

    child_sigs = []
    if node.kind in {"indexed_add", "indexed_tensor_product", "indexed_expr"}:
        for ch in node.children:
            if isinstance(ch, SemanticNode) and str(ch.kind).startswith("indexed"):
                child_sigs.append(indexed_contraction_graph_signature(ch, rounds=rounds))
            else:
                child_sigs.append(("child", structural_key(ch.value if isinstance(ch, SemanticNode) else ch)))
        if node.kind in {"indexed_add", "indexed_tensor_product"}:
            child_sigs = sorted(child_sigs)

    return (
        "indexed_graph",
        node.kind,
        node.value,
        tuple(sorted(leaf_sigs)),
        tuple(child_sigs),
    )

def indexed_graph_equivalent(left: Any, right: Any) -> bool:
    from .semantic_core import compile_semantic_node
    lnode = left if type(left).__name__ == "SemanticNode" else compile_semantic_node(left)
    rnode = right if type(right).__name__ == "SemanticNode" else compile_semantic_node(right)

    # Fast path for named identity families.  This avoids constructing the full
    # orbit of an additive curvature identity, which can grow combinatorially
    # when every Riemann factor carries antisymmetry and pair-exchange metadata.
    lspecial = {s for s in indexed_identity_rewrite_signatures(lnode) if s and s[0] == "identity_rewrite"}
    rspecial = {s for s in indexed_identity_rewrite_signatures(rnode) if s and s[0] == "identity_rewrite"}
    if lspecial or rspecial:
        return bool(lspecial & rspecial)

    if indexed_graph_family_signature(lnode) == indexed_graph_family_signature(rnode):
        return True

    try:
        lvars = indexed_expression_orbit_nodes(lnode)
    except Exception:
        lvars = ((lnode, 1),)
    try:
        rvars = indexed_expression_orbit_nodes(rnode)
    except Exception:
        rvars = ((rnode, 1),)
    lsigs = set()
    for v, _ in lvars:
        lsigs.update(indexed_identity_rewrite_signatures(v))
    rsigs = set()
    for v, _ in rvars:
        rsigs.update(indexed_identity_rewrite_signatures(v))
    return bool(lsigs & rsigs)


def indexed_identity_family_signature(node: Any) -> tuple[Any, ...]:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode):
        return tuple()

    families = []

    def visit(cur):
        if not isinstance(cur, SemanticNode):
            return
        if cur.kind == "indexed_tensor":
            original = cur.metadata.get("_original_obj")
            tensor = getattr(original, "tensor", None) if original is not None else None
            md = dict(getattr(tensor, "symmetry_metadata", {}) or {})
            name = cur.value
            variance = cur.metadata.get("variance_spec")
            tags = []

            for key in ("symmetric", "antisymmetric", "cyclic", "pair_symmetric", "pair_antisymmetric"):
                if md.get(key):
                    norm = tuple(_normalize_symmetry_group(g) for g in md.get(key, ()))
                    try:
                        norm = tuple(sorted(norm))
                    except Exception:
                        pass
                    tags.append((key, norm))

            for key in ("riemann", "weyl", "bianchi", "ricci_symmetric", "metric", "epsilon", "delta", "levi_civita"):
                if md.get(key):
                    tags.append((key, True))

            lname = str(name).lower() if name is not None else ""
            if "riemann" in lname or lname == "r":
                tags.append(("riemann_like", True))
            if "weyl" in lname or lname == "c":
                tags.append(("weyl_like", True))
            if "ricci" in lname:
                tags.append(("ricci_like", True))
            if lname in {"g", "metric"}:
                tags.append(("metric_like", True))
            if lname in {"eps", "epsilon"}:
                tags.append(("epsilon_like", True))
            if lname in {"delta", "kronecker"}:
                tags.append(("delta_like", True))

            if tags:
                families.append((name, variance, tuple(sorted(tags))))
        for ch in cur.children:
            visit(ch)

    visit(node)
    return tuple(sorted(families))

def indexed_graph_family_signature(node: Any, *, rounds: int = 4) -> tuple[Any, ...]:
    return (
        indexed_contraction_graph_signature(node, rounds=rounds),
        indexed_identity_family_signature(node),
    )


def _tensor_index_names_from_node(node: Any) -> tuple[str, ...]:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode) or node.kind != "indexed_tensor":
        return tuple()
    out = []
    for ch in node.children:
        if isinstance(ch, SemanticNode) and ch.kind == "tensor_index" and ch.children:
            out.append(str(getattr(ch.children[0], "value", "")))
    return tuple(out)

def _is_riemann_family_node(node: Any) -> bool:
    fam = repr(indexed_identity_family_signature(node)).lower()
    return "riemann" in fam or "bianchi" in fam

def _riemann_bianchi_sum_signature(node: Any) -> tuple[Any, ...] | None:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode) or node.kind != "indexed_add":
        return None
    leaves = [ch for ch in node.children if isinstance(ch, SemanticNode) and ch.kind == "indexed_tensor"]
    if len(leaves) != 3:
        return None
    if not all(_is_riemann_family_node(leaf) for leaf in leaves):
        return None
    names = [leaf.value for leaf in leaves]
    variances = [leaf.metadata.get("variance_spec") for leaf in leaves]
    if len(set(names)) != 1 or len(set(variances)) != 1:
        return None

    tuples = [_tensor_index_names_from_node(leaf) for leaf in leaves]
    if not all(len(t) == 4 for t in tuples):
        return None

    for base in tuples:
        a, b, c, d = base
        expected = {
            (a, b, c, d),
            (a, c, d, b),
            (a, d, b, c),
        }
        if set(tuples) == expected:
            return ("riemann_bianchi_sum", names[0], variances[0], 3)
    return None

def indexed_identity_rewrite_signatures(node: Any, *, rounds: int = 4) -> tuple[tuple[Any, ...], ...]:
    sigs = [indexed_graph_family_signature(node, rounds=rounds)]
    specials = [
        _riemann_bianchi_sum_signature(node),
        _weyl_tracefree_signature(node),
        _ricci_symmetry_signature(node),
        _metric_trace_signature(node),
        _epsilon_delta_signature(node),
        _metric_raise_lower_signature(node),
        _riemann_pair_exchange_signature(node),
        _riemann_antisym_pair_signature(node),
        _ricci_metric_sum_signature(node),
    ]
    for spec in specials:
        if spec is not None:
            sigs.append(("identity_rewrite", spec))
    uniq = list({repr(s): s for s in sigs}.values())
    return tuple(sorted(uniq, key=repr))


def _flatten_indexed_add_terms(node: Any) -> tuple[Any, ...]:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode):
        return tuple()
    if node.kind != "indexed_add":
        return (node,)
    out = []
    for ch in node.children:
        if isinstance(ch, SemanticNode) and ch.kind == "indexed_add":
            out.extend(_flatten_indexed_add_terms(ch))
        else:
            out.append(ch)
    return tuple(out)

def _leaf_tensor_family(node: Any) -> str:
    sig = indexed_identity_family_signature(node)
    text = repr(sig).lower()
    if not sig:
        return ""
    # prefer explicit family tags over substring accidents from chart metadata
    explicit = text
    if "epsilon', true" in explicit or "epsilon_like" in explicit or "levi_civita" in explicit:
        return "epsilon"
    if "delta', true" in explicit or "delta_like" in explicit:
        return "delta"
    if "weyl', true" in explicit or "weyl_like" in explicit:
        return "weyl"
    if "riemann', true" in explicit or "riemann_like" in explicit or "bianchi', true" in explicit:
        return "riemann"
    if "ricci_symmetric" in explicit or "ricci_like" in explicit:
        return "ricci"
    if "metric', true" in explicit or "metric_like" in explicit:
        return "metric"
    return ""

def _weyl_tracefree_signature(node: Any) -> tuple[Any, ...] | None:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode) or node.kind != "indexed_tensor" or _leaf_tensor_family(node) != "weyl":
        return None
    counts = {}
    for nm in _tensor_index_names_from_node(node):
        counts[nm] = counts.get(nm, 0) + 1
    if any(v >= 2 for v in counts.values()):
        return ("weyl_tracefree", node.value, node.metadata.get("variance_spec"))
    return None

def _ricci_symmetry_signature(node: Any) -> tuple[Any, ...] | None:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode) or node.kind != "indexed_tensor" or _leaf_tensor_family(node) != "ricci":
        return None
    names = _tensor_index_names_from_node(node)
    if len(names) != 2:
        return None
    return ("ricci_symmetric", node.value, frozenset(names), node.metadata.get("variance_spec"))

def _metric_trace_signature(node: Any) -> tuple[Any, ...] | None:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode) or node.kind != "indexed_tensor_product":
        return None
    leaves = [ch for ch in node.children if isinstance(ch, SemanticNode) and ch.kind == "indexed_tensor"]
    if len(leaves) != 2:
        return None
    fams = [_leaf_tensor_family(ch) for ch in leaves]
    if "metric" not in fams:
        return None
    metric = leaves[0] if fams[0] == "metric" else leaves[1]
    other = leaves[1] if fams[0] == "metric" else leaves[0]
    shared = tuple(sorted(set(_tensor_index_names_from_node(metric)) & set(_tensor_index_names_from_node(other))))
    if not shared:
        return None
    return ("metric_trace", other.value, other.metadata.get("variance_spec"), len(shared))

def _epsilon_delta_signature(node: Any) -> tuple[Any, ...] | None:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode) or node.kind != "indexed_tensor_product":
        return None
    leaves = [ch for ch in node.children if isinstance(ch, SemanticNode) and ch.kind == "indexed_tensor"]
    if len(leaves) < 2:
        return None
    fams = [_leaf_tensor_family(ch) for ch in leaves]
    if "epsilon" not in fams or "delta" not in fams:
        return None
    return ("epsilon_delta",
            tuple(sorted(ch.value for i, ch in enumerate(leaves) if fams[i] == "epsilon")),
            tuple(sorted(ch.value for i, ch in enumerate(leaves) if fams[i] == "delta")),
            len(leaves))

def _metric_raise_lower_signature(node: Any) -> tuple[Any, ...] | None:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode) or node.kind != "indexed_tensor_product":
        return None
    leaves = [ch for ch in node.children if isinstance(ch, SemanticNode) and ch.kind == "indexed_tensor"]
    if len(leaves) != 2:
        return None
    fams = [_leaf_tensor_family(ch) for ch in leaves]
    if "metric" not in fams:
        return None
    metric = leaves[0] if fams[0] == "metric" else leaves[1]
    other = leaves[1] if fams[0] == "metric" else leaves[0]
    shared = tuple(sorted(set(_tensor_index_names_from_node(metric)) & set(_tensor_index_names_from_node(other))))
    if not shared:
        return None
    mvars = "".join(v for v, _ in [child.value for child in metric.children if isinstance(child.value, tuple)])
    ovars = "".join(v for v, _ in [child.value for child in other.children if isinstance(child.value, tuple)])
    return ("metric_raise_lower", metric.value, mvars, other.value, ovars, len(shared))

def _riemann_pair_exchange_signature(node: Any) -> tuple[Any, ...] | None:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode) or node.kind != "indexed_tensor" or _leaf_tensor_family(node) != "riemann":
        return None
    names = _tensor_index_names_from_node(node)
    if len(names) != 4:
        return None
    a, b, c, d = names
    return ("riemann_pair_family", node.value, frozenset({(a,b,c,d),(c,d,a,b)}), node.metadata.get("variance_spec"))

def _riemann_antisym_pair_signature(node: Any) -> tuple[Any, ...] | None:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode) or node.kind != "indexed_tensor" or _leaf_tensor_family(node) != "riemann":
        return None
    names = _tensor_index_names_from_node(node)
    if len(names) != 4:
        return None
    a, b, c, d = names
    return ("riemann_antisym_family", node.value, frozenset({(a,b,c,d),(b,a,c,d),(a,b,d,c)}), node.metadata.get("variance_spec"))

def _ricci_metric_sum_signature(node: Any) -> tuple[Any, ...] | None:
    from .semantic_core import SemanticNode
    if not isinstance(node, SemanticNode) or node.kind != "indexed_add":
        return None
    leaves = [ch for ch in _flatten_indexed_add_terms(node) if hasattr(ch, "kind")]
    fams = [_leaf_tensor_family(ch) for ch in leaves]
    if not leaves or not ("ricci" in fams and "metric" in fams):
        return None
    return ("ricci_metric_sum_family", tuple(sorted(ch.value for ch in leaves if hasattr(ch, "value"))), len(leaves))
