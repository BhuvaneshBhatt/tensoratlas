from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import sympy as sp
from sympy.core.sorting import default_sort_key
from sympy.polys.rootoftools import ComplexRootOf, RootOf
from sympy.simplify.sqrtdenest import sqrtdenest

from .cache_utils import BoundedCache
from .simplification_core import light_simplify, canonical_simplify, sympify_expr


@dataclass(frozen=True)
class BasicAlgebraicReductionLimits:
    max_ops: int = 200
    max_atoms: int = 8
    max_minpoly_degree: int = 24
    max_minpoly_generators: int = 6


DEFAULT_BASIC_ALGEBRAIC_LIMITS = BasicAlgebraicReductionLimits()
_SPECIAL_TRIG_HEADS = (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)
_NEGATIVE_CACHE_SENTINEL = object()
_BASIC_ROOT_CACHE: BoundedCache[tuple[str, BasicAlgebraicReductionLimits], object] = BoundedCache(maxsize=4096)


def _safe_sympify(expr: Any) -> sp.Expr:
    return sympify_expr(expr)


def _is_explicit_algebraic_atom(e: sp.Expr) -> bool:
    if e in (sp.I, sp.GoldenRatio):
        return True
    if isinstance(e, (RootOf, ComplexRootOf, sp.AlgebraicNumber)):
        return True
    if e.is_Integer or e.is_Rational:
        return True
    if e.is_number:
        return bool(getattr(e, "is_algebraic", False)) and not bool(getattr(e, "is_transcendental", False))
    return False


def _is_special_angle_algebraic(e: sp.Expr) -> bool:
    if e.func in _SPECIAL_TRIG_HEADS and len(e.args) == 1:
        try:
            q = sp.cancel(e.args[0] / sp.pi)
        except Exception:
            q = None
        return bool(getattr(q, "is_rational", False))
    if e.func == sp.exp and len(e.args) == 1:
        try:
            q = sp.cancel(e.args[0] / (sp.I * sp.pi))
        except Exception:
            q = None
        return bool(getattr(q, "is_rational", False))
    return False


def _is_rational_power_of_algebraic(base: sp.Expr, exp: sp.Expr, *, nested: bool) -> bool:
    if exp.is_rational is not True:
        return False
    if exp.is_negative and getattr(base, "is_zero", None) is True:
        return False
    return is_nested_algebraic_form(base) if nested else is_plain_algebraic_form(base)


def is_plain_algebraic_form(expr: Any) -> bool:
    e = _safe_sympify(expr)
    if _is_explicit_algebraic_atom(e):
        return True
    if e.is_Atom:
        return False
    if _is_special_angle_algebraic(e):
        return True
    if isinstance(e, (sp.Add, sp.Mul)):
        return all(is_plain_algebraic_form(arg) for arg in e.args)
    if isinstance(e, sp.Pow):
        base, exp = e.as_base_exp()
        return _is_rational_power_of_algebraic(base, exp, nested=False)
    return False


def is_nested_algebraic_form(expr: Any) -> bool:
    e = _safe_sympify(expr)
    if is_plain_algebraic_form(e):
        return True
    if e.is_Atom:
        return False
    if _is_special_angle_algebraic(e):
        return True
    if isinstance(e, (sp.Add, sp.Mul)):
        return all(is_nested_algebraic_form(arg) for arg in e.args)
    if isinstance(e, sp.Pow):
        base, exp = e.as_base_exp()
        return _is_rational_power_of_algebraic(base, exp, nested=True)
    return False


def _special_angle_rewrite_once(expr: sp.Expr) -> sp.Expr:
    if expr.func in _SPECIAL_TRIG_HEADS and len(expr.args) == 1:
        try:
            q = sp.cancel(expr.args[0] / sp.pi)
        except Exception:
            q = None
        if getattr(q, "is_rational", False):
            try:
                return canonical_simplify(sp.expand_func(expr), final=False)
            except Exception:
                return expr
    if expr.func == sp.exp and len(expr.args) == 1:
        try:
            q = sp.cancel(expr.args[0] / (sp.I * sp.pi))
        except Exception:
            q = None
        if getattr(q, "is_rational", False):
            try:
                return canonical_simplify(sp.expand_func(expr.rewrite(sp.cos)), final=False)
            except Exception:
                return expr
    return expr


@lru_cache(maxsize=4096)
def _rewrite_special_angle_cached(expr: sp.Expr) -> sp.Expr:
    if expr.is_Atom:
        return expr
    new_args = tuple(_rewrite_special_angle_cached(arg) for arg in expr.args)
    rebuilt = expr.func(*new_args) if new_args != expr.args else expr
    return _special_angle_rewrite_once(rebuilt)


def rewrite_special_angle_forms(expr: Any) -> sp.Expr:
    return _rewrite_special_angle_cached(_safe_sympify(expr))


def _algebraic_simplify_expr(expr: sp.Expr, *, level: str = "normal") -> sp.Expr:
    if level == "cheap":
        return light_simplify(expr)
    expr = canonical_simplify(expr, final=False)
    if level == "strong":
        return canonical_simplify(expr, final=True)
    return expr


def _normalize_algebraic_forms_impl(expr: sp.Expr, *, already_rewritten: bool = False) -> sp.Expr:
    e = expr if already_rewritten else rewrite_special_angle_forms(expr)
    e = _algebraic_simplify_expr(e, level="cheap")
    for func in (lambda x: sp.powsimp(x, force=False), sqrtdenest, sp.radsimp):
        try:
            e2 = func(e)
            e = _algebraic_simplify_expr(e2, level="normal")
        except Exception:
            pass
    try:
        e = _algebraic_simplify_expr(sp.cancel(sp.together(e)), level="normal")
    except Exception:
        pass
    return e


def normalize_algebraic_forms(expr: Any) -> sp.Expr:
    return _normalize_algebraic_forms_impl(_safe_sympify(expr), already_rewritten=False)


def _atom_sort_key(expr: sp.Expr):
    return default_sort_key(expr)


def _extract_algebraic_atoms_impl(expr: sp.Expr, *, already_rewritten: bool = False) -> tuple[sp.Expr, ...]:
    e = expr if already_rewritten else rewrite_special_angle_forms(expr)
    atoms: set[sp.Expr] = set()
    for sub in sp.preorder_traversal(e):
        if _is_explicit_algebraic_atom(sub):
            atoms.add(sub)
            continue
        if isinstance(sub, sp.Pow):
            base, exp = sub.as_base_exp()
            if _is_rational_power_of_algebraic(base, exp, nested=True):
                atoms.add(sub)
                continue
        if _is_special_angle_algebraic(sub):
            atoms.add(_special_angle_rewrite_once(sub))
    return tuple(sorted(atoms, key=_atom_sort_key))


def extract_algebraic_atoms(expr: Any) -> tuple[sp.Expr, ...]:
    return _extract_algebraic_atoms_impl(_safe_sympify(expr), already_rewritten=False)


def _bounded_common_field(expr: sp.Expr, atoms: tuple[sp.Expr, ...], *, limits: BasicAlgebraicReductionLimits) -> sp.Expr:
    if not atoms:
        return expr
    if len(atoms) > limits.max_atoms:
        return expr
    if sp.count_ops(expr, visual=False) > limits.max_ops:
        return expr
    try:
        primitive = sp.to_number_field(list(atoms))
        lifted = sp.to_number_field(expr, primitive)
        out = lifted.as_expr() if hasattr(lifted, "as_expr") else sp.sympify(lifted)
        return _algebraic_simplify_expr(out, level="normal")
    except Exception:
        return expr


def _root_family_key(root: sp.Expr) -> tuple[str, str] | None:
    if not isinstance(root, (RootOf, ComplexRootOf)):
        return None
    try:
        poly = root.poly
        return (sp.srepr(poly.as_expr()), str(poly.gen))
    except Exception:
        return None


def _complete_root_family_signature(root: sp.Expr):
    if not isinstance(root, (RootOf, ComplexRootOf)):
        return None
    try:
        poly = root.poly
        degree = poly.degree()
        if degree is None:
            return None
        return (_root_family_key(root), degree, int(root.index))
    except Exception:
        return None


def _vieta_sum_for_root(root: sp.Expr) -> sp.Expr | None:
    try:
        poly = root.poly
        coeffs = poly.all_coeffs()
        if len(coeffs) < 2:
            return None
        return sp.cancel(-coeffs[1] / coeffs[0])
    except Exception:
        return None


def _vieta_product_for_root(root: sp.Expr) -> sp.Expr | None:
    try:
        poly = root.poly
        coeffs = poly.all_coeffs()
        degree = poly.degree()
        if degree is None or not coeffs:
            return None
        return sp.cancel(((-1) ** int(degree)) * coeffs[-1] / coeffs[0])
    except Exception:
        return None


def compress_root_sums(expr: Any) -> sp.Expr:
    e = _safe_sympify(expr)
    if not isinstance(e, sp.Add):
        return e
    terms = list(sp.Add.make_args(e))
    groups: dict[tuple[str, str], list[tuple[int, sp.Expr]]] = defaultdict(list)
    for idx, term in enumerate(terms):
        if isinstance(term, (RootOf, ComplexRootOf)):
            key = _root_family_key(term)
            if key is not None:
                groups[key].append((idx, term))
    replaced: set[int] = set()
    extras: list[sp.Expr] = []
    for items in groups.values():
        if not items:
            continue
        root0 = items[0][1]
        sigs = [_complete_root_family_signature(r) for _, r in items]
        if any(sig is None for sig in sigs):
            continue
        degree = sigs[0][1]
        if len(items) != degree:
            continue
        indices = sorted(sig[2] for sig in sigs)
        if indices != list(range(degree)):
            continue
        repl = _vieta_sum_for_root(root0)
        if repl is None:
            continue
        replaced.update(idx for idx, _ in items)
        extras.append(repl)
    if not replaced:
        return e
    new_terms = [term for idx, term in enumerate(terms) if idx not in replaced] + extras
    if not new_terms:
        return sp.Integer(0)
    return _algebraic_simplify_expr(sp.Add(*new_terms), level="normal")


def compress_root_products(expr: Any) -> sp.Expr:
    e = _safe_sympify(expr)
    if not isinstance(e, sp.Mul):
        return e
    factors = list(sp.Mul.make_args(e))
    groups: dict[tuple[str, str], list[tuple[int, sp.Expr]]] = defaultdict(list)
    for idx, factor in enumerate(factors):
        if isinstance(factor, (RootOf, ComplexRootOf)):
            key = _root_family_key(factor)
            if key is not None:
                groups[key].append((idx, factor))
    replaced: set[int] = set()
    extras: list[sp.Expr] = []
    for items in groups.values():
        if not items:
            continue
        root0 = items[0][1]
        sigs = [_complete_root_family_signature(r) for _, r in items]
        if any(sig is None for sig in sigs):
            continue
        degree = sigs[0][1]
        if len(items) != degree:
            continue
        indices = sorted(sig[2] for sig in sigs)
        if indices != list(range(degree)):
            continue
        repl = _vieta_product_for_root(root0)
        if repl is None:
            continue
        replaced.update(idx for idx, _ in items)
        extras.append(repl)
    if not replaced:
        return e
    new_factors = [factor for idx, factor in enumerate(factors) if idx not in replaced] + extras
    if not new_factors:
        return sp.Integer(1)
    return _algebraic_simplify_expr(sp.Mul(*new_factors), level="normal")


def _polynomialize_algebraic_expr(expr: sp.Expr, *, limits: BasicAlgebraicReductionLimits):
    atoms = _extract_algebraic_atoms_impl(expr, already_rewritten=True)
    if not atoms or len(atoms) > limits.max_minpoly_generators:
        return None
    syms = sp.symbols(f"_taa0:{len(atoms)}")
    try:
        num, den = sp.fraction(sp.together(expr.xreplace(dict(zip(atoms, syms)))))
    except Exception:
        return None
    polys = []
    for atom, sym in zip(atoms, syms):
        try:
            poly = sp.minimal_polynomial(atom, sym)
        except Exception:
            return None
        try:
            deg = sp.Poly(poly, sym).degree()
        except Exception:
            return None
        if deg is None or deg > limits.max_minpoly_degree:
            return None
        polys.append(poly)
    return atoms, syms, num, den, polys


def reduce_with_minpoly_relations(expr: Any, *, limits: BasicAlgebraicReductionLimits = DEFAULT_BASIC_ALGEBRAIC_LIMITS) -> sp.Expr | None:
    e = normalize_algebraic_forms(expr)
    packed = _polynomialize_algebraic_expr(e, limits=limits)
    if packed is None:
        return None
    atoms, syms, num, den, polys = packed
    try:
        basis = sp.groebner(list(polys), *syms, order="lex")
        num_rem = basis.reduce(num)[1]
        den_rem = basis.reduce(den)[1]
    except Exception:
        return None
    if num_rem == 0 and den_rem != 0:
        return sp.Integer(0)
    try:
        reduced = sp.cancel(sp.together(num_rem / den_rem))
    except Exception:
        return None
    out = reduced.xreplace(dict(zip(syms, atoms)))
    return _algebraic_simplify_expr(out, level="normal")


def prove_zero_via_minpolys(expr: Any, *, limits: BasicAlgebraicReductionLimits = DEFAULT_BASIC_ALGEBRAIC_LIMITS) -> bool | None:
    reduced = reduce_with_minpoly_relations(expr, limits=limits)
    if reduced is None:
        return None
    if reduced == 0 or getattr(reduced, "is_zero", None) is True:
        return True
    if getattr(reduced, "is_zero", None) is False:
        return False
    try:
        x = sp.Symbol("_tax")
        mp = sp.minimal_polynomial(reduced, x)
        if mp == x:
            return True
        if getattr(mp, "is_zero", None) is False:
            return False
    except Exception:
        pass
    return None


def _canonical_cache_key(expr: sp.Expr, limits: BasicAlgebraicReductionLimits) -> tuple[str, BasicAlgebraicReductionLimits]:
    return (sp.srepr(expr), limits)


def basic_root_canonicalize(expr: Any, *, limits: BasicAlgebraicReductionLimits = DEFAULT_BASIC_ALGEBRAIC_LIMITS) -> sp.Expr | None:
    original = _safe_sympify(expr)
    rewritten = rewrite_special_angle_forms(original)
    key = _canonical_cache_key(rewritten, limits)
    cached = _BASIC_ROOT_CACHE.get(key)
    if cached is not None:
        return None if cached is _NEGATIVE_CACHE_SENTINEL else cached  # type: ignore[return-value]
    if not is_nested_algebraic_form(rewritten):
        _BASIC_ROOT_CACHE[key] = _NEGATIVE_CACHE_SENTINEL
        return None
    out = _normalize_algebraic_forms_impl(rewritten, already_rewritten=True)
    out = compress_root_sums(out)
    out = compress_root_products(out)
    atoms = _extract_algebraic_atoms_impl(out, already_rewritten=True)
    out = _bounded_common_field(out, atoms, limits=limits)
    minpoly_reduced = reduce_with_minpoly_relations(out, limits=limits)
    if minpoly_reduced is not None:
        out = minpoly_reduced
    try:
        out = sp.factor_terms(out)
    except Exception:
        pass
    out = _algebraic_simplify_expr(out, level="normal")
    try:
        out = sqrtdenest(out)
    except Exception:
        pass
    out = _algebraic_simplify_expr(out, level="normal")
    _BASIC_ROOT_CACHE[key] = out
    return out


def algebraically_equal_basic(left: Any, right: Any, *, limits: BasicAlgebraicReductionLimits = DEFAULT_BASIC_ALGEBRAIC_LIMITS) -> bool | None:
    try:
        diff = _safe_sympify(left) - _safe_sympify(right)
    except Exception:
        return None
    rewritten = rewrite_special_angle_forms(diff)
    if not is_nested_algebraic_form(rewritten):
        return None
    reduced = basic_root_canonicalize(rewritten, limits=limits)
    if reduced is None:
        return prove_zero_via_minpolys(rewritten, limits=limits)
    if reduced == 0 or getattr(reduced, "is_zero", None) is True:
        return True
    if getattr(reduced, "is_zero", None) is False:
        return False
    return prove_zero_via_minpolys(reduced, limits=limits)


def clear_basic_root_cache() -> None:
    _BASIC_ROOT_CACHE.clear()


def basic_root_cache_stats() -> dict[str, int]:
    return _BASIC_ROOT_CACHE.stats()


__all__ = [
    "BasicAlgebraicReductionLimits",
    "DEFAULT_BASIC_ALGEBRAIC_LIMITS",
    "is_plain_algebraic_form",
    "is_nested_algebraic_form",
    "rewrite_special_angle_forms",
    "normalize_algebraic_forms",
    "extract_algebraic_atoms",
    "reduce_with_minpoly_relations",
    "prove_zero_via_minpolys",
    "compress_root_sums",
    "compress_root_products",
    "basic_root_canonicalize",
    "algebraically_equal_basic",
    "clear_basic_root_cache",
    "basic_root_cache_stats",
]
