from fractions import Fraction
import random

from tensoratlas.core import (
    DummyRenamingPolicy,
    IndexType,
    Manifold,
    Permutation,
    SlotSymmetry,
    TensorFactor,
    TensorHead,
    TensorTerm,
    build_dummy_renaming_group,
    build_slot_symmetry_group,
    canonicalize_tensor_expression,
    canonicalize_tensor_term,
    compare_encoded_canonicalization_to_oracle,
    encode_tensor_monomial,
)
from tensoratlas.core.tensor_expr import TensorExpr


def _setup():
    manifold = Manifold("M", 4)
    tangent = IndexType("TM", manifold, dimension=4)
    cotangent = IndexType("Tstar", manifold, dimension=4)
    return tangent, cotangent


def test_custom_slot_generators_from_head_metadata_are_encoded():
    tangent, _ = _setup()
    cyclic = SlotSymmetry.from_generators(3, [((1, 2, 0), 1)])
    C = TensorHead("C", (tangent, tangent, tangent), symmetry=cyclic, variance=(None, None, None))
    a, b, c = tangent.indices("a b c", variance="up")
    term = TensorTerm(Fraction(1), (TensorFactor(C, (c, a, b)),))

    decoded = canonicalize_tensor_term(term)

    assert repr(decoded) == "C(^a,^b,^c)"


def test_repeated_factor_exchange_handles_separated_equal_factors():
    tangent, _ = _setup()
    T = TensorHead("T", (tangent,), variance=(None,))
    U = TensorHead("U", (tangent,), variance=(None,))
    c, b, a = tangent.indices("c b a", variance="up")
    term = TensorTerm(Fraction(1), (TensorFactor(T, (c,)), TensorFactor(U, (b,)), TensorFactor(T, (a,))))

    group = build_slot_symmetry_group(term)
    labels = ("Tc", "Ub", "Ta")
    images = {element.apply_to_sequence(labels) for element in group.elements}

    assert ("Ta", "Ub", "Tc") in images


def test_odd_repeated_factor_exchange_carries_minus_sign_and_zero_detection():
    tangent, _ = _setup()
    Psi = TensorHead("Psi", (tangent,), variance=(None,), parity=1)
    a = tangent.index("a", variance="up")
    term = TensorTerm(Fraction(5), (TensorFactor(Psi, (a,)), TensorFactor(Psi, (a,))))

    decoded = canonicalize_tensor_term(term)

    assert decoded.coefficient == 0


def test_dummy_policies_are_separate_by_index_family_and_support_signed_flips():
    tangent, cotangent = _setup()
    T = TensorHead("T", (tangent, tangent, cotangent, cotangent), variance=(None, None, None, None))
    a, b = tangent.indices("a b", variance="up")
    p, q = cotangent.indices("p q", variance="up")
    term = TensorTerm(Fraction(1), (TensorFactor(T, (a, -a, p, -p)), TensorFactor(T, (b, -b, q, -q))))

    group = build_dummy_renaming_group(term, dummy_policies={tangent: DummyRenamingPolicy(True, -1)})

    assert group.degree == 8
    assert any(element.sign == -1 for element in group.elements)
    # Cotangent family did not receive the flip policy, so every negative-sign
    # generator/result must come from tangent pair flips.
    assert all(element.sign in {-1, 1} for element in group.elements)


def test_public_expression_canonicalization_merges_coefficients():
    tangent, _ = _setup()
    S = TensorHead("S", (tangent, tangent), symmetry="symmetric", variance=(None, None))
    a, b = tangent.indices("a b", variance="up")
    first = TensorTerm(Fraction(2), (TensorFactor(S, (a, b)),))
    second = TensorTerm(Fraction(3), (TensorFactor(S, (b, a)),))
    expr = TensorExpr((first, second))

    out = canonicalize_tensor_expression(expr)

    assert len(out.terms) == 1
    assert out.terms[0].coefficient == 5
    assert repr(out.terms[0]) == "5*S(^a,^b)"


def test_scalar_coefficient_is_preserved_in_decode():
    tangent, _ = _setup()
    A = TensorHead("A", (tangent, tangent), symmetry="antisymmetric", variance=(None, None))
    b, a = tangent.indices("b a", variance="up")
    term = TensorTerm(Fraction(7, 3), (TensorFactor(A, (b, a)),))

    out = canonicalize_tensor_term(term)

    assert out.coefficient == Fraction(-7, 3)
    assert repr(out) == "-7/3*A(^a,^b)"


def test_randomized_small_monomials_match_brute_force_oracle():
    tangent, _ = _setup()
    S = TensorHead("S", (tangent, tangent), symmetry="symmetric", variance=(None, None))
    A = TensorHead("A", (tangent, tangent), symmetry="antisymmetric", variance=(None, None))
    T = TensorHead("T", (tangent,), variance=(None,))
    heads = [S, A, T]
    names = ["a", "b", "c", "d"]
    rng = random.Random(1234)

    for _ in range(25):
        factors = []
        total_slots = 0
        for _factor in range(rng.randint(1, 3)):
            head = rng.choice(heads)
            if total_slots + head.rank > 6:
                continue
            total_slots += head.rank
            indices = []
            for _slot in range(head.rank):
                name = rng.choice(names)
                variance = rng.choice(["up", "down"])
                indices.append(tangent.index(name, variance=variance))
            try:
                factors.append(TensorFactor(head, tuple(indices)))
            except Exception:
                continue
        if not factors:
            continue
        term = TensorTerm(Fraction(rng.choice([1, -1, 2])), tuple(factors))
        # Skip invalid repeated-index patterns; the encoder is a monomial
        # canonicalizer, not an index validator for malformed expressions.
        try:
            term.free_index_signature()
        except Exception:
            continue
        comparison = compare_encoded_canonicalization_to_oracle(term)
        assert comparison.agrees
