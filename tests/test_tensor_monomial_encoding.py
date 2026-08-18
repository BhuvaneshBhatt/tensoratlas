from fractions import Fraction

from tensoratlas.core import (
    Manifold,
    IndexType,
    TensorHead,
    TensorTerm,
    TensorFactor,
    Permutation,
    SignedPermutation,
    PermutationGroup,
    build_dummy_renaming_group,
    build_slot_symmetry_group,
    canonical_double_coset_reference,
    canonicalize_encoded_monomial,
    decode_canonical_result,
    encode_tensor_monomial,
)


def _setup():
    manifold = Manifold("M", 4)
    tangent = IndexType("TM", manifold, dimension=4)
    return tangent


def test_build_slot_symmetry_group_encodes_riemann_pair_symmetries():
    tangent = _setup()
    R = TensorHead.riemann("R", tangent, variance=("down",) * 4)
    a, b, c, d = tangent.indices("a b c d", variance="down")
    term = TensorTerm(Fraction(1), (TensorFactor(R, (a, b, c, d)),))

    group = build_slot_symmetry_group(term)

    assert group.degree == 4
    assert SignedPermutation(Permutation.transposition(4, 0, 1), -1) in group.elements
    assert SignedPermutation(Permutation.transposition(4, 2, 3), -1) in group.elements


def test_build_dummy_renaming_group_exchanges_dummy_pairs_by_variance():
    tangent = _setup()
    T = TensorHead("T", (tangent, tangent), variance=(None, None))
    a, b = tangent.indices("a b", variance="up")
    ad, bd = -a, -b
    term = TensorTerm(Fraction(1), (TensorFactor(T, (a, ad)), TensorFactor(T, (b, bd))))

    group = build_dummy_renaming_group(term)

    assert group.degree == 4
    labels = ("a^", "a_", "b^", "b_")
    images = {element.apply_to_sequence(labels) for element in group.elements}
    assert ("b^", "b_", "a^", "a_") in images


def test_encode_decode_canonical_result_for_symmetric_slots():
    tangent = _setup()
    S = TensorHead("S", (tangent, tangent), symmetry="symmetric", variance=(None, None))
    a, b = tangent.indices("a b", variance="up")
    term = TensorTerm(Fraction(3), (TensorFactor(S, (b, a)),))
    encoded = encode_tensor_monomial(term)

    result = canonical_double_coset_reference(
        encoded.dummy_group,
        Permutation.identity(encoded.degree),
        encoded.slot_group,
        labels=encoded.labels,
    )
    decoded = decode_canonical_result(encoded, result)

    assert not decoded.zero
    assert decoded.term.coefficient == 3
    assert repr(decoded.term) == "3*S(^a,^b)"


def test_canonicalize_encoded_monomial_detects_antisymmetric_repeated_zero():
    tangent = _setup()
    A = TensorHead("A", (tangent, tangent), symmetry="antisymmetric", variance=(None, None))
    a = tangent.index("a", variance="up")
    term = TensorTerm(Fraction(1), (TensorFactor(A, (a, a)),))

    decoded = canonicalize_encoded_monomial(term)

    assert decoded.zero
    assert decoded.term.coefficient == 0


def test_canonicalize_encoded_monomial_merges_dummy_pair_ordering():
    tangent = _setup()
    T = TensorHead("T", (tangent, tangent), variance=(None, None))
    a, b = tangent.indices("a b", variance="up")
    ad, bd = -a, -b
    first = TensorTerm(Fraction(1), (TensorFactor(T, (a, ad)), TensorFactor(T, (b, bd))))
    second = TensorTerm(Fraction(1), (TensorFactor(T, (b, bd)), TensorFactor(T, (a, ad))))

    canon_first = canonicalize_encoded_monomial(first).term
    canon_second = canonicalize_encoded_monomial(second).term

    assert canon_first.key() == canon_second.key()
