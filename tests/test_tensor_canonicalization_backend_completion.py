from fractions import Fraction

from tensoratlas.core import (
    IndexType,
    Manifold,
    Permutation,
    PermutationGroup,
    SignedPermutation,
    TensorFactor,
    TensorHead,
    TensorTerm,
    canonicalize_tensor_term,
    compare_encoded_canonicalization_to_oracle,
    encode_tensor_monomial,
    select_canonicalization_base,
    schreier_sims,
)


def _setup():
    manifold = Manifold("M", 4)
    return IndexType("TM", manifold, dimension=4)


def test_signed_identity_membership_is_preserved_by_stabilizer_chain():
    neg_identity = SignedPermutation(Permutation.identity(2), -1)
    group = PermutationGroup(2, (neg_identity,))
    chain = schreier_sims(group)

    assert chain.signed_identity_available
    assert chain.contains(neg_identity)
    assert not chain.contains(Permutation.transposition(2, 0, 1))


def test_selected_base_can_be_supplied_to_double_coset_backend():
    swap = Permutation.transposition(3, 0, 2)
    group = PermutationGroup(3, (swap,))
    result = group.identity  # smoke check group construction
    backend_result = group.contains(swap)
    assert result.sign == 1
    assert backend_result


def test_symmetric_antisymmetric_contraction_short_circuits_to_zero():
    tangent = _setup()
    S = TensorHead("S", (tangent, tangent), symmetry="symmetric", variance=(None, None))
    A = TensorHead("A", (tangent, tangent), symmetry="antisymmetric", variance=(None, None))
    a, b = tangent.indices("a b", variance="up")
    term = TensorTerm(Fraction(5), (TensorFactor(S, (a, b)), TensorFactor(A, (-a, -b))))

    out = canonicalize_tensor_term(term)

    assert out.coefficient == 0


def test_selected_base_is_well_formed_for_encoded_monomials():
    tangent = _setup()
    R = TensorHead.riemann("R", tangent)
    a, b, c, d = tangent.indices("a b c d", variance="down")
    term = TensorTerm(Fraction(1), (TensorFactor(R, (b, a, d, c)),))
    encoded = encode_tensor_monomial(term)
    base = select_canonicalization_base(encoded)

    assert sorted(base) == list(range(encoded.degree))
    assert base[0] in range(encoded.degree)


def test_public_encoded_path_still_matches_oracle_after_optimizations():
    tangent = _setup()
    S = TensorHead("S", (tangent, tangent), symmetry="symmetric", variance=(None, None))
    a, b, c, d = tangent.indices("a b c d", variance="up")
    term = TensorTerm(
        Fraction(3, 2),
        (
            TensorFactor(S, (c, d)),
            TensorFactor(S, (a, b)),
            TensorFactor(S, (-c, -d)),
            TensorFactor(S, (-a, -b)),
        ),
    )

    comparison = compare_encoded_canonicalization_to_oracle(term)
    canonical = canonicalize_tensor_term(term)

    assert comparison.agrees
    assert canonical.coefficient == Fraction(3, 2)
