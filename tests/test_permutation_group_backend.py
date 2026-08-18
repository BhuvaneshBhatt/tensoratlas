from tensoratlas.core import (
    Permutation,
    PermutationGroup,
    SignedPermutation,
    canonical_double_coset_reference,
    schreier_sims,
)


def test_permutation_composition_inverse_and_sequence_action():
    swap = Permutation.transposition(3, 0, 1)
    cycle = Permutation.cycle(3, 0, 1, 2)
    composed = cycle.compose(swap)

    assert composed.mapping == (2, 1, 0)
    assert composed.inverse().compose(composed) == Permutation.identity(3)
    assert swap.apply_to_sequence(("a", "b", "c")) == ("b", "a", "c")


def test_signed_permutation_composition_tracks_signs():
    odd_swap = SignedPermutation(Permutation.transposition(3, 0, 1), -1)
    cycle = SignedPermutation(Permutation.cycle(3, 0, 1, 2), 1)

    result = cycle.compose(odd_swap)

    assert result.sign == -1
    assert result.permutation.mapping == (2, 1, 0)


def test_permutation_group_closure_orbit_and_stabilizer():
    group = PermutationGroup.symmetric(3)

    assert group.order == 6
    assert group.orbit(0) == (0, 1, 2)
    assert group.stabilizer(0).order == 2
    assert group.contains(Permutation.cycle(3, 0, 1, 2))


def test_schreier_sims_returns_exact_chain_data_for_small_group():
    group = PermutationGroup.symmetric(3)
    chain = schreier_sims(group, base=(0, 1))

    assert chain.base == (0, 1)
    assert chain.group_order == 6
    assert chain.orders == (6, 2, 1)
    assert chain.orbits == ((0, 1, 2), (1, 2))
    assert chain.transversals[0].keys() >= {0, 1, 2}


def test_canonical_double_coset_chooses_lexicographic_label_image():
    left = PermutationGroup.symmetric(3, points=(0, 1))
    right = PermutationGroup.trivial(3)
    representative = Permutation.identity(3)

    result = canonical_double_coset_reference(left, representative, right, labels=("b", "a", "c"))

    assert result.image == ("a", "b", "c")
    assert result.zero is False
    assert result.candidates_considered == 2


def test_canonical_double_coset_detects_signed_cancellation():
    left = PermutationGroup(
        2,
        [SignedPermutation(Permutation.transposition(2, 0, 1), -1)],
    )
    right = PermutationGroup.trivial(2)

    result = canonical_double_coset_reference(left, Permutation.identity(2), right, labels=("x", "x"))

    assert result.zero is True
    assert result.sign == 0
    assert result.canonical is None


def _brute_force_oracle(left_group, representative, right_group, labels=None):
    best_image = None
    best_signs = set()
    best_candidate = None
    middle = representative if isinstance(representative, SignedPermutation) else SignedPermutation(representative, 1)
    labels = tuple(range(left_group.degree)) if labels is None else tuple(labels)
    count = 0
    for left in left_group.elements:
        for right in right_group.elements:
            candidate = left.compose(middle).compose(right)
            image = candidate.apply_to_sequence(labels)
            count += 1
            if best_image is None or image < best_image:
                best_image = image
                best_candidate = candidate
                best_signs = {candidate.sign}
            elif image == best_image:
                best_signs.add(candidate.sign)
    return best_image, best_signs, best_candidate, count


def test_stabilizer_chain_sift_membership_and_coset_representatives():
    group = PermutationGroup.symmetric(4)
    chain = schreier_sims(group, base=(0, 1, 2))
    element = SignedPermutation(Permutation.cycle(4, 0, 1, 2, 3), 1)

    sifted = chain.sift(element)

    assert sifted.success is True
    assert chain.contains(element)
    assert chain.membership_test(element)
    assert chain.coset_representative(0, 3).permutation.apply(0) == 3


def test_stabilizer_chain_rejects_wrong_signed_lift():
    unsigned_group = PermutationGroup.symmetric(3)
    chain = schreier_sims(unsigned_group, base=(0, 1))
    wrong_sign = SignedPermutation(Permutation.cycle(3, 0, 1, 2), -1)

    assert chain.contains(wrong_sign) is False
    assert unsigned_group.contains(wrong_sign) is False


def test_schreier_generators_fix_base_point():
    group = PermutationGroup.symmetric(4)
    generators = group.stabilizer(0).generators

    assert generators
    assert all(generator.permutation.apply(0) == 0 for generator in generators)
    assert group.stabilizer(0).order == 6


def test_signed_generators_are_group_metadata():
    group = PermutationGroup.antisymmetric(3)

    assert any(generator.sign == -1 for generator in group.signed_generators)
    assert all(generator.sign == 1 for generator in PermutationGroup.symmetric(3).signed_generators)


def test_canonical_double_coset_matches_independent_bruteforce_oracle():
    left = PermutationGroup.antisymmetric(4, points=(0, 1, 2))
    right = PermutationGroup.symmetric(4, points=(2, 3))
    representative = Permutation.cycle(4, 0, 2, 3)
    labels = ("c", "a", "b", "a")

    result = canonical_double_coset_reference(left, representative, right, labels=labels)
    image, signs, candidate, count = _brute_force_oracle(left, representative, right, labels)

    assert result.image == image
    assert result.candidates_considered == count
    assert result.zero == (len(signs) > 1)
    if not result.zero:
        assert result.sign == next(iter(signs))
        assert result.canonical.apply_to_sequence(labels) == candidate.apply_to_sequence(labels)


def test_backend_protocol_default_backend_and_reference_oracle():
    from tensoratlas.core import (
        CanonicalizationBackend,
        default_permutation_backend,
        brute_force_double_coset,
    )

    backend = default_permutation_backend()
    assert isinstance(backend, CanonicalizationBackend)

    left = PermutationGroup.symmetric(3, points=(0, 1))
    right = PermutationGroup.symmetric(3, points=(1, 2))
    representative = Permutation.identity(3)
    labels = ("b", "c", "a")

    assert backend.canonicalize_double_coset(left, representative, right, labels=labels) == brute_force_double_coset(
        left,
        representative,
        right,
        labels=labels,
    )
