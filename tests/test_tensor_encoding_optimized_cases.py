from fractions import Fraction

from tensoratlas.core import (
    CommonCaseAnalysis,
    IndexType,
    Manifold,
    SlotSymmetry,
    TensorFactor,
    TensorHead,
    TensorTerm,
    analyze_common_canonicalization_cases,
    canonicalize_repeated_factors,
    canonicalize_tensor_term,
    canonicalize_total_symmetry_blocks,
    compare_encoded_canonicalization_to_oracle,
    decompose_dummy_pair_blocks,
    encode_tensor_monomial,
    repeated_factor_blocks,
    select_canonicalization_base,
)


def _setup():
    manifold = Manifold("M", 4)
    tangent = IndexType("TM", manifold, dimension=4)
    return tangent


def test_total_antisymmetric_repeated_label_short_circuits_to_zero():
    tangent = _setup()
    A = TensorHead("A", (tangent, tangent, tangent), symmetry="antisymmetric", variance=(None, None, None))
    a, b = tangent.indices("a b", variance="up")
    term = TensorTerm(Fraction(3), (TensorFactor(A, (b, a, a)),))

    encoded = encode_tensor_monomial(term)
    analysis = analyze_common_canonicalization_cases(encoded)
    out = canonicalize_tensor_term(term)

    assert analysis.zero
    assert "antisymmetric" in analysis.reason
    assert out.coefficient == 0


def test_total_symmetric_and_antisymmetric_blocks_are_sorted_locally():
    tangent = _setup()
    S = TensorHead("S", (tangent, tangent, tangent), symmetry="symmetric", variance=(None, None, None))
    A = TensorHead("A", (tangent, tangent), symmetry="antisymmetric", variance=(None, None))
    c, b, a = tangent.indices("c b a", variance="up")
    term = TensorTerm(Fraction(2), (TensorFactor(S, (c, a, b)), TensorFactor(A, (b, a))))

    out = canonicalize_total_symmetry_blocks(term)

    assert out.coefficient == Fraction(-2)
    assert repr(out) == "-2*S(^a,^b,^c)*A(^a,^b)"


def test_dummy_pair_blocks_are_decomposed_by_index_family():
    tangent = _setup()
    T = TensorHead("T", (tangent, tangent, tangent, tangent), variance=(None, None, None, None))
    a, b = tangent.indices("a b", variance="up")
    term = TensorTerm(Fraction(1), (TensorFactor(T, (a, -a, b, -b)),))

    blocks = decompose_dummy_pair_blocks(term)

    assert [(block.name, block.up_position, block.down_position) for block in blocks] == [("a", 0, 1), ("b", 2, 3)]


def test_repeated_factor_blocks_and_canonicalization_cover_nonadjacent_factors():
    tangent = _setup()
    T = TensorHead("T", (tangent,), variance=(None,))
    U = TensorHead("U", (tangent,), variance=(None,))
    c, b, a = tangent.indices("c b a", variance="up")
    term = TensorTerm(Fraction(1), (TensorFactor(T, (c,)), TensorFactor(U, (b,)), TensorFactor(T, (a,))))

    blocks = repeated_factor_blocks(term)
    out = canonicalize_repeated_factors(term)

    assert blocks == ((0, 2),)
    assert repr(out) == "T(^a)*T(^c)*U(^b)"


def test_odd_repeated_factor_is_early_zero_without_double_coset_search():
    tangent = _setup()
    Psi = TensorHead("Psi", (tangent,), variance=(None,), parity=1)
    a = tangent.index("a", variance="up")
    term = TensorTerm(Fraction(1), (TensorFactor(Psi, (a,)), TensorFactor(Psi, (a,))))

    encoded = encode_tensor_monomial(term)
    analysis = analyze_common_canonicalization_cases(encoded)
    decoded = canonicalize_tensor_term(term)

    assert analysis.zero
    assert "odd factor" in analysis.reason
    assert decoded.coefficient == 0


def test_base_selection_prioritizes_free_indices_and_dummy_representatives():
    tangent = _setup()
    T = TensorHead("T", (tangent, tangent, tangent), variance=(None, None, None))
    f, a = tangent.indices("f a", variance="up")
    term = TensorTerm(Fraction(1), (TensorFactor(T, (a, -a, f)),))
    encoded = encode_tensor_monomial(term)
    blocks = decompose_dummy_pair_blocks(term)

    base = select_canonicalization_base(encoded, dummy_blocks=blocks)

    assert base[0] == 2
    assert 0 in base[:2]
    assert sorted(base) == [0, 1, 2]


def test_custom_negative_generator_fixed_image_early_zero():
    tangent = _setup()
    sym = SlotSymmetry.from_generators(2, [((0, 1), -1)])
    A = TensorHead("A", (tangent, tangent), symmetry=sym, variance=(None, None))
    a = tangent.index("a", variance="up")
    term = TensorTerm(Fraction(1), (TensorFactor(A, (a, a)),))

    analysis = analyze_common_canonicalization_cases(encode_tensor_monomial(term))

    assert analysis.zero
    assert "negative slot symmetry" in analysis.reason


def test_optimized_path_still_matches_bruteforce_oracle_for_small_case():
    tangent = _setup()
    R = TensorHead.riemann("R", tangent)
    a, b, c, d = tangent.indices("a b c d", variance="down")
    term = TensorTerm(Fraction(1), (TensorFactor(R, (b, a, d, c)),))

    oracle = compare_encoded_canonicalization_to_oracle(term)
    out = canonicalize_tensor_term(term)

    assert oracle.agrees
    assert repr(out) == "R(_a,_b,_c,_d)"
