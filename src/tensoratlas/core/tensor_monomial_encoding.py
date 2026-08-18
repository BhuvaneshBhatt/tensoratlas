"""Encode semantic tensor monomials for permutation-group canonicalization.

This module is the bridge between TensorAtlas' semantic tensor IR and the
permutation-group backend.  It builds slot-symmetry and dummy-renaming groups,
encodes indexed factors as slot-label configurations, invokes a backend, and
reconstructs semantic tensor terms from canonical images.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Iterable, Mapping, Sequence

from .indices import AbstractIndex, IndexType
from .manifolds import TensorKernelError
from .permutation_group_backend import (
    CanonicalDoubleCosetResult,
    CanonicalizationBackend,
    Permutation,
    PermutationGroup,
    SignedPermutation,
    brute_force_double_coset,
    default_permutation_backend,
)
from .tensor_expr import TensorExpr, TensorFactor, TensorTerm

VarianceCode = int
SlotLabel = tuple[str, tuple, int, str, VarianceCode]

_FREE_LABEL = 0
_DUMMY_LABEL = 1
_UP = 0
_DOWN = 1


@dataclass(frozen=True, slots=True)
class SlotMetadata:
    """Metadata for one flattened tensor slot."""

    factor_position: int
    slot_position: int
    index_type: IndexType
    variance: str


@dataclass(frozen=True, slots=True)
class DummyRenamingPolicy:
    """Conventions for dummy-pair renaming within one index family.

    ``allow_pair_flip`` permits exchanging the two slots of one dummy pair.
    This is disabled by default because up/down contractions normally have a
    fixed variance orientation.  If a symplectic or nonstandard metric
    convention allows pair flips, ``pair_flip_sign`` records the sign.
    """

    allow_pair_flip: bool = False
    pair_flip_sign: int = 1

    def __post_init__(self) -> None:
        if self.pair_flip_sign not in {-1, 1}:
            raise TensorKernelError("Dummy-pair flip sign must be +1 or -1.")


@dataclass(frozen=True, slots=True)
class EncodedTensorMonomial:
    """A tensor monomial encoded as a permutation canonicalization problem."""

    term: TensorTerm
    labels: tuple[SlotLabel, ...]
    slot_metadata: tuple[SlotMetadata, ...]
    slot_group: PermutationGroup
    dummy_group: PermutationGroup

    @property
    def degree(self) -> int:
        return len(self.labels)


@dataclass(frozen=True, slots=True)
class DecodedCanonicalMonomial:
    """Decoded result of a permutation-level monomial canonicalization."""

    term: TensorTerm
    zero: bool
    sign: int
    backend_result: CanonicalDoubleCosetResult


@dataclass(frozen=True, slots=True)
class TensorCanonicalizationResult:
    """User-facing canonicalization result with diagnostics."""

    expression: TensorExpr
    backend_name: str
    warnings: tuple[str, ...] = ()
    trace: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TensorCanonicalizationOracleResult:
    """Comparison between encoded canonicalization and brute-force oracle."""

    encoded: EncodedTensorMonomial
    backend_result: CanonicalDoubleCosetResult
    oracle_result: CanonicalDoubleCosetResult

    @property
    def agrees(self) -> bool:
        return (
            self.backend_result.image == self.oracle_result.image
            and self.backend_result.sign == self.oracle_result.sign
            and self.backend_result.zero == self.oracle_result.zero
        )



@dataclass(frozen=True, slots=True)
class DummyPairBlock:
    """Dummy-pair positions for one index family."""

    index_type: IndexType
    name: str
    up_position: int
    down_position: int


@dataclass(frozen=True, slots=True)
class CommonCaseAnalysis:
    """Common-case canonicalization information for one tensor monomial."""

    zero: bool
    reason: str | None
    dummy_blocks: tuple[DummyPairBlock, ...]
    repeated_factor_blocks: tuple[tuple[int, ...], ...]
    base: tuple[int, ...]



def encode_tensor_monomial(
    term: TensorTerm,
    *,
    dummy_policies: Mapping[IndexType, DummyRenamingPolicy] | None = None,
) -> EncodedTensorMonomial:
    """Encode a semantic tensor term for slot/dummy canonicalization."""
    if term.coefficient == 0:
        return EncodedTensorMonomial(
            term=term,
            labels=(),
            slot_metadata=(),
            slot_group=PermutationGroup.trivial(0),
            dummy_group=PermutationGroup.trivial(0),
        )
    validate_tensor_monomial_indices(term)
    labels = _labels_for_term(term)
    metadata = _slot_metadata(term)
    slot_group = build_slot_symmetry_group(term)
    dummy_group = build_dummy_renaming_group(term, dummy_policies=dummy_policies)
    return EncodedTensorMonomial(term, labels, metadata, slot_group, dummy_group)


def build_slot_symmetry_group(term: TensorTerm) -> PermutationGroup:
    """Build the signed slot-symmetry group for a tensor monomial.

    The group includes arbitrary signed local-slot generators declared on each
    tensor head, built-in monoterm symmetries, and the full repeated-factor
    exchange group for equivalent commutative tensor factors.  Factor-exchange
    signs follow the tensor-head parity: swapping two odd factors contributes a
    minus sign.
    """
    degree = sum(factor.head.rank for factor in term.factors)
    generators: list[SignedPermutation] = []
    offsets: list[int] = []
    offset = 0
    for factor in term.factors:
        offsets.append(offset)
        generators.extend(_factor_slot_generators(degree, offset, factor))
        offset += factor.head.rank

    generators.extend(_factor_exchange_generators(term, degree, offsets))
    return PermutationGroup(degree, generators)


def build_dummy_renaming_group(
    term: TensorTerm,
    *,
    dummy_policies: Mapping[IndexType, DummyRenamingPolicy] | None = None,
    allow_pair_flips: bool | None = None,
) -> PermutationGroup:
    """Build a generator group for dummy-index renamings.

    Dummy-pair exchanges are generated independently for every index family. A
    pair exchange swaps the up occurrence of one dummy with the up occurrence of
    another and likewise for down occurrences.  Optional family-specific pair
    flips are controlled by ``DummyRenamingPolicy``.  The optional ``allow_pair_flips`` argument supplies a default family policy when explicit dummy policies are not provided.
    """
    flat = _flat_indices(term)
    degree = len(flat)
    occurrences: dict[tuple[str, IndexType], list[tuple[int, AbstractIndex]]] = {}
    for pos, idx in enumerate(flat):
        occurrences.setdefault((idx.name, idx.index_type), []).append((pos, idx))

    pairs_by_type: dict[IndexType, list[tuple[str, int, int]]] = {}
    for (name, index_type), items in occurrences.items():
        if len(items) != 2:
            continue
        up_positions = [pos for pos, idx in items if idx.is_up]
        down_positions = [pos for pos, idx in items if idx.is_down]
        if len(up_positions) == 1 and len(down_positions) == 1:
            pairs_by_type.setdefault(index_type, []).append((name, up_positions[0], down_positions[0]))

    generators: list[SignedPermutation] = []
    policy_map = dict(dummy_policies or {})
    for index_type, pairs in pairs_by_type.items():
        pairs = sorted(pairs, key=lambda item: item[0])
        # Adjacent pair exchanges generate the full symmetric group on dummy
        # pair names within this family, without crossing other families.
        for left, right in zip(pairs, pairs[1:]):
            mapping = list(range(degree))
            _swap_positions(mapping, left[1], right[1])
            _swap_positions(mapping, left[2], right[2])
            generators.append(SignedPermutation(Permutation(mapping), 1))
        policy = policy_map.get(index_type)
        if policy is None and allow_pair_flips is not None:
            policy = DummyRenamingPolicy(bool(allow_pair_flips), 1)
        if policy and policy.allow_pair_flip:
            for _name, up_pos, down_pos in pairs:
                mapping = list(range(degree))
                _swap_positions(mapping, up_pos, down_pos)
                generators.append(SignedPermutation(Permutation(mapping), policy.pair_flip_sign))
    return PermutationGroup(degree, generators)


def decode_canonical_result(
    encoded: EncodedTensorMonomial,
    result: CanonicalDoubleCosetResult,
    *,
    canonicalize_structural: bool = False,
) -> DecodedCanonicalMonomial:
    """Decode a permutation-level canonical result back to a tensor term."""
    if result.zero:
        return DecodedCanonicalMonomial(TensorTerm.zero(), True, 0, result)
    coefficient = encoded.term.coefficient * Fraction(result.sign)
    labels = tuple(result.image)
    if len(labels) != encoded.degree:
        raise TensorKernelError("Canonical image length does not match encoded monomial degree.")
    index_types = {_index_type_key(idx_type): idx_type for idx_type in _index_types_in_term(encoded.term)}
    indices = tuple(_index_from_label(label, index_types) for label in labels)

    factors: list[TensorFactor] = []
    cursor = 0
    for original in encoded.term.factors:
        next_cursor = cursor + original.head.rank
        factors.append(TensorFactor(original.head, indices[cursor:next_cursor]))
        cursor = next_cursor
    decoded = TensorTerm(coefficient, tuple(factors)).rename_dummies()
    if canonicalize_structural:
        decoded = decoded.canonicalized()
    return DecodedCanonicalMonomial(decoded, False, result.sign, result)


def canonicalize_encoded_monomial(
    term: TensorTerm,
    *,
    backend: CanonicalizationBackend | None = None,
    dummy_policies: Mapping[IndexType, DummyRenamingPolicy] | None = None,
    canonicalize_structural: bool = False,
    use_common_case_optimizations: bool = True,
) -> DecodedCanonicalMonomial:
    """Canonicalize one tensor monomial using the permutation backend.

    Common-case checks conservatively short-circuit forced zeros and compute
    dummy-block/base-selection metadata for stabilizer/native backends.
    """
    encoded = encode_tensor_monomial(term, dummy_policies=dummy_policies)
    analysis = analyze_common_canonicalization_cases(encoded) if use_common_case_optimizations else None
    if analysis and analysis.zero:
        zero_result = CanonicalDoubleCosetResult(
            canonical=None,
            image=encoded.labels,
            sign=0,
            zero=True,
            candidates_considered=0,
        )
        return DecodedCanonicalMonomial(TensorTerm.zero(), True, 0, zero_result)
    use_backend = backend if backend is not None else default_permutation_backend()
    result = use_backend.canonicalize_double_coset(
        encoded.dummy_group,
        Permutation.identity(encoded.degree),
        encoded.slot_group,
        labels=encoded.labels,
        base=analysis.base if analysis else None,
    )
    return decode_canonical_result(encoded, result, canonicalize_structural=canonicalize_structural)


def canonicalize_tensor_term(
    term: TensorTerm,
    *,
    backend: CanonicalizationBackend | None = None,
    dummy_policies: Mapping[IndexType, DummyRenamingPolicy] | None = None,
) -> TensorTerm:
    """Public tensor-term canonicalization through the permutation backend.

    Safe common-case normalization is applied first so obvious total-symmetry
    and repeated-factor cases do not reach the general double-coset search.
    """
    normalized = canonicalize_total_symmetry_blocks(term)
    normalized = canonicalize_repeated_factors(normalized)
    return canonicalize_encoded_monomial(normalized, backend=backend, dummy_policies=dummy_policies).term


def canonicalize_tensor_expression(
    expr: TensorExpr,
    *,
    backend: CanonicalizationBackend | None = None,
    dummy_policies: Mapping[IndexType, DummyRenamingPolicy] | None = None,
) -> TensorExpr:
    """Canonicalize every monomial in an expression through the backend."""
    terms = [canonicalize_tensor_term(term, backend=backend, dummy_policies=dummy_policies) for term in expr.terms]
    return TensorExpr(tuple(term for term in terms if term.coefficient)).canonicalized()


def compare_encoded_canonicalization_to_oracle(term: TensorTerm) -> TensorCanonicalizationOracleResult:
    """Compare backend canonicalization with brute-force double-coset enumeration."""
    encoded = encode_tensor_monomial(term)
    backend_result = default_permutation_backend().canonicalize_double_coset(
        encoded.dummy_group,
        Permutation.identity(encoded.degree),
        encoded.slot_group,
        labels=encoded.labels,
    )
    oracle_result = brute_force_double_coset(
        encoded.dummy_group,
        Permutation.identity(encoded.degree),
        encoded.slot_group,
        labels=encoded.labels,
    )
    return TensorCanonicalizationOracleResult(encoded, backend_result, oracle_result)



def validate_tensor_monomial_indices(term: TensorTerm) -> None:
    """Reject malformed Einstein-index use before permutation encoding.

    Ordinary monomial encoding permits at most two occurrences of one index name
    in one index family.  Two occurrences are a dummy pair only when their
    variances are opposite.  Repeated same-variance labels inside an
    antisymmetric factor are left to the zero detector so antisymmetry
    simplifications can return zero cleanly.
    """
    occurrences: dict[tuple[tuple, str], list[tuple[int, int, AbstractIndex, TensorFactor]]] = {}
    for factor_pos, factor in enumerate(term.factors):
        for slot_pos, idx in enumerate(factor.indices):
            occurrences.setdefault((_index_type_key(idx.index_type), idx.name), []).append((factor_pos, slot_pos, idx, factor))
    for (_family_key, name), items in occurrences.items():
        if len(items) > 2:
            raise TensorKernelError(f"Index {name!r} occurs more than twice in one monomial.")
        if len(items) == 2 and items[0][2].variance == items[1][2].variance:
            if _same_factor_negative_symmetry_zero(items) or _repeated_odd_factor_zero(items):
                continue
            raise TensorKernelError(
                f"Index {name!r} occurs twice with the same variance; "
                "dummy contractions require one up and one down occurrence."
            )


def _same_factor_negative_symmetry_zero(items: Sequence[tuple[int, int, AbstractIndex, TensorFactor]]) -> bool:
    left, right = items
    if left[0] != right[0]:
        return False
    factor = left[3]
    kind = factor.head.symmetry.kind
    if kind == "antisymmetric":
        return True
    if kind == "antisym_last2" and {left[1], right[1]} == {factor.head.rank - 2, factor.head.rank - 1}:
        return True
    if kind in {"riemann", "weyl"} and ({left[1], right[1]} == {0, 1} or {left[1], right[1]} == {2, 3}):
        return True
    labels = tuple(_factor_label_tuples(factor))
    for mapping, sign in factor.head.symmetry.signed_generators:
        if sign != -1 or len(mapping) != factor.head.rank:
            continue
        image = tuple(labels[mapping[source]] for source in range(factor.head.rank))
        if image == labels:
            return True
    return False


def _repeated_odd_factor_zero(items: Sequence[tuple[int, int, AbstractIndex, TensorFactor]]) -> bool:
    left, right = items
    return left[0] != right[0] and left[3].head.parity and right[3].head.parity and left[3].key() == right[3].key()


def analyze_common_canonicalization_cases(encoded: EncodedTensorMonomial) -> CommonCaseAnalysis:
    """Analyze optimized common cases before general double-coset search."""
    zero_reason = _early_zero_reason(encoded.term)
    dummy_blocks = decompose_dummy_pair_blocks(encoded.term)
    repeated_blocks = repeated_factor_blocks(encoded.term)
    base = select_canonicalization_base(
        encoded,
        dummy_blocks=dummy_blocks,
        repeated_factor_blocks=repeated_blocks,
    )
    return CommonCaseAnalysis(zero_reason is not None, zero_reason, dummy_blocks, repeated_blocks, base)


def decompose_dummy_pair_blocks(term: TensorTerm) -> tuple[DummyPairBlock, ...]:
    """Return up/down dummy-pair positions grouped by index family."""
    flat = _flat_indices(term)
    occurrences: dict[tuple[str, IndexType], list[tuple[int, AbstractIndex]]] = {}
    for pos, idx in enumerate(flat):
        occurrences.setdefault((idx.name, idx.index_type), []).append((pos, idx))
    blocks: list[DummyPairBlock] = []
    for (name, index_type), items in occurrences.items():
        if len(items) != 2:
            continue
        up = [pos for pos, idx in items if idx.is_up]
        down = [pos for pos, idx in items if idx.is_down]
        if len(up) == 1 and len(down) == 1:
            blocks.append(DummyPairBlock(index_type, name, up[0], down[0]))
    return tuple(sorted(blocks, key=lambda block: (_index_type_key(block.index_type), min(block.up_position, block.down_position))))


def repeated_factor_blocks(term: TensorTerm) -> tuple[tuple[int, ...], ...]:
    """Return blocks of exchange-equivalent repeated commutative factors."""
    groups: dict[tuple, list[int]] = {}
    for pos, factor in enumerate(term.factors):
        if factor.head.commutative and factor.head.rank:
            groups.setdefault(_exchange_key(factor), []).append(pos)
    return tuple(tuple(items) for items in groups.values() if len(items) > 1)


def select_canonicalization_base(
    encoded: EncodedTensorMonomial,
    *,
    dummy_blocks: Sequence[DummyPairBlock] | None = None,
    repeated_factor_blocks: Sequence[Sequence[int]] | None = None,
) -> tuple[int, ...]:
    """Choose a base order for stabilizer-chain/native canonicalizers."""
    dummy_positions = {pos for block in (dummy_blocks or ()) for pos in (block.up_position, block.down_position)}
    free_positions = [pos for pos, label in enumerate(encoded.labels) if label[2] == _FREE_LABEL]
    pair_reps = [min(block.up_position, block.down_position) for block in (dummy_blocks or ())]
    repeated_slots: set[int] = set()
    if repeated_factor_blocks:
        offsets = _factor_offsets(encoded.term)
        for block in repeated_factor_blocks:
            for factor_pos in block:
                factor = encoded.term.factors[factor_pos]
                repeated_slots.update(range(offsets[factor_pos], offsets[factor_pos] + factor.head.rank))
    preferred: list[int] = []
    for group in (free_positions, pair_reps, sorted(repeated_slots)):
        for pos in group:
            if pos not in preferred:
                preferred.append(pos)
    remaining = [pos for pos in range(encoded.degree) if pos not in preferred]
    orbit_cache: dict[int, int] = {}

    def orbit_size(pos: int) -> int:
        if not encoded.degree:
            return 1
        if pos not in orbit_cache:
            orbit_cache[pos] = len(encoded.slot_group.orbit(pos))
        return orbit_cache[pos]

    remaining.sort(key=lambda pos: (orbit_size(pos), pos in dummy_positions, pos))
    return tuple(preferred + remaining)


def canonicalize_total_symmetry_blocks(term: TensorTerm) -> TensorTerm:
    """Apply safe local sorting for total symmetric/antisymmetric factors."""
    coeff = term.coefficient
    factors: list[TensorFactor] = []
    for factor in term.factors:
        kind = factor.head.symmetry.kind
        if kind not in {"symmetric", "antisymmetric"} or factor.head.rank < 2:
            factors.append(factor)
            continue
        sign, canonical = factor.canonicalized()
        if sign == 0:
            return TensorTerm.zero()
        coeff *= sign
        factors.append(canonical)
    return TensorTerm(coeff, tuple(factors))


def canonicalize_repeated_factors(term: TensorTerm) -> TensorTerm:
    """Canonicalize repeated commutative factor blocks without group expansion."""
    if not term.factors or term.coefficient == 0:
        return term
    seen_odd: set[tuple] = set()
    for factor in term.factors:
        if factor.head.commutative and factor.head.parity:
            key = factor.key()
            if key in seen_odd:
                return TensorTerm.zero()
            seen_odd.add(key)
    commutative = [factor for factor in term.factors if factor.head.commutative]
    if not commutative:
        return term
    order = sorted(range(len(commutative)), key=lambda idx: commutative[idx].key())
    rank = {old_pos: new_pos for new_pos, old_pos in enumerate(order)}
    sign = 1
    for left in range(len(commutative)):
        for right in range(left + 1, len(commutative)):
            if rank[left] > rank[right] and commutative[left].head.parity and commutative[right].head.parity:
                sign *= -1
    sorted_commutative = [commutative[idx] for idx in order]
    out: list[TensorFactor] = []
    comm_iter = iter(sorted_commutative)
    for factor in term.factors:
        if factor.head.commutative:
            out.append(next(comm_iter))
        else:
            out.append(factor)
    return TensorTerm(term.coefficient * sign, tuple(out))


def _early_zero_reason(term: TensorTerm) -> str | None:
    if term.coefficient == 0:
        return "zero coefficient"
    for factor in term.factors:
        labels = _factor_label_tuples(factor)
        kind = factor.head.symmetry.kind
        rank = factor.head.rank
        if kind == "antisymmetric" and len(set(labels)) < len(labels):
            return f"repeated label in antisymmetric tensor {factor.head.name}"
        if kind == "antisym_last2" and rank >= 2 and labels[-2] == labels[-1]:
            return f"repeated label in antisymmetric slot pair of {factor.head.name}"
        if kind in {"riemann", "weyl"} and rank == 4 and (labels[0] == labels[1] or labels[2] == labels[3]):
            return f"repeated label in antisymmetric curvature pair of {factor.head.name}"
        for mapping, sign in factor.head.symmetry.signed_generators:
            if sign != -1 or len(mapping) != rank:
                continue
            image = tuple(labels[mapping[source]] for source in range(rank))
            if image == labels:
                return f"negative slot symmetry fixes tensor {factor.head.name}"
    contraction_reason = _symmetric_antisymmetric_contraction_zero(term)
    if contraction_reason is not None:
        return contraction_reason
    odd_factor_keys: set[tuple] = set()
    for factor in term.factors:
        if factor.head.commutative and factor.head.parity:
            key = factor.key()
            if key in odd_factor_keys:
                return f"identical odd factor {factor.head.name} appears twice"
            odd_factor_keys.add(key)
    return None


def _symmetric_antisymmetric_contraction_zero(term: TensorTerm) -> str | None:
    """Detect the common contraction of symmetric and antisymmetric rank-2 factors."""
    factors = tuple(term.factors)
    for left_pos, left in enumerate(factors):
        if left.head.rank != 2:
            continue
        for right in factors[left_pos + 1:]:
            if right.head.rank != 2:
                continue
            kinds = {left.head.symmetry.kind, right.head.symmetry.kind}
            if kinds != {"symmetric", "antisymmetric"}:
                continue
            if tuple(idx.index_type for idx in left.indices) != tuple(idx.index_type for idx in right.indices):
                continue
            left_pairs = {(idx.name, idx.index_type, idx.variance) for idx in left.indices}
            right_opposite = {(idx.name, idx.index_type, "down" if idx.is_up else "up") for idx in right.indices}
            if left_pairs == right_opposite:
                return "symmetric tensor contracted with antisymmetric tensor"
    return None


def _factor_label_tuples(factor: TensorFactor) -> tuple[tuple, ...]:
    return tuple((idx.name, _index_type_key(idx.index_type), idx.variance) for idx in factor.indices)


def _factor_offsets(term: TensorTerm) -> tuple[int, ...]:
    offsets: list[int] = []
    cursor = 0
    for factor in term.factors:
        offsets.append(cursor)
        cursor += factor.head.rank
    return tuple(offsets)

def _factor_slot_generators(degree: int, offset: int, factor: TensorFactor) -> list[SignedPermutation]:
    rank = factor.head.rank
    symmetry = factor.head.symmetry
    kind = symmetry.kind
    generators: list[SignedPermutation] = []
    for local_mapping, sign in symmetry.signed_generators:
        if len(local_mapping) != rank:
            raise TensorKernelError("Tensor-head slot generator rank does not match the tensor rank.")
        mapping = list(range(degree))
        for source, target in enumerate(local_mapping):
            mapping[offset + source] = offset + target
        generators.append(SignedPermutation(Permutation(mapping), sign))
    if rank < 2 or kind == "none" or kind == "custom":
        return generators
    if kind == "symmetric":
        sign = 1
        slots = range(rank - 1)
    elif kind == "antisymmetric":
        sign = -1
        slots = range(rank - 1)
    elif kind == "antisym_last2":
        generators.append(_signed_transposition(degree, offset + rank - 2, offset + rank - 1, -1))
        return generators
    elif kind in {"riemann", "weyl"} and rank == 4:
        generators.extend(
            [
                _signed_transposition(degree, offset + 0, offset + 1, -1),
                _signed_transposition(degree, offset + 2, offset + 3, -1),
                _signed_pair_exchange(degree, offset, (0, 1), (2, 3), 1),
            ]
        )
        return generators
    else:
        return generators
    for local in slots:
        generators.append(_signed_transposition(degree, offset + local, offset + local + 1, sign))
    return generators


def _factor_exchange_generators(term: TensorTerm, degree: int, offsets: Sequence[int]) -> list[SignedPermutation]:
    groups: dict[tuple, list[int]] = {}
    for pos, factor in enumerate(term.factors):
        if not factor.head.commutative or factor.head.rank == 0:
            continue
        key = _exchange_key(factor)
        groups.setdefault(key, []).append(pos)
    generators: list[SignedPermutation] = []
    for positions in groups.values():
        if len(positions) < 2:
            continue
        # Adjacent transpositions among the positions of equivalent factors
        # generate the full repeated-factor exchange group even when equivalent
        # factors are separated by other commutative factors in the product.
        for left_pos, right_pos in zip(positions, positions[1:]):
            left = term.factors[left_pos]
            rank = left.head.rank
            mapping = list(range(degree))
            left_offset = offsets[left_pos]
            right_offset = offsets[right_pos]
            for slot in range(rank):
                mapping[left_offset + slot] = right_offset + slot
                mapping[right_offset + slot] = left_offset + slot
            sign = -1 if (left.head.parity and term.factors[right_pos].head.parity) else 1
            generators.append(SignedPermutation(Permutation(mapping), sign))
    return generators


def _exchange_key(factor: TensorFactor) -> tuple:
    return (
        factor.head,
        tuple(idx.variance for idx in factor.indices),
    )


def _signed_transposition(degree: int, first: int, second: int, sign: int) -> SignedPermutation:
    return SignedPermutation(Permutation.transposition(degree, first, second), sign)


def _signed_pair_exchange(
    degree: int,
    offset: int,
    left_pair: tuple[int, int],
    right_pair: tuple[int, int],
    sign: int,
) -> SignedPermutation:
    mapping = list(range(degree))
    for left, right in zip(left_pair, right_pair):
        _swap_positions(mapping, offset + left, offset + right)
    return SignedPermutation(Permutation(mapping), sign)


def _swap_positions(mapping: list[int], first: int, second: int) -> None:
    mapping[first], mapping[second] = mapping[second], mapping[first]


def _slot_metadata(term: TensorTerm) -> tuple[SlotMetadata, ...]:
    out: list[SlotMetadata] = []
    for factor_position, factor in enumerate(term.factors):
        for slot_position, idx in enumerate(factor.indices):
            out.append(SlotMetadata(factor_position, slot_position, idx.index_type, idx.variance))
    return tuple(out)


def _flat_indices(term: TensorTerm) -> tuple[AbstractIndex, ...]:
    return tuple(idx for factor in term.factors for idx in factor.indices)


def _index_types_in_term(term: TensorTerm) -> tuple[IndexType, ...]:
    seen: dict[tuple, IndexType] = {}
    for idx in _flat_indices(term):
        seen[_index_type_key(idx.index_type)] = idx.index_type
    return tuple(seen.values())


def _labels_for_term(term: TensorTerm) -> tuple[SlotLabel, ...]:
    flat = _flat_indices(term)
    occurrences: dict[tuple[str, IndexType], list[AbstractIndex]] = {}
    for idx in flat:
        occurrences.setdefault((idx.name, idx.index_type), []).append(idx)
    dummy_names: set[tuple[str, IndexType]] = set()
    for key, items in occurrences.items():
        if len(items) == 2 and {item.variance for item in items} == {"up", "down"}:
            dummy_names.add(key)
    dummy_name_map: dict[tuple[str, IndexType], str] = {}
    counters: dict[IndexType, int] = {}
    for idx in flat:
        key = (idx.name, idx.index_type)
        if key in dummy_names and key not in dummy_name_map:
            counters[idx.index_type] = counters.get(idx.index_type, 0) + 1
            dummy_name_map[key] = f"d{counters[idx.index_type]}"

    labels: list[SlotLabel] = []
    for idx in flat:
        key = (idx.name, idx.index_type)
        label_kind = _DUMMY_LABEL if key in dummy_names else _FREE_LABEL
        name = dummy_name_map.get(key, idx.name)
        labels.append((_type_name(idx.index_type), _stable_index_type_id(idx.index_type), label_kind, name, _variance_code(idx.variance)))
    return tuple(labels)


def _index_type_key(index_type: IndexType) -> tuple:
    manifold = index_type.manifold
    bundle = index_type.bundle
    bundle_key = None if bundle is None else (bundle.name, bundle.rank)
    return (
        getattr(manifold, "name", repr(manifold)),
        getattr(manifold, "dimension", None),
        index_type.name,
        index_type.dimension,
        index_type.metric,
        bundle_key,
    )


def _stable_index_type_id(index_type: IndexType) -> tuple:
    return _index_type_key(index_type)


def _type_name(index_type: IndexType) -> str:
    return index_type.name


def _variance_code(variance: str) -> int:
    if variance == "up":
        return _UP
    if variance == "down":
        return _DOWN
    raise TensorKernelError(f"Unsupported variance in tensor encoding: {variance!r}.")


def _variance_from_code(code: int) -> str:
    if code == _UP:
        return "up"
    if code == _DOWN:
        return "down"
    raise TensorKernelError(f"Unsupported variance code in tensor encoding: {code!r}.")


def _index_from_label(label: Sequence, index_types: dict[tuple, IndexType]) -> AbstractIndex:
    if len(label) != 5:
        raise TensorKernelError(f"Invalid slot label: {label!r}.")
    type_name, type_id, _label_kind, name, variance_code = label
    key = tuple(type_id) if isinstance(type_id, tuple) else (str(type_name), type_id)
    index_type = index_types.get(key)
    if index_type is None:
        raise TensorKernelError(f"Cannot decode index type from slot label: {label!r}.")
    return AbstractIndex(str(name), index_type, _variance_from_code(int(variance_code)))


def canonicalize_tensor(
    expr: TensorExpr | TensorTerm,
    *,
    backend: CanonicalizationBackend | None = None,
    dummy_policies: Mapping[IndexType, DummyRenamingPolicy] | None = None,
    explain: bool = False,
) -> TensorExpr | TensorCanonicalizationResult:
    """Canonicalize a semantic tensor term/expression through the public backend path.

    The current default is the guarded pure-Python reference backend.  The
    return value is normally a ``TensorExpr``; with ``explain=True`` a result
    object includes backend and warning metadata.
    """
    use_backend = backend if backend is not None else default_permutation_backend()
    backend_name = type(use_backend).__name__
    input_expr = TensorExpr((expr,)) if isinstance(expr, TensorTerm) else expr
    output = canonicalize_tensor_expression(input_expr, backend=use_backend, dummy_policies=dummy_policies)
    warnings: list[str] = []
    if backend is None and backend_name == "PythonPermutationBackend":
        warnings.append("Using guarded pure-Python reference canonicalization backend; install/select a native backend for large expressions.")
    if not explain:
        return output
    return TensorCanonicalizationResult(output, backend_name, tuple(warnings), ("strict index validation",))
