# Canonicalization foundation

This document records the canonicalization consolidation goals for TensorAtlas.

## Scope

The canonicalization foundation is the algebraic foundation layer:

- one shared permutation and symmetry backend
- one shared canonicalization kernel for abstract, indexed, and component expressions
- structural semantic keys instead of print/repr/srepr identity
- canonicalization caches and performance guardrails

## Current implementation direction

The repository now prefers structural semantic keys through `tensoratlas.canonical_keys` and exposes Normal-form and bridge contracts through `tensoratlas.contracts`.

## Success criteria

A canonicalization path should be:

- idempotent
- invariant under supported slot symmetries and dummy renaming
- stable under supported abstract/indexed bridge round-trips
- independent of textual rendering details
