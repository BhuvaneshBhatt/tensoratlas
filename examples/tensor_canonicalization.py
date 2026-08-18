"""Inspect a canonical symbolic tensor expression key.

Run from the repository root with:

    python examples/tensor_canonicalization.py
"""

from tensoratlas.examples import canonicalization_workflow

if __name__ == "__main__":
    result = canonicalization_workflow()
    print("Canonical expression:", result["canonical"])
    print("Canonical key:", result["canonical_key"])
