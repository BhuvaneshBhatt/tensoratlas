"""Compute selected FLRW Einstein-tensor components.

Run from the repository root with:

    python examples/flrw_einstein_tensor.py
"""

from tensoratlas.examples import flrw_workflow

if __name__ == "__main__":
    result = flrw_workflow()
    print("G_tt:", result["einstein_tt"])
    print("G_rr:", result["einstein_rr"])
