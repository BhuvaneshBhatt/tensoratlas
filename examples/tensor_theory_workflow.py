"""Run the TensorAtlas tensor-theory tutorial examples.

Run from the repository root with:

    python examples/tensor_theory_workflow.py
"""

from pprint import pprint

from tensoratlas.examples.tensor_theory import tensor_theory_workflow

if __name__ == "__main__":
    pprint(tensor_theory_workflow())
