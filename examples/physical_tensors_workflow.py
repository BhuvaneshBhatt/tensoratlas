"""Run the TensorAtlas physical tensor tutorial examples.

Run from the repository root with:

    python examples/physical_tensors_workflow.py
"""

from pprint import pprint

from tensoratlas.examples.physical_tensors import physical_tensor_workflow

if __name__ == "__main__":
    pprint(physical_tensor_workflow())
