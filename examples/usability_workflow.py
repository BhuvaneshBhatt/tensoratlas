"""Run the TensorAtlas usability workflow examples.

Run from the repository root with:

    python examples/usability_workflow.py
"""

from pprint import pprint

from tensoratlas.examples.usability import usability_workflow_examples


if __name__ == "__main__":
    pprint(usability_workflow_examples())
