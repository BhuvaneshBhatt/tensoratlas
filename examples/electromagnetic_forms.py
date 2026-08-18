"""Build formal electromagnetic form expressions.

Run from the repository root with:

    python examples/electromagnetic_forms.py
"""

from tensoratlas.examples import electromagnetic_workflow

if __name__ == "__main__":
    result = electromagnetic_workflow()
    print("Potential:", result["potential"])
    print("Field strength:", result["field_strength"])
    print("Bianchi expression:", result["bianchi"])
