"""Demonstrate a rotor in orthogonal-metric geometric algebra.

Run from the repository root with:

    python examples/geometric_algebra_rotor.py
"""

from tensoratlas.examples import geometric_algebra_workflow

if __name__ == "__main__":
    result = geometric_algebra_workflow()
    print("e1^2:", result["e1_squared"])
    print("bivector:", result["bivector"])
    print("rotor:", result["rotor"])
