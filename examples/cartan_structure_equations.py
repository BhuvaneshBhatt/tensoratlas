"""Evaluate formal Cartan structure-equation workflows.

Run from the repository root with:

    python examples/cartan_structure_equations.py
"""

from tensoratlas.examples import cartan_structure_workflow

if __name__ == "__main__":
    result = cartan_structure_workflow()
    print("Torsion:", result["torsion"])
    print("Curvature:", result["curvature"])
