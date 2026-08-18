"""Check selected Schwarzschild vacuum Ricci components.

Run from the repository root with:

    python examples/schwarzschild_vacuum.py
"""

from tensoratlas.examples import schwarzschild_workflow

if __name__ == "__main__":
    result = schwarzschild_workflow()
    print("Ricci_tt:", result["ricci_tt"])
    print("Ricci_rr:", result["ricci_rr"])
    print("Sample Christoffel:", result["sample_christoffel"])
