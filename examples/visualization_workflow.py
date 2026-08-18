"""Run the TensorAtlas visualization examples.

Run from the repository root with:

    python examples/visualization_workflow.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensoratlas.examples.visualizations import _VISUALIZATION_FUNCTIONS


def main() -> None:
    for make_figure in _VISUALIZATION_FUNCTIONS:
        fig = make_figure()
        print(f"created {make_figure.__name__}: {type(fig).__name__}")
        plt.close(fig)


if __name__ == "__main__":
    main()
