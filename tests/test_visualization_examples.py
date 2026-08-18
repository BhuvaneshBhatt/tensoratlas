from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tensoratlas.examples.visualizations import (
    _VISUALIZATION_FUNCTIONS,
    visualization_workflow,
)


def test_visualization_workflow_lists_all_public_visualizations():
    names = visualization_workflow()
    assert len(names) == 18
    assert "plot_basis_change" in names
    assert "plot_canonicalization_tree" in names


def test_all_visualization_examples_return_figures():
    for make_figure in _VISUALIZATION_FUNCTIONS:
        fig = make_figure()
        try:
            assert isinstance(fig, Figure), make_figure.__name__
            assert fig.axes, make_figure.__name__
        finally:
            plt.close(fig)
