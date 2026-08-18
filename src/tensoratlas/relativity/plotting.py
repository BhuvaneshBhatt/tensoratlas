"""Small plotting helpers for relativity examples."""

from __future__ import annotations

from typing import Iterable


def geodesic_plot_2d(points: Iterable[tuple[float, float]], *, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    pts = list(points)
    ax.plot([point[0] for point in pts], [point[1] for point in pts])
    ax.set_aspect("equal", adjustable="box")
    return ax
