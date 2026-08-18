"""Pytest collection markers for optional or slower public-release checks."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        nodeid = item.nodeid.lower()
        if "notebook" in nodeid:
            item.add_marker(pytest.mark.notebook)
        if "visualization" in nodeid or "plot" in nodeid:
            item.add_marker(pytest.mark.plot)
        if any(token in nodeid for token in ("benchmark", "performance", "perf", "schwarzschild", "flrw")):
            item.add_marker(pytest.mark.slow)
