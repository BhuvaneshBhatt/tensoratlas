from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensoratlas.geometric_algebra import GeometricAlgebra
from tensoratlas.relativity import christoffel_component, two_sphere_metric
from tensoratlas.validation import ValidationReport, check_indices


def test_validation_report_helpers_on_geometric_algebra():
    ga = GeometricAlgebra.euclidean(3)
    report = ga.validation_report()
    assert isinstance(report, ValidationReport)
    assert report.ok
    assert report.warnings


def test_shared_index_checker_reports_context():
    with pytest.raises(IndexError, match="example component"):
        check_indices("example component", 2, 0, 2)


def test_selected_component_uses_shared_index_checker():
    metric = two_sphere_metric()
    with pytest.raises(IndexError, match="Christoffel component"):
        christoffel_component(metric, 0, 0, 99)


def test_package_root_import_does_not_load_matplotlib():
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo / "src")
    code = "import sys; import tensoratlas; print('matplotlib' in sys.modules)"
    result = subprocess.run([sys.executable, "-S", "-c", code], env=env, text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "False"


def test_benchmark_scripts_exist():
    repo = Path(__file__).resolve().parents[1]
    for name in [
        "benchmark_import_time.py",
        "benchmark_relativity.py",
        "benchmark_geometric_algebra.py",
        "benchmark_tensor_canonicalization.py",
    ]:
        assert (repo / "benchmarks" / name).exists()
