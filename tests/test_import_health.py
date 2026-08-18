from __future__ import annotations

import importlib
import json
import pkgutil
import subprocess
import sys
import time

import pytest


_PUBLIC_MODULES = tuple(
    sorted(module.name for module in pkgutil.iter_modules(["src/tensoratlas"]) if not module.ispkg)
)


def _run_python(script: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def test_package_root_import_is_lazy_and_cheap() -> None:
    script = r'''
import json, sys, time
start = time.perf_counter()
import tensoratlas
elapsed = time.perf_counter() - start
print(json.dumps({
    "elapsed": elapsed,
    "loaded_heavy": sorted(
        name for name in sys.modules
        if name in {
            "tensoratlas.abstract_tensor",
            "tensoratlas.curvature_relations",
            "tensoratlas.curvature_normal_forms",
            "tensoratlas.variational_gr",
        }
    ),
}))
'''
    data = json.loads(_run_python(script))
    assert data["elapsed"] < 0.5
    assert data["loaded_heavy"] == []


def test_lazy_public_attribute_imports_only_owning_module() -> None:
    script = r'''
import json, sys
import tensoratlas
name = tensoratlas.TensorExpr.__name__
print(json.dumps({
    "name": name,
    "semantic_ir_loaded": "tensoratlas.semantic_ir" in sys.modules,
    "curvature_normal_forms_loaded": "tensoratlas.curvature_normal_forms" in sys.modules,
}))
'''
    data = json.loads(_run_python(script))
    assert data["name"] in {"TensorExpr", "TensorExpr"}
    assert data["semantic_ir_loaded"] is True
    assert data["curvature_normal_forms_loaded"] is False


@pytest.mark.parametrize("module_name", _PUBLIC_MODULES)
def test_public_module_import_smoke(module_name: str) -> None:
    importlib.import_module(f"tensoratlas.{module_name}")


def test_public_module_imports_are_individually_cheap() -> None:
    slow: list[tuple[str, float]] = []
    for module_name in _PUBLIC_MODULES:
        start = time.perf_counter()
        importlib.import_module(f"tensoratlas.{module_name}")
        elapsed = time.perf_counter() - start
        if elapsed > 1.5:
            slow.append((module_name, elapsed))
    assert not slow
