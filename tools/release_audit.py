"""Release-audit checks for the TensorAtlas source tree.

The checks are intentionally lightweight and deterministic.  They catch common
problems that make a public source release look unpolished: generated artifacts,
duplicate top-level definitions, stale iterative wording, missing package data,
and broken public example scripts.
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tensoratlas"
PUBLIC_DIRS = [ROOT / "docs", ROOT / "examples", SRC]
ROOT_TEXT_FILES = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "CHANGELOG.md", ROOT / "pyproject.toml"]
GENERATED_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "__MACOSX__",
    "__MACOSX",
    ".DS_Store",
}
GENERATED_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp", ".bak", ".orig"}
FORBIDDEN_TEXT = {
    "milestone",
    "phase 1",
    "phase 2",
    "phase 3",
    "phase-1",
    "phase-2",
    "phase-3",
    "TransformedField",
}
ALLOW_TEXT = {
    "docs/publishing_checklist.md:milestone",
}
EXAMPLE_SCRIPTS = [
    "examples/five_minute_tour.py",
    "examples/two_sphere_curvature.py",
    "examples/electromagnetic_forms.py",
    "examples/geometric_algebra_rotor.py",
]


def _iter_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in root.rglob("*")]


def check_generated_artifacts(errors: list[str]) -> None:
    for path in _iter_paths(ROOT):
        rel = path.relative_to(ROOT)
        if any(part in GENERATED_NAMES for part in rel.parts) or path.suffix in GENERATED_SUFFIXES:
            errors.append(f"generated artifact should not be committed: {rel}")


def check_duplicate_defs(errors: list[str]) -> None:
    if not SRC.exists():
        return
    for path in SRC.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"syntax error in {path.relative_to(ROOT)}: {exc}")
            continue
        seen: dict[tuple[str, str], int] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                key = (type(node).__name__, node.name)
                if key in seen:
                    errors.append(
                        "duplicate top-level definition "
                        f"{node.name!r} in {path.relative_to(ROOT)} at lines {seen[key]} and {node.lineno}"
                    )
                else:
                    seen[key] = node.lineno


def check_text(errors: list[str]) -> None:
    candidates: list[Path] = []
    for root in PUBLIC_DIRS:
        candidates.extend(path for path in _iter_paths(root) if path.suffix in {".py", ".md", ".toml", ".yml", ".yaml"})
    candidates.extend(path for path in ROOT_TEXT_FILES if path.exists())

    for path in sorted(set(candidates)):
        rel = path.relative_to(ROOT)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for token in FORBIDDEN_TEXT:
            marker = f"{rel}:{token}"
            if token in text and marker not in ALLOW_TEXT:
                errors.append(f"release text artifact {token!r} found in {rel}")


def check_required_files(errors: list[str]) -> None:
    required = [
        ROOT / ".gitignore",
        ROOT / "tools" / "release_audit.py",
        SRC / "py.typed",
        ROOT / "examples" / "five_minute_tour.py",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"required release file missing: {path.relative_to(ROOT)}")


def check_example_scripts(errors: list[str]) -> None:
    for script in EXAMPLE_SCRIPTS:
        path = ROOT / script
        if not path.exists():
            errors.append(f"public example missing: {script}")
            continue
        old_path = list(sys.path)
        sys.path.insert(0, str(ROOT / "src"))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(path), run_name="__main__")
        except Exception as exc:  # pragma: no cover - exercised by release checks.
            errors.append(f"public example failed: {script}: {exc!r}")
        finally:
            sys.path[:] = old_path


def main() -> int:
    errors: list[str] = []
    check_required_files(errors)
    check_generated_artifacts(errors)
    check_duplicate_defs(errors)
    check_text(errors)
    if os.getenv("TENSORATLAS_AUDIT_RUN_EXAMPLES") == "1":
        check_example_scripts(errors)
    if errors:
        print("Release audit failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Release audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
