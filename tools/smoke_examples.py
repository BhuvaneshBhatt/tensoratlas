"""Run a compact smoke suite of public TensorAtlas examples.

Run from the repository root with:

    python tools/smoke_examples.py
"""

from __future__ import annotations

import contextlib
import io
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [
    "examples/five_minute_tour.py",
    "examples/two_sphere_curvature.py",
    "examples/electromagnetic_forms.py",
    "examples/geometric_algebra_rotor.py",
]


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    for example in EXAMPLES:
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT / example), run_name="__main__")
        print(f"ok: {example}", flush=True)
    return 0


if __name__ == "__main__":
    os._exit(main())
