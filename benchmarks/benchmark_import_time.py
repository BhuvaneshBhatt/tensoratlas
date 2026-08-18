"""Measure import-time behavior for the package root.

Run with:
    python benchmarks/benchmark_import_time.py
"""
from __future__ import annotations

import subprocess
import sys
import textwrap


def main() -> None:
    code = textwrap.dedent(
        """
        import sys, time
        start = time.perf_counter()
        import tensoratlas
        elapsed = time.perf_counter() - start
        print(f"import tensoratlas: {elapsed:.6f}s")
        print(f"matplotlib_loaded={ 'matplotlib' in sys.modules }")
        print(f"sympy_loaded={ 'sympy' in sys.modules }")
        sys.stdout.flush()
        import os; os._exit(0)
        """
    )
    subprocess.run([sys.executable, "-S", "-c", code], check=True)


if __name__ == "__main__":
    main()
