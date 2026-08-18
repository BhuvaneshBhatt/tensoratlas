from __future__ import annotations

from tensoratlas.notebook_examples import run_regression_benchmarks


def main() -> None:
    for result in run_regression_benchmarks(repeat=1):
        print(f"{result.name}: {result.seconds:.6f}s key={result.canonical_key!r}")


if __name__ == "__main__":
    main()
