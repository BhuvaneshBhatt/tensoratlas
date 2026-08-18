"""Run notebook-grade geometry examples.

Run from the repository root with:

    python examples/notebook_grade_geometry_examples.py
"""

from tensoratlas.notebook_examples import all_notebook_examples, run_regression_benchmarks


def main() -> None:
    for example in all_notebook_examples():
        print(f"# {example.name}")
        print("dimension:", example.dimension)
        print("known components:", dict(example.known_components))
        print("zero reductions:", {k: v.kind for k, v in example.zero_reductions.items()})
        print("canonical forms:", {k: v.kind for k, v in example.canonical_forms.items()})
        print()
    print("Benchmarks")
    for result in run_regression_benchmarks():
        print(result)


if __name__ == "__main__":
    main()
