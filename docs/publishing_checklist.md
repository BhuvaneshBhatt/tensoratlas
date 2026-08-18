# First-publication checklist

Before the first public publication, complete these checks from a clean checkout.

## Metadata

- Confirm `pyproject.toml` uses the package name, version, author, GPL-3.0-only license, classifiers, and package-data settings intended for publication.
- Add the repository URL and issue tracker URL once the public repository exists. Do not publish placeholder URLs.
- Confirm `LICENSE`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, and `CODE_OF_CONDUCT.md` are present.

## Tests and audits

```bash
python tools/release_audit.py
python -m pytest
python -m pytest --nbmake notebooks/tensoratlas_demo.ipynb  # if nbmake is installed
python -m ruff check src tests examples tools
python -m build
python -m twine check dist/*
```

## Documentation

- Execute the tutorial notebook from a fresh kernel.
- Verify that all plain scripts in `examples/` run.
- Verify that public examples avoid private implementation modules.
- Verify that all convention-sensitive formulas state their conventions.

## Publication

- Tag the release after tests, notebook execution, and package checks pass.
- Upload artifacts only from a clean tree.
- After publication, install the wheel in a fresh virtual environment and run a small smoke test.
