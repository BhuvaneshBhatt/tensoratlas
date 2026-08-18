# Contributing

TensorAtlas is authored by Bhuvanesh Bhatt and distributed under GPL-3.0-only. Contributions should be compatible with that license.

Please keep public APIs convention-heavy, documented, and covered by tests. New symbolic algorithms should expose simplification controls, use TensorAtlas-specific errors for user-facing failures, and avoid import-time side effects.

Before opening a pull request, run:

```bash
python tools/release_audit.py
python -m pytest
python -m ruff check src tests examples tools
```
