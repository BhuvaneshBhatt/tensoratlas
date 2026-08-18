# Testing

Run the full test suite from the repository root:

```bash
python -m pytest
```

Run the release audit before publishing:

```bash
python tools/release_audit.py
```

For coordinate/field-transformation work:

```bash
python -m pytest tests/test_coordinate_catalog_and_vector_calculus.py \
  tests/test_coordinate_completion_pass.py \
  tests/test_coordinate_transform_field_array_completion.py \
  tests/test_coordinate_tensor_bugfixes.py \
  tests/test_backend_coordinate_usability.py
```

For tensor-canonicalization work:

```bash
python -m pytest tests/test_permutation_group_backend.py \
  tests/test_tensor_monomial_encoding.py \
  tests/test_tensor_monomial_encoding_completion.py \
  tests/test_tensor_encoding_optimized_cases.py \
  tests/test_tensor_canonicalization_backend_completion.py
```

For notebook-facing examples and usability helpers:

```bash
python -m pytest tests/test_demo_notebook_content.py \
  tests/test_public_examples.py \
  tests/test_usability_improvements.py
```

If `nbmake` is installed, execute the tutorial notebook from a clean kernel:

```bash
python -m pytest --nbmake notebooks/tensoratlas_demo.ipynb
```

The pure-Python permutation backend uses guarded explicit group closure in reference paths. Tests should keep such examples small and compare optimized paths against the reference oracle only where closure is safely bounded.
