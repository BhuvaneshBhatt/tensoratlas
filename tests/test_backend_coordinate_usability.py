from fractions import Fraction

import pytest

from tensoratlas.core import (
    CoordinateMap,
    CoordinateSystem,
    Manifold,
    TensorFactor,
    TensorHead,
    TensorTerm,
    canonical_double_coset_reference,
    canonicalize_tensor,
    coordinate_laplacian_result,
    transform_tensor_field,
)
from tensoratlas.core.permutation_group_backend import Permutation, PermutationGroup


def test_coordinate_system_materializes_symbols():
    system = CoordinateSystem("local", Manifold("M", 2), ("u", "v"))
    assert system.coordinate_symbols is not None
    assert tuple(str(symbol) for symbol in system.coordinate_symbols) == ("u", "v")
    assert system.coordinate_symbols[0] is system.coordinate_symbols[0]


def test_reference_double_coset_name_is_available():
    identity = Permutation.identity(2)
    group = PermutationGroup(2, (identity,))
    result = canonical_double_coset_reference(group, identity, group)
    assert result.image == (0, 1)
    assert result.sign == 1


def test_transform_tensor_field_skips_zeros_and_records_conventions():
    import sympy as sp

    manifold = Manifold("M", 2)
    source = CoordinateSystem("xy", manifold, ("x", "y"))
    target = CoordinateSystem("uv", manifold, ("u", "v"))
    x, y = source.coordinate_symbols
    u, v = target.coordinate_symbols
    cmap = CoordinateMap(source, target, (2 * x, y), inverse=(u / 2, v), name="scale_x")

    result = transform_tensor_field(((1, 0), (0, 0)), cmap, ("up", "down"), density_weight=1)
    assert result.metadata["source_nonzero_components"] == 1
    assert result.convention_metadata["density_convention"]
    assert result.components[0][0] == sp.Abs(sp.Rational(1, 2))


def test_vector_calculus_result_metadata():
    import sympy as sp

    x, y = sp.symbols("x y", real=True)
    result = coordinate_laplacian_result(x**2 + y**2, (x, y), convention="laplace_beltrami")
    assert result.components == 4
    assert result.convention_metadata["operator"] == "laplacian"
    assert result.convention_metadata["component_basis"] == "coordinate"


def test_canonicalize_tensor_explain_result():
    manifold = Manifold("M", 2)
    bundle = manifold.index_type("T")
    a = bundle.index("a", variance="up")
    V = TensorHead("V", (bundle,), variance=("up",))
    expr = TensorTerm(Fraction(1), (TensorFactor(V, (a,)),))
    result = canonicalize_tensor(expr, explain=True)
    assert result.expression.terms
    assert result.backend_name
    assert result.warnings
