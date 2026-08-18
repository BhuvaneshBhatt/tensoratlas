from fractions import Fraction

import pytest

from tensoratlas.core.components import CoordinateSystem
from tensoratlas.core.coordinate_tools import CoordinateMap
from tensoratlas.core.manifolds import Manifold, TensorKernelError
from tensoratlas.core.permutation_group_backend import Permutation, PermutationGroup
from tensoratlas.core.tensor_expr import TensorFactor, TensorTerm
from tensoratlas.core.tensor_heads import TensorHead
from tensoratlas.core.indices import IndexType
from tensoratlas.core.tensor_monomial_encoding import canonicalize_repeated_factors, encode_tensor_monomial
from tensoratlas.core.field_transformations import transform_scalar_field, transform_tensor_density


def test_coordinate_map_reuses_coordinate_symbols_for_jacobian():
    import sympy as sp

    x = sp.Symbol("x", positive=True)
    u = sp.Symbol("u", positive=True)
    manifold = Manifold("Line", 1)
    source = CoordinateSystem("source", manifold, ("x",), coordinate_symbols=(x,))
    target = CoordinateSystem("target", manifold, ("u",), coordinate_symbols=(u,))
    cmap = CoordinateMap(source, target, (x**2,), inverse=(sp.sqrt(u),))

    assert cmap.source_symbols == (x,)
    assert cmap.jacobian() == ((2 * x,),)


def test_tensor_density_scaling_map_uses_inverse_jacobian_factor():
    import sympy as sp

    x = sp.Symbol("x", real=True)
    u = sp.Symbol("u", real=True)
    manifold = Manifold("Line", 1)
    source = CoordinateSystem("xchart", manifold, ("x",), coordinate_symbols=(x,))
    target = CoordinateSystem("uchart", manifold, ("u",), coordinate_symbols=(u,))
    cmap = CoordinateMap(source, target, (2 * x,), inverse=(u / 2,))

    assert transform_scalar_field(1, cmap, density_weight=1) == sp.Rational(1, 2)
    result = transform_tensor_density((1,), cmap, ("up",), 1)
    assert result.components == (1,)


def test_permutation_group_closure_guard_raises():
    group = PermutationGroup.symmetric(6)
    with pytest.raises(TensorKernelError):
        group.closure(max_size=10)


def test_canonicalize_repeated_factors_tracks_graded_sort_sign():
    manifold = Manifold("M", 2)
    tangent = IndexType("TM", manifold, dimension=2)
    OddA = TensorHead("A", (tangent,), variance=(None,), parity=1)
    OddB = TensorHead("B", (tangent,), variance=(None,), parity=1)
    a = tangent.index("a")
    term = TensorTerm(Fraction(1), (TensorFactor(OddB, (a,)), TensorFactor(OddA, (a,))))

    out = canonicalize_repeated_factors(term)

    assert out.coefficient == -1
    assert tuple(f.head.name for f in out.factors) == ("A", "B")


def test_invalid_index_use_rejected_before_encoding():
    manifold = Manifold("M", 3)
    tangent = IndexType("TM", manifold, dimension=3)
    T = TensorHead("T", (tangent, tangent, tangent), variance=(None, None, None))
    a = tangent.index("a", variance="up")
    term = TensorTerm(Fraction(1), (TensorFactor(T, (a, a, a)),))

    with pytest.raises(TensorKernelError):
        encode_tensor_monomial(term)


def test_stable_index_labels_do_not_use_object_id():
    manifold = Manifold("M", 3)
    tangent = IndexType("TM", manifold, dimension=3)
    T = TensorHead("T", (tangent,), variance=(None,))
    a = tangent.index("a", variance="up")
    encoded = encode_tensor_monomial(TensorTerm(Fraction(1), (TensorFactor(T, (a,)),)))

    assert encoded.labels[0][1][0:3] == ("M", 3, "TM")
    assert not isinstance(encoded.labels[0][1], int)
