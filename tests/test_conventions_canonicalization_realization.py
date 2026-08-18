from fractions import Fraction

from tensoratlas.core import (
    CliffordAlgebra,
    CliffordConvention,
    ComponentRealizationRegistry,
    ComponentTensor,
    CoordinateSystem,
    CurvatureConvention,
    GeometryConvention,
    HodgeConvention,
    Manifold,
    SignatureConvention,
    TensorHead,
    canonicalize_expression_strong,
    epsilon_product_to_generalized_delta,
    generalized_delta_component_tensor,
    generalized_delta_expansion,
    generalized_delta_head,
    levi_civita_symbol_component_tensor,
    realize_tensor_expression,
)


def test_formal_convention_signs_are_explicit():
    sig = SignatureConvention.lorentzian_mostly_plus(4)
    assert sig.positive == 3
    assert sig.negative == 1
    assert HodgeConvention(sig).star_square_sign(2) == -1
    assert CurvatureConvention("plus").conversion_factor_to(CurvatureConvention("minus")) == -1
    assert CliffordConvention(sig, "minus").gamma_factor == -2
    assert isinstance(GeometryConvention(sig).hodge, HodgeConvention)


def test_strong_canonicalization_handles_riemann_pair_symmetry_and_dummy_names():
    manifold = Manifold("M", 4)
    index_type = manifold.index_type("T")
    a, b, c, d = index_type.indices("a b c d", variance="down")
    riemann = TensorHead.riemann("R", index_type, variance=("down", "down", "down", "down"))
    expr = riemann(b, a, d, c)
    report = canonicalize_expression_strong(expr)
    assert not report.budget_exhausted
    assert repr(report.expression) == "R(_a,_b,_c,_d)"


def test_strong_canonicalization_merges_dummy_equivalent_products():
    manifold = Manifold("M", 3)
    index_type = manifold.index_type("T")
    a, b = index_type.indices("a b")
    x, y = index_type.indices("x y")
    cov_a, cov_b = index_type.indices("a b", variance="down")
    cov_x, cov_y = index_type.indices("x y", variance="down")
    tensor = TensorHead("T", (index_type, index_type), variance=("up", "down"))
    expr = tensor(a, cov_b) * tensor(b, cov_a) - tensor(x, cov_y) * tensor(y, cov_x)
    report = canonicalize_expression_strong(expr)
    assert report.expression.is_zero


def test_epsilon_and_generalized_delta_component_identities():
    manifold = Manifold("M", 3)
    coordinates = CoordinateSystem("cart", manifold, ("x", "y", "z"))
    basis = coordinates.coordinate_basis()
    epsilon = levi_civita_symbol_component_tensor("eps", basis)
    assert epsilon.component(0, 1, 2) == 1
    assert epsilon.component(1, 0, 2) == -1
    delta2 = generalized_delta_component_tensor("Delta", basis, 2)
    assert delta2.component(0, 1, 0, 1) == 1
    assert delta2.component(0, 1, 1, 0) == -1


def test_generalized_delta_expands_to_ordinary_delta_products():
    manifold = Manifold("M", 3)
    index_type = manifold.index_type("T")
    a, b = index_type.indices("a b")
    c, d = index_type.indices("c d", variance="down")
    head = generalized_delta_head("Delta", index_type, 2)
    expansion = generalized_delta_expansion(head, (a, b, c, d))
    assert "delta_T(^a,_c)*delta_T(^b,_d)" in repr(expansion)
    assert "delta_T(^a,_d)*delta_T(^b,_c)" in repr(expansion)


def test_epsilon_product_rewrites_to_generalized_delta():
    manifold = Manifold("M", 3)
    index_type = manifold.index_type("T")
    a, b, c = index_type.indices("a b c")
    cov_a, cov_b, cov_c = index_type.indices("a b c", variance="down")
    eps_up = TensorHead.epsilon("eps", index_type, variance="up")
    eps_down = TensorHead.epsilon("eps", index_type, variance="down")
    result = epsilon_product_to_generalized_delta(eps_up(a, b, c) * eps_down(cov_a, cov_b, cov_c))
    assert repr(result).startswith("Delta(")


def test_abstract_expression_realizes_to_components_with_dummy_summation():
    manifold = Manifold("M", 3)
    coordinates = CoordinateSystem("cart", manifold, ("x", "y", "z"))
    basis = coordinates.coordinate_basis()
    index_type = basis.index_type
    i = index_type.index("i")
    j = index_type.index("j")
    cov_i = index_type.index("i", variance="down")
    vector_head = TensorHead("V", (index_type,), variance=("up",))
    covector_head = TensorHead("W", (index_type,), variance=("down",))
    vector = ComponentTensor(vector_head, basis, {(0,): 2, (1,): 3, (2,): 5}, variance=("up",))
    covector = ComponentTensor(covector_head, basis, {(0,): 7, (1,): 11, (2,): 13}, variance=("down",))
    registry = ComponentRealizationRegistry(basis)
    registry.register(vector)
    registry.register(covector)
    assert realize_tensor_expression(vector_head(i), registry).to_dense() == [Fraction(2), Fraction(3), Fraction(5)]
    contracted = realize_tensor_expression(vector_head(i) * covector_head(cov_i), registry)
    assert contracted == Fraction(2 * 7 + 3 * 11 + 5 * 13)
    free = realize_tensor_expression(vector_head(i) * covector_head(-j), registry)
    assert free.component(1, 2) == Fraction(3 * 13)


def test_clifford_convention_changes_gamma_anticommutator_sign():
    manifold = Manifold("M", 4)
    vector_type = manifold.index_type("T")
    spinor_type = manifold.index_type("S")
    a, b = vector_type.indices("a b")
    A = spinor_type.index("A", variance="down")
    B = spinor_type.index("B")
    metric = TensorHead.inverse_metric("g", vector_type)
    convention = CliffordConvention(SignatureConvention.lorentzian_mostly_plus(4), "minus")
    algebra = CliffordAlgebra(vector_type, metric, convention=convention)
    assert repr(algebra.anticommutator(a, b, A, B)).startswith("-2*")
