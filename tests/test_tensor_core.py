import sympy as sp

from tensoratlas import (
    coordinate_chart,
    coordinate_map,
    ScalarField,
    TensorExpr,
    TensorObject,
    VectorField,
    ExprPattern,
    PatternRewriteRule,
    cotangent_basis,
    identity_tensor,
    orthonormal_tangent_basis,
    orthonormal_cotangent_basis,
    tangent_basis,
    direct_sum_tensor,
    indices,
    indexed_equal,
    match_indexed_pattern,
    metric_tensor,
    rewrite_with_patterns,
    TensorPattern,
)


def test_basis_change_vector_cylindrical_to_orthonormal():
    cyl = coordinate_chart("Euclidean", "Cylindrical", 3)
    r, theta, z = cyl.symbols()
    vec = VectorField(cyl, sp.Matrix([[1], [0], [0]]), "contravariant")
    tob = TensorObject.from_vector_field(vec)
    tob2 = tob.change_basis((orthonormal_tangent_basis(cyl),))
    assert sp.simplify(tob2.components[(0,)] - 1) == 0
    assert tob2.slot_bases[0].kind == "orthonormal_tangent"


def test_symmetrize_and_antisymmetrize_metadata():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    arr = sp.MutableDenseNDimArray.zeros(2, 2)
    arr[0, 1] = x
    arr[1, 0] = y
    tob = TensorObject(cart, arr, "ll", (cotangent_basis(cart), cotangent_basis(cart)))
    sym = tob.symmetrize_slots((0, 1))
    asym = tob.antisymmetrize_slots((0, 1))
    assert "symmetric" in sym.symmetry_metadata
    assert "antisymmetric" in asym.symmetry_metadata
    assert sp.simplify(sym.components[(0, 1)] - (x + y) / 2) == 0
    assert sp.simplify(asym.components[(0, 1)] - (x - y) / 2) == 0


def test_tensor_expr_addition_and_product():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    v1 = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[x], [y]]), "contravariant"))
    v2 = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[1], [2]]), "contravariant"))
    expr = TensorExpr("add", (TensorExpr("tensor", (v1,)), TensorExpr("tensor", (v2,))))
    out = expr.evaluate()
    assert sp.simplify(out.components[(0,)] - (x + 1)) == 0
    assert sp.simplify(out.components[(1,)] - (y + 2)) == 0

    cov = TensorObject.from_tensor_field(v1.to_vector_field().lower_index().as_tensor())
    prod = TensorExpr("tensor_product", (TensorExpr("tensor", (v1,)), TensorExpr("tensor", (cov,)))).evaluate()
    assert prod.variance_spec == "ul"
    assert prod.slot_bases[0].kind == "tangent"
    assert prod.slot_bases[1].kind == "cotangent"


def test_transform_preserves_basis_kinds():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    from tensoratlas import coordinate_map
    mp = coordinate_map(cart, polar)
    x, y = cart.symbols()
    v = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[x], [y]]), "contravariant"))
    vp = v.transform(mp)
    assert vp.chart == polar
    assert vp.slot_bases[0].kind == "tangent"

from tensoratlas import TensorIndex, indexed


def test_antisymmetric_canonicalize_repeated_indices_zero():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    arr = sp.MutableDenseNDimArray.zeros(3, 3)
    arr[0, 1] = 5
    arr[1, 0] = -5
    tob = TensorObject(cart, arr, "ll", (cotangent_basis(cart), cotangent_basis(cart)), symmetry_metadata={"antisymmetric": ((0, 1),)})
    reduced = tob.canonicalize_symmetry()
    assert reduced.components[(1, 1)] == 0
    assert reduced.components[(1, 0)] == -reduced.components[(0, 1)]


def test_basis_aware_tensorobject_pushforward_vector():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    polar = coordinate_chart("Euclidean", "Polar", 2)
    x, y = cart.symbols()
    vec = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[x], [y]]), "contravariant"))
    pushed = vec.push_forward(coordinate_map(cart, polar))
    assert pushed.chart == polar
    assert pushed.variance_spec == 'u'


def test_indexed_tensor_contraction_trace_identity():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    tob = TensorObject.from_tensor_field(identity_tensor(cart, 'ul'))
    i_u = TensorIndex('i', 'u')
    i_l = TensorIndex('i', 'l')
    traced = tob.with_indices(i_u, i_l).evaluate()
    assert isinstance(traced, ScalarField)
    assert traced.expr == 2


def test_indexed_addition_canonicalizes_symmetric_slots():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    arr = sp.MutableDenseNDimArray.zeros(2, 2)
    arr[0, 1] = 3
    arr[1, 0] = 3
    tob = TensorObject(cart, arr, 'll', (cotangent_basis(cart), cotangent_basis(cart)), symmetry_metadata={'symmetric': ((0,1),)})
    i = TensorIndex('i', 'l')
    j = TensorIndex('j', 'l')
    left = tob.with_indices(j, i)
    right = tob.with_indices(i, j)
    summed = (left + right).evaluate()
    assert isinstance(summed, indexed(tob, i, j).__class__)
    assert summed.indices[0].name == 'i'
    assert summed.indices[1].name == 'j'
    assert summed.tensor.components[(0,1)] == 6

from tensoratlas import kronecker_delta_tensor, metric_tensor


def test_metric_tensor_helpers_and_delta_trace():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    g = metric_tensor(polar, 'll')
    ginv = metric_tensor(polar, 'uu')
    r, theta = polar.symbols()
    assert sp.simplify(g.components[(1,1)] - r**2) == 0
    assert sp.simplify(ginv.components[(1,1)] - 1/r**2) == 0
    delta = kronecker_delta_tensor(polar)
    tob = TensorObject.from_tensor_field(delta)
    i_u = TensorIndex('i', 'u')
    i_l = TensorIndex('i', 'l')
    traced = tob.with_indices(i_u, i_l).evaluate()
    assert traced.expr == 2


def test_dummy_index_renaming_canonicalizes_equivalent_products():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    v = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[x], [y]]), 'contravariant'))
    w = TensorObject.from_tensor_field(v.to_vector_field().lower_index().as_tensor())
    a = (v.with_indices(TensorIndex('i', 'u')) * w.with_indices(TensorIndex('i', 'l'))).evaluate()
    b = (v.with_indices(TensorIndex('j', 'u')) * w.with_indices(TensorIndex('j', 'l'))).evaluate()
    assert isinstance(a, ScalarField) and isinstance(b, ScalarField)
    assert sp.simplify(a.expr - b.expr) == 0


def test_young_project_slots_basic_hook_shape():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    arr = sp.MutableDenseNDimArray.zeros(3, 3, 3)
    arr[0, 1, 2] = 1
    tob = TensorObject(cart, arr, 'lll', (cotangent_basis(cart),) * 3)
    yp = tob.young_project_slots(((0, 1), (2,)))
    # row symmetrized in first two slots
    assert sp.simplify(yp.components[(0, 1, 2)] - yp.components[(1, 0, 2)]) == 0


def test_free_index_ordering_prefers_canonical_order():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    arr = sp.MutableDenseNDimArray.zeros(2, 2)
    arr[0, 1] = 7
    arr[1, 0] = 7
    tob = TensorObject(cart, arr, 'll', (cotangent_basis(cart), cotangent_basis(cart)), symmetry_metadata={'symmetric': ((0,1),)})
    out = tob.with_indices(TensorIndex('z', 'l'), TensorIndex('a', 'l')).canonicalize_free_indices()
    assert out.indices[0].name == 'a'
    assert out.indices[1].name == 'z'


def test_delta_rewrite_substitutes_indices_before_component_expansion():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    delta = TensorObject.from_tensor_field(kronecker_delta_tensor(cart))
    v = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[x], [y]]), 'contravariant'))
    i = TensorIndex('i', 'u')
    j = TensorIndex('j', 'u')
    jl = TensorIndex('j', 'l')
    out = (delta.with_indices(i, jl) * v.with_indices(j)).evaluate()
    assert isinstance(out, indexed(v, i).__class__)
    assert out.indices[0].name == 'i'
    assert sp.simplify(out.tensor.components[(0,)] - x) == 0
    assert sp.simplify(out.tensor.components[(1,)] - y) == 0


def test_metric_rewrite_lowers_vector_index_before_full_expansion():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    g = TensorObject.from_tensor_field(metric_tensor(polar, 'll'))
    v = TensorObject.from_vector_field(VectorField(polar, sp.Matrix([[1], [2]]), 'contravariant'))
    a = TensorIndex('a', 'l')
    b_u = TensorIndex('b', 'u')
    b_l = TensorIndex('b', 'l')
    out = (g.with_indices(a, b_l) * v.with_indices(b_u)).evaluate()
    assert isinstance(out, indexed(TensorObject.from_tensor_field(v.to_vector_field().lower_index().as_tensor()), a).__class__)
    assert sp.simplify(out.tensor.components[(0,)] - 1) == 0
    assert sp.simplify(out.tensor.components[(1,)] - 2 * r**2) == 0


def test_indexed_equality_is_dummy_name_invariant():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    v = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[x], [y]]), 'contravariant'))
    w = TensorObject.from_tensor_field(v.to_vector_field().lower_index().as_tensor())
    expr1 = (v.with_indices(TensorIndex('i', 'u')) * w.with_indices(TensorIndex('i', 'l')))
    expr2 = (v.with_indices(TensorIndex('k', 'u')) * w.with_indices(TensorIndex('k', 'l')))
    left = expr1.evaluate()
    right = expr2.evaluate()
    assert isinstance(left, ScalarField) and isinstance(right, ScalarField)
    assert sp.simplify(left.expr - right.expr) == 0

from tensoratlas import levi_civita_symbol, volume_form, pretty_indexed


def test_levi_civita_symbol_basic_signs():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    eps = levi_civita_symbol(cart, 'lll')
    assert eps.components[(0, 1, 2)] == 1
    assert eps.components[(1, 0, 2)] == -1
    assert eps.components[(0, 0, 2)] == 0


def test_multi_contract_trace_rank2_identity():
    cart = coordinate_chart("Euclidean", "Cartesian", 3)
    eye = TensorObject.from_tensor_field(identity_tensor(cart, 'ul'))
    traced = eye.multi_contract([(0, 1)])
    assert isinstance(traced, ScalarField)
    assert traced.expr == 3


def test_pretty_and_substitute_indexed_expr():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    v = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[x], [y]]), 'contravariant'), name='V')
    i = TensorIndex('i', 'u')
    j = TensorIndex('j', 'u')
    expr = v.with_indices(i)
    assert 'V' in pretty_indexed(expr)
    expr2 = (expr + v.with_indices(j)).substitute(v.with_indices(j), expr)
    out = expr2.evaluate()
    assert out.indices[0].name == 'i'
    assert sp.simplify(out.tensor.components[(0,)] - 2*x) == 0


def test_metric_rewrite_lowers_tensor_slot_not_only_vector():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    r, theta = polar.symbols()
    g = TensorObject.from_tensor_field(metric_tensor(polar, 'll'), name='g')
    arr = sp.MutableDenseNDimArray.zeros(2, 2)
    arr[0, 0] = 1
    arr[1, 1] = 2
    T = TensorObject(polar, arr, 'uu', (tangent_basis(polar), tangent_basis(polar)), name='T')
    a = TensorIndex('a', 'l')
    b = TensorIndex('b', 'l')
    c = TensorIndex('c', 'u')
    out = (g.with_indices(a, b) * T.with_indices(c, b.dual())).evaluate()
    assert hasattr(out, 'indices')
    assert len(out.indices) == 2


from tensoratlas import IndexedRewriteEngine, IndexedRewriteRule, rewrite_fixed_point, canonical_indexed_form, tensor_replace, indexed_equal, IndexedTensor, IndexedTensorExpr


def test_rewrite_engine_canonicalizes_nested_grouping_and_order():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    v = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[x], [y]]), "contravariant"), name="V")
    w = TensorObject.from_tensor_field(v.to_vector_field().lower_index().as_tensor(), name="W")
    i = TensorIndex("i", "u")
    il = TensorIndex("i", "l")
    expr1 = (v.with_indices(i) * w.with_indices(il))
    expr2 = IndexedTensorExpr('tensor_product', (IndexedTensorExpr('tensor', (w.with_indices(il),)), IndexedTensorExpr('tensor', (v.with_indices(i),))))
    c1 = canonical_indexed_form(expr1)
    c2 = canonical_indexed_form(expr2)
    assert indexed_equal(c1, c2)


def test_custom_rewrite_rule_can_replace_named_leaf():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    v = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[x], [y]]), "contravariant"), name="V")
    w = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[1], [1]]), "contravariant"), name="W")
    i = TensorIndex("i", "u")
    expr = v.with_indices(i)
    rule = IndexedRewriteRule(
        'replace_V_with_W',
        lambda o: isinstance(o, IndexedTensor) and (o.tensor.name == 'V'),
        lambda o: w.with_indices(*o.indices),
    )
    out = rewrite_fixed_point(expr, [rule])
    assert isinstance(out, IndexedTensor)
    assert out.tensor.name == 'W'


def test_tensor_replace_and_canonical_form_work_together():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    v = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[x], [y]]), "contravariant"), name="V")
    u = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[1], [2]]), "contravariant"), name="U")
    i = TensorIndex("i", "u")
    j = TensorIndex("j", "u")
    expr = v.with_indices(j) + v.with_indices(i)
    out = tensor_replace(expr, v.with_indices(j), u.with_indices(j), simplify=True)
    cf = canonical_indexed_form(out)
    assert 'U' in pretty_indexed(cf) and 'V' in pretty_indexed(cf)


def test_indexed_equal_handles_grouping_differences():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    v = TensorObject.from_vector_field(VectorField(cart, sp.Matrix([[x], [y]]), 'contravariant'), name='V')
    w = TensorObject.from_tensor_field(v.to_vector_field().lower_index().as_tensor(), name='W')
    i = TensorIndex('i', 'u')
    il = TensorIndex('i', 'l')
    expr1 = (v.with_indices(i) * w.with_indices(il))
    expr2 = IndexedTensorExpr('tensor_product', (IndexedTensorExpr('tensor', (v.with_indices(i),)), IndexedTensorExpr('tensor', (w.with_indices(il),))))
    assert indexed_equal(expr1, expr2)


from tensoratlas import (
    antisymmetrizer,
    alpha_rename_dummies,
    indices,
    block_tensor,
    contract_by_index_names,
    diagonal_tensor,
    direct_sum_rank2,
    match_indexed_pattern,
    symmetrizer,
    TensorPattern,
    validate_index_sequence,
)


def test_reusable_projectors_on_rank2_tensor():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    x, y = cart.symbols()
    arr = sp.MutableDenseNDimArray.zeros(2, 2)
    arr[0, 1] = x
    arr[1, 0] = y
    tob = TensorObject(cart, arr, "ll", (cotangent_basis(cart), cotangent_basis(cart)))
    sym = symmetrizer((0, 1))(tob)
    asym = antisymmetrizer((0, 1))(tob)
    assert sp.simplify(sym.components[(0, 1)] - (x + y) / 2) == 0
    assert sp.simplify(asym.components[(0, 1)] - (x - y) / 2) == 0


def test_rank2_linear_algebra_utilities():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    arr = sp.MutableDenseNDimArray.zeros(2, 2)
    arr[0, 0] = 2
    arr[1, 1] = 3
    A = TensorObject(cart, arr, "ul", (tangent_basis(cart), cotangent_basis(cart)), name="A")
    invA = A.inverse()
    assert invA.variance_spec == "ul"
    assert sp.simplify(invA.components[(0, 0)] - sp.Rational(1, 2)) == 0
    assert sp.simplify(A.determinant() - 6) == 0
    lam = sp.Symbol("lam")
    assert sp.expand(A.characteristic_polynomial(lam) - ((lam - 2) * (lam - 3))) == 0
    evals = A.eigenvals()
    assert 2 in evals and 3 in evals


def test_commutator_and_decompositions():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    Aarr = sp.MutableDenseNDimArray.zeros(2, 2)
    Barr = sp.MutableDenseNDimArray.zeros(2, 2)
    Aarr[0, 1] = 1
    Barr[1, 0] = 1
    A = TensorObject(cart, Aarr, "ul", (tangent_basis(cart), cotangent_basis(cart)), name="A")
    B = TensorObject(cart, Barr, "ul", (tangent_basis(cart), cotangent_basis(cart)), name="B")
    C = A.commutator(B)
    assert C.variance_spec == "ul"
    assert C.components[(0, 0)] == 1
    assert C.components[(1, 1)] == -1
    S = (A + B).symmetric_part()
    K = (A + B).skew_part()
    assert sp.simplify(S.components[(0, 1)] - S.components[(1, 0)]) == 0
    assert sp.simplify(K.components[(0, 1)] + K.components[(1, 0)]) == 0


def test_diagonal_direct_sum_and_block_tensor():
    c2 = coordinate_chart("Euclidean", "Cartesian", 2)
    D = diagonal_tensor(c2, [1, 2], "ll")
    E = diagonal_tensor(c2, [3, 4], "ll")
    DS = direct_sum_rank2(D, E)
    assert DS.chart.dimension == 4
    assert DS.components[(0, 0)] == 1
    assert DS.components[(1, 1)] == 2
    assert DS.components[(2, 2)] == 3
    assert DS.components[(3, 3)] == 4

    z = TensorObject(c2, sp.MutableDenseNDimArray.zeros(2, 2), "ll", (cotangent_basis(c2), cotangent_basis(c2)))
    BLK = block_tensor([[D, z], [z, E]])
    assert BLK.chart.dimension == 4
    assert BLK.components[(0, 0)] == 1
    assert BLK.components[(3, 3)] == 4


def test_validate_index_sequence_and_contract_names():
    inds = indices("a^ a^")
    errs = validate_index_sequence(inds)
    assert errs and "repeatedly with the same variance" in errs[0]

    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    I = TensorObject.from_tensor_field(identity_tensor(cart, "ul"))
    expr = I.with_indices(*indices("i^ i_"))
    tr = expr.trace_over("i")
    assert isinstance(tr, ScalarField)
    assert sp.simplify(tr.expr - 2) == 0


def test_alpha_rename_and_pattern_matching():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    I = TensorObject.from_tensor_field(identity_tensor(cart, "ul"), name="I")
    expr = (I.with_indices(*indices("i^ i_")) * I.with_indices(*indices("i^ j_")))
    renamed = alpha_rename_dummies(expr)
    s = renamed.pretty() if hasattr(renamed, "pretty") else str(renamed)
    assert "d0" in s
    pat = TensorPattern(tensor_name="I", variance_spec="ul", index_variances=("u", "l"))
    m = match_indexed_pattern(pat, I.with_indices(*indices("a^ b_")))
    assert m is not None and m["tensor"].name == "I"

def test_expr_pattern_and_pattern_rewrite_engine():
    cart = coordinate_chart("Euclidean", "Cartesian", 2)
    I = TensorObject.from_tensor_field(identity_tensor(cart, "ul"), name="I")
    leaf = I.with_indices(*indices("a^ b_"))
    pat = ExprPattern(op='tensor_product', args=(TensorPattern(tensor_name='I'), TensorPattern(tensor_name='I')))
    expr = leaf * I.with_indices(*indices("b^ c_"))
    assert match_indexed_pattern(pat, expr) is not None
    rule = PatternRewriteRule('collapse_id', pat, lambda obj: obj.simplify())
    out = rewrite_with_patterns(expr, rule)
    assert isinstance(out, IndexedTensor)
    assert indexed_equal(out, I.with_indices(*indices('a^ c_')))


def test_indexed_equal_modulo_basis_change():
    polar = coordinate_chart("Euclidean", "Polar", 2)
    g = TensorObject.from_tensor_field(metric_tensor(polar, 'll'), name='g')
    coord = g.with_indices(*indices('a_ b_'))
    ortho_tensor = g.change_basis((orthonormal_cotangent_basis(polar), orthonormal_cotangent_basis(polar)))
    ortho = ortho_tensor.with_indices(*indices('a_ b_'))
    assert indexed_equal(coord, ortho)


def test_direct_sum_tensor_general_rank3():
    c2 = coordinate_chart("Euclidean", "Cartesian", 2)
    arr = sp.MutableDenseNDimArray.zeros(2,2,2)
    arr[(0,0,0)] = 1
    T = TensorObject(c2, arr, 'ull', (tangent_basis(c2), cotangent_basis(c2), cotangent_basis(c2)), name='T')
    DS = direct_sum_tensor(T, T)
    assert DS.chart.dimension == 4
    assert DS.components[(0,0,0)] == 1
    assert DS.components[(2,2,2)] == 1


def test_tensorobject_equivalent_modulo_symmetry():
    c2 = coordinate_chart("Euclidean", "Cartesian", 2)
    arr = sp.MutableDenseNDimArray.zeros(2,2)
    arr[(0,1)] = 1
    arr[(1,0)] = 1
    A = TensorObject(c2, arr, 'll', (cotangent_basis(c2), cotangent_basis(c2)), symmetry_metadata={'symmetric': ((0,1),)})
    B = A.permute_slots((1,0))
    assert A.equivalent(B)
