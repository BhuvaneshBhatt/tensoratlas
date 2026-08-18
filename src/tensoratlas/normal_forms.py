from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from functools import lru_cache
from typing import Iterable, Sequence

import sympy as sp

from .symbolic_decision import is_equal, is_zero
from .simplification_core import light_simplify, canonical_simplify

TNF_SIMPLIFY_INNER = False


def _tnf_simplify_expr(expr):
    expr = sp.sympify(expr)
    if TNF_SIMPLIFY_INNER:
        return canonical_simplify(expr, final=True)
    return light_simplify(expr)


def _tnf_is_zero(expr) -> bool:
    return is_zero(_tnf_simplify_expr(expr))


def tnf_iter_indices(shape: tuple[int, ...]):
    return [()] if not shape else product(*[range(s) for s in shape])


def tnf_build_array(shape: tuple[int, ...], value_func) -> 'TNFTensorArray':
    if not shape:
        return TNFTensorArray((), (_tnf_simplify_expr(value_func(())),))
    entries = []
    for idx in tnf_iter_indices(shape):
        entries.append(_tnf_simplify_expr(value_func(idx)))
    return TNFTensorArray(shape, tuple(entries))


def tnf_map_array(array: 'TNFTensorArray', func) -> 'TNFTensorArray':
    return TNFTensorArray(array.shape, tuple(_tnf_simplify_expr(func(entry)) for entry in array.entries))


def tnf_scalar_array(expr, *, cleaner=None) -> "TNFTensorArray":
    if cleaner is None:
        cleaner = _tnf_cleanup
    return TNFTensorArray((), (cleaner(expr),))


def tnf_build_matrix(rows: int, cols: int, value_func) -> 'TNFMatrix':
    return TNFMatrix(rows, cols, tuple(tuple(_tnf_simplify_expr(value_func(i, j)) for j in range(cols)) for i in range(rows)))


def tnf_map_matrix(matrix: 'TNFMatrix', func) -> 'TNFMatrix':
    return TNFMatrix(matrix.rows, matrix.cols, tuple(tuple(_tnf_simplify_expr(func(matrix.entries[i][j])) for j in range(matrix.cols)) for i in range(matrix.rows)))


def _is_sequence_like(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_scalar_like(value) -> bool:
    return isinstance(value, (TNFScalarAtom, sp.Expr, int, float, complex)) and not isinstance(value, (TNFMatrix, TNFTensorArray))




def _tnf_zero_matrix(rows: int, cols: int) -> list[list[sp.Expr]]:
    return [[sp.Integer(0) for _ in range(cols)] for _ in range(rows)]


def _tnf_eye(size: int) -> list[list[sp.Expr]]:
    return [[sp.Integer(1) if i == j else sp.Integer(0) for j in range(size)] for i in range(size)]


def _tnf_minor_entries(entries: tuple[tuple[sp.Expr, ...], ...], row: int, col: int) -> tuple[tuple[sp.Expr, ...], ...]:
    return tuple(
        tuple(entries[i][j] for j in range(len(entries[0])) if j != col)
        for i in range(len(entries)) if i != row
    )


def _tnf_det_recursive_entries(entries: tuple[tuple[sp.Expr, ...], ...]) -> sp.Expr:
    n = len(entries)
    if n == 0:
        return sp.Integer(1)
    if n == 1:
        return _tnf_simplify_expr(entries[0][0])
    if n == 2:
        return _tnf_simplify_expr(entries[0][0] * entries[1][1] - entries[0][1] * entries[1][0])
    total = sp.Integer(0)
    for j, value in enumerate(entries[0]):
        if _tnf_is_zero(value):
            continue
        cofactor = _tnf_det_recursive_entries(_tnf_minor_entries(entries, 0, j))
        total += ((-1) ** j) * value * cofactor
    return _tnf_simplify_expr(total)


def _tnf_bareiss_det(matrix: "TNFMatrix") -> sp.Expr:
    if matrix.rows != matrix.cols:
        raise ValueError('Determinant requires a square matrix.')
    n = matrix.rows
    if n == 0:
        return sp.Integer(1)
    work = [list(row) for row in matrix.entries]
    sign = sp.Integer(1)
    prev = sp.Integer(1)
    for k in range(n - 1):
        pivot_row = None
        for r in range(k, n):
            if not _tnf_is_zero(work[r][k]):
                pivot_row = r
                break
        if pivot_row is None:
            return sp.Integer(0)
        if pivot_row != k:
            work[k], work[pivot_row] = work[pivot_row], work[k]
            sign = -sign
        pivot = _tnf_simplify_expr(work[k][k])
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                work[i][j] = _tnf_simplify_expr((work[i][j] * pivot - work[i][k] * work[k][j]) / prev)
        prev = pivot
        for i in range(k + 1, n):
            work[i][k] = sp.Integer(0)
        for j in range(k + 1, n):
            work[k][j] = work[k][j]
    return _tnf_simplify_expr(sign * work[n - 1][n - 1])


def _tnf_adjugate(matrix: "TNFMatrix") -> "TNFMatrix":
    if matrix.rows != matrix.cols:
        raise ValueError('Adjugate requires a square matrix.')
    n = matrix.rows
    if n == 0:
        return TNFMatrix(0, 0, tuple())
    if n == 1:
        return TNFMatrix(1, 1, ((sp.Integer(1),),))
    return tnf_build_matrix(
        n,
        n,
        lambda i, j: _tnf_simplify_expr(((-1) ** (i + j)) * _tnf_det_recursive_entries(_tnf_minor_entries(matrix.entries, j, i))),
    )


def _tnf_gauss_jordan_inverse(matrix: "TNFMatrix") -> "TNFMatrix":
    if matrix.rows != matrix.cols:
        raise ValueError('Matrix inverse requires a square matrix.')
    n = matrix.rows
    left = [list(row) for row in matrix.entries]
    right = _tnf_eye(n)
    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if not _tnf_is_zero(left[r][col]):
                pivot_row = r
                break
        if pivot_row is None:
            raise ValueError('Matrix is singular.')
        if pivot_row != col:
            left[col], left[pivot_row] = left[pivot_row], left[col]
            right[col], right[pivot_row] = right[pivot_row], right[col]
        pivot = _tnf_simplify_expr(left[col][col])
        left[col] = [_tnf_simplify_expr(v / pivot) for v in left[col]]
        right[col] = [_tnf_simplify_expr(v / pivot) for v in right[col]]
        for r in range(n):
            if r == col:
                continue
            factor = _tnf_simplify_expr(left[r][col])
            if factor == 0:
                continue
            left[r] = [_tnf_simplify_expr(left[r][c] - factor * left[col][c]) for c in range(n)]
            right[r] = [_tnf_simplify_expr(right[r][c] - factor * right[col][c]) for c in range(n)]
    return TNFMatrix(n, n, tuple(tuple(_tnf_simplify_expr(right[i][j]) for j in range(n)) for i in range(n)))


def _tnf_trace(matrix: "TNFMatrix") -> sp.Expr:
    if matrix.rows != matrix.cols:
        raise ValueError('Trace requires a square matrix.')
    return _tnf_simplify_expr(sum(matrix.entries[i][i] for i in range(matrix.rows)))


def _tnf_rref_with_pivots(matrix: "TNFMatrix") -> tuple[list[list[sp.Expr]], list[int]]:
    rows, cols = matrix.rows, matrix.cols
    work = [list(row) for row in matrix.entries]
    pivot_cols: list[int] = []
    r = 0
    for c in range(cols):
        pivot = None
        for rr in range(r, rows):
            if not _tnf_is_zero(work[rr][c]):
                pivot = rr
                break
        if pivot is None:
            continue
        if pivot != r:
            work[r], work[pivot] = work[pivot], work[r]
        pv = _tnf_simplify_expr(work[r][c])
        work[r] = [_tnf_simplify_expr(v / pv) for v in work[r]]
        for rr in range(rows):
            if rr == r:
                continue
            factor = _tnf_simplify_expr(work[rr][c])
            if _tnf_is_zero(factor):
                continue
            work[rr] = [_tnf_simplify_expr(work[rr][jj] - factor * work[r][jj]) for jj in range(cols)]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    return work, pivot_cols


def _tnf_rank(matrix: "TNFMatrix") -> int:
    _, pivot_cols = _tnf_rref_with_pivots(matrix)
    return len(pivot_cols)


def _tnf_from_sympy_matrix(matrix: sp.MatrixBase) -> "TNFMatrix":
    return TNFMatrix.from_sympy(sp.Matrix(matrix))


def _tnf_nullspace_basis(matrix: "TNFMatrix") -> list[list[sp.Expr]]:
    rows, cols = matrix.rows, matrix.cols
    work, pivot_cols = _tnf_rref_with_pivots(matrix)
    free_cols = [c for c in range(cols) if c not in pivot_cols]
    if not free_cols:
        return []
    basis: list[list[sp.Expr]] = []
    for free in free_cols:
        vec = [sp.Integer(0) for _ in range(cols)]
        vec[free] = sp.Integer(1)
        for row_idx, pivot_col in enumerate(pivot_cols):
            vec[pivot_col] = _tnf_simplify_expr(-work[row_idx][free])
        basis.append(vec)
    return basis


def _tnf_column_basis(matrix: "TNFMatrix", pivot_cols: list[int]) -> "TNFMatrix":
    return TNFMatrix(matrix.rows, len(pivot_cols), tuple(tuple(matrix.entries[i][j] for j in pivot_cols) for i in range(matrix.rows)))


def _tnf_row_basis_from_rref(rref_rows: list[list[sp.Expr]], rank: int, cols: int) -> "TNFMatrix":
    return TNFMatrix(rank, cols, tuple(tuple(_tnf_simplify_expr(rref_rows[i][j]) for j in range(cols)) for i in range(rank)))


def _tnf_rank_decomposition(matrix: "TNFMatrix") -> tuple["TNFMatrix", "TNFMatrix", int]:
    rref_rows, pivot_cols = _tnf_rref_with_pivots(matrix)
    rank = len(pivot_cols)
    C = _tnf_column_basis(matrix, pivot_cols)
    F = _tnf_row_basis_from_rref(rref_rows, rank, matrix.cols)
    return C, F, rank


def _tnf_general_pinv(matrix: "TNFMatrix") -> "TNFMatrix":
    C, F, rank = _tnf_rank_decomposition(matrix)
    if rank == 0:
        return TNFMatrix(matrix.cols, matrix.rows, tuple(tuple(sp.Integer(0) for _ in range(matrix.rows)) for _ in range(matrix.cols)))
    left = (C.T @ C).inv()
    right = (F @ F.T).inv()
    return F.T @ right @ left @ C.T




def _tnf_matrix_power(matrix: "TNFMatrix", exponent: int) -> "TNFMatrix":
    if matrix.rows != matrix.cols:
        raise ValueError('Matrix powers require a square matrix.')
    if exponent < 0:
        raise ValueError('Matrix powers require a nonnegative exponent.')
    result = TNFMatrix(matrix.rows, matrix.cols, tuple(tuple(sp.Integer(1) if i == j else sp.Integer(0) for j in range(matrix.cols)) for i in range(matrix.rows)))
    base = matrix
    exp = exponent
    while exp:
        if exp & 1:
            result = result @ base
        exp >>= 1
        if exp:
            base = base @ base
    return result


def _tnf_matrix_from_columns(columns: list[list[sp.Expr]], rows: int | None = None) -> "TNFMatrix":
    if not columns:
        r = 0 if rows is None else rows
        return TNFMatrix(r, 0, tuple(() for _ in range(r)))
    r = len(columns[0]) if rows is None else rows
    return TNFMatrix(r, len(columns), tuple(tuple(_tnf_simplify_expr(columns[j][i]) for j in range(len(columns))) for i in range(r)))


def _tnf_apply_matrix_to_vector(matrix: "TNFMatrix", vector: list[sp.Expr]) -> list[sp.Expr]:
    if matrix.cols != len(vector):
        raise ValueError('Matrix/vector shapes are not compatible.')
    return [_tnf_simplify_expr(sum(matrix.entries[i][j] * vector[j] for j in range(matrix.cols))) for i in range(matrix.rows)]


def _tnf_vector_is_zero(vector: list[sp.Expr]) -> bool:
    return all(_tnf_is_zero(v) for v in vector)


def _tnf_vector_in_span(vector: list[sp.Expr], basis: list[list[sp.Expr]]) -> bool:
    if not basis:
        return _tnf_vector_is_zero(vector)
    rows = len(vector)
    basis_rank = len(_tnf_rref_with_pivots(_tnf_matrix_from_columns(basis, rows))[1])
    aug_rank = len(_tnf_rref_with_pivots(_tnf_matrix_from_columns(basis + [vector], rows))[1])
    return aug_rank == basis_rank


def _tnf_select_independent_vectors(vectors: list[list[sp.Expr]], rows: int | None = None) -> list[list[sp.Expr]]:
    out: list[list[sp.Expr]] = []
    expected_rows = rows
    for vec in vectors:
        cleaned = [_tnf_simplify_expr(v) for v in vec]
        if expected_rows is None:
            expected_rows = len(cleaned)
        if not _tnf_vector_in_span(cleaned, out):
            out.append(cleaned)
    return out


def _tnf_jordan_chains_for_eigenvalue(matrix: "TNFMatrix", eigenvalue, multiplicity: int) -> list[list[list[sp.Expr]]]:
    n = matrix.rows
    identity = TNFMatrix(n, n, tuple(tuple(sp.Integer(1) if i == j else sp.Integer(0) for j in range(n)) for i in range(n)))
    nilpotent = matrix - eigenvalue * identity
    nilpotent_powers = {0: identity}
    kernels: dict[int, list[list[sp.Expr]]] = {0: []}
    dims = [0]
    for k in range(1, multiplicity + 1):
        nilpotent_powers[k] = nilpotent @ nilpotent_powers[k - 1]
        kernel = _tnf_nullspace_basis(nilpotent_powers[k])
        kernels[k] = _tnf_select_independent_vectors(kernel, n)
        dims.append(len(kernels[k]))
    dims.append(dims[-1])
    chains: list[list[list[sp.Expr]]] = []
    for k in range(multiplicity, 0, -1):
        exact_blocks = 2 * dims[k] - dims[k - 1] - dims[k + 1]
        if exact_blocks <= 0:
            continue
        quotient_basis = _tnf_select_independent_vectors(list(kernels[k - 1]), n)
        if k < multiplicity:
            images = [_tnf_apply_matrix_to_vector(nilpotent, vec) for vec in kernels[k + 1]]
            quotient_basis = _tnf_select_independent_vectors(quotient_basis + images, n)
        heads: list[list[sp.Expr]] = []
        for vec in kernels[k]:
            if len(heads) >= exact_blocks:
                break
            if _tnf_vector_in_span(vec, quotient_basis + heads):
                continue
            highest = _tnf_apply_matrix_to_vector(nilpotent_powers[k - 1], vec)
            if _tnf_vector_is_zero(highest):
                continue
            heads.append(vec)
        for head in heads:
            chain = []
            current = head
            lifts = [head]
            for _ in range(1, k):
                current = _tnf_apply_matrix_to_vector(nilpotent, current)
                lifts.append(current)
            chain = list(reversed(lifts))
            chains.append(chain)
    return chains


def _tnf_jordan_block(eigenvalue, size: int) -> sp.Matrix:
    block = sp.zeros(size)
    for i in range(size):
        block[i, i] = eigenvalue
        if i + 1 < size:
            block[i, i + 1] = 1
    return block

def _tnf_charpoly_expr(matrix: "TNFMatrix", lam=None) -> sp.Expr:
    if matrix.rows != matrix.cols:
        raise ValueError('Characteristic polynomial requires a square matrix.')
    n = matrix.rows
    x = sp.Symbol('lambda') if lam is None else lam
    if n == 0:
        return sp.Integer(1)
    B = TNFMatrix(n, n, tuple(tuple(matrix.entries[i][j] for j in range(n)) for i in range(n)))
    coeffs: list[sp.Expr] = []
    I = TNFMatrix(n, n, tuple(tuple(sp.Integer(1) if i == j else sp.Integer(0) for j in range(n)) for i in range(n)))
    for k in range(1, n + 1):
        ck = sp.simplify(-_tnf_trace(B) / k)
        coeffs.append(ck)
        if k != n:
            B = matrix @ (B + ck * I)
    poly = x ** n
    for k, ck in enumerate(coeffs, start=1):
        poly += sp.simplify(ck) * x ** (n - k)
    return sp.expand(poly)


def _tnf_pfaffian_recursive(entries: tuple[tuple[sp.Expr, ...], ...]) -> sp.Expr:
    n = len(entries)
    if n % 2 == 1:
        return sp.Integer(0)
    if n == 0:
        return sp.Integer(1)
    if n == 2:
        return sp.simplify(entries[0][1])
    total = sp.Integer(0)
    for j in range(1, n):
        a0j = sp.simplify(entries[0][j])
        if a0j == 0:
            continue
        keep = [k for k in range(1, n) if k != j]
        sub = tuple(tuple(entries[r][c] for c in keep) for r in keep)
        total += ((-1) ** (j + 1)) * a0j * _tnf_pfaffian_recursive(sub)
    return _tnf_simplify_expr(total)

def _coerce_nested_tensor(value) -> tuple[tuple[int, ...], tuple[sp.Expr, ...]]:
    if isinstance(value, TNFTensorArray):
        return value.shape, value.entries
    if isinstance(value, TNFMatrix):
        return (value.rows, value.cols), tuple(value.entries[i][j] for i in range(value.rows) for j in range(value.cols))
    if isinstance(value, sp.MatrixBase):
        mat = sp.Matrix(value)
        return (mat.rows, mat.cols), tuple(_tnf_simplify_expr(mat[i, j]) for i in range(mat.rows) for j in range(mat.cols))
    if isinstance(value, sp.NDimArray):
        array = sp.MutableDenseNDimArray(value)
        shape = tuple(array.shape)
        if not shape:
            return (), (_tnf_simplify_expr(array[()]),)
        return shape, tuple(_tnf_simplify_expr(array[idx]) for idx in tnf_iter_indices(shape))
    if isinstance(value, (sp.Expr, int, float, complex)):
        return (), (_tnf_simplify_expr(value),)
    if _is_sequence_like(value):
        array = sp.MutableDenseNDimArray(value)
        shape = tuple(array.shape)
        if not shape:
            return (), (_tnf_simplify_expr(array[()]),)
        return shape, tuple(_tnf_simplify_expr(array[idx]) for idx in tnf_iter_indices(shape))
    raise TypeError(f'Cannot coerce {type(value)!r} to an TNF tensor container.')


@dataclass(frozen=True)
class TNFScalarAtom:
    expr: sp.Expr

    def to_sympy(self) -> sp.Expr:
        return _tnf_simplify_expr(self.expr)

    def __sympy__(self):
        return self.to_sympy()

    def __str__(self) -> str:
        return str(self.to_sympy())

    def subs(self, *args, **kwargs):
        return TNFScalarAtom(_tnf_simplify_expr(self.expr.subs(*args, **kwargs)))

    def replace(self, *args, **kwargs):
        return TNFScalarAtom(_tnf_simplify_expr(self.expr.replace(*args, **kwargs)))

    def __add__(self, other):
        return _tnf_simplify_expr(self.to_sympy() + tnf_scalar_to_sympy(other))

    def __radd__(self, other):
        return _tnf_simplify_expr(tnf_scalar_to_sympy(other) + self.to_sympy())

    def __sub__(self, other):
        return _tnf_simplify_expr(self.to_sympy() - tnf_scalar_to_sympy(other))

    def __rsub__(self, other):
        return _tnf_simplify_expr(tnf_scalar_to_sympy(other) - self.to_sympy())

    def __mul__(self, other):
        return _tnf_simplify_expr(self.to_sympy() * tnf_scalar_to_sympy(other))

    def __rmul__(self, other):
        return _tnf_simplify_expr(tnf_scalar_to_sympy(other) * self.to_sympy())

    def __truediv__(self, other):
        return _tnf_simplify_expr(self.to_sympy() / tnf_scalar_to_sympy(other))

    def __rtruediv__(self, other):
        return _tnf_simplify_expr(tnf_scalar_to_sympy(other) / self.to_sympy())

    def __neg__(self):
        return _tnf_simplify_expr(-self.to_sympy())


@dataclass(frozen=True)
class TNFMatrix:
    @classmethod
    def zeros(cls, rows: int, cols: int) -> 'TNFMatrix':
        return cls(rows, cols, tuple(tuple(sp.Integer(0) for _ in range(cols)) for _ in range(rows)))

    rows: int
    cols: int
    entries: tuple[tuple[sp.Expr, ...], ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @classmethod
    def from_sympy(cls, matrix: sp.Matrix) -> 'TNFMatrix':
        matrix = sp.Matrix(matrix)
        return cls(matrix.rows, matrix.cols, tuple(tuple(_tnf_simplify_expr(matrix[i, j]) for j in range(matrix.cols)) for i in range(matrix.rows)))

    def to_sympy(self) -> sp.Matrix:
        return sp.Matrix(self.rows, self.cols, lambda i, j: self.entries[i][j])

    def transpose(self) -> 'TNFMatrix':
        return tnf_build_matrix(self.cols, self.rows, lambda i, j: self.entries[j][i])

    def map_entries(self, func) -> 'TNFMatrix':
        return tnf_map_matrix(self, func)

    @property
    def T(self) -> 'TNFMatrix':
        return self.transpose()

    def inv(self) -> 'TNFMatrix':
        return _tnf_gauss_jordan_inverse(self)

    def det(self) -> TNFScalarAtom:
        return TNFScalarAtom(_tnf_bareiss_det(self))

    def jacobian(self, coords: Iterable[sp.Symbol]) -> 'TNFMatrix':
        coords = tuple(coords)
        if self.cols == 1:
            return tnf_build_matrix(self.rows, len(coords), lambda i, j: sp.diff(self.entries[i][0], coords[j]))
        if self.rows == 1:
            return tnf_build_matrix(self.cols, len(coords), lambda i, j: sp.diff(self.entries[0][i], coords[j]))
        pieces = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.extend(_tnf_simplify_expr(sp.diff(self.entries[i][j], coord)) for coord in coords)
            pieces.append(tuple(row))
        return TNFMatrix(self.rows, self.cols * len(coords), tuple(pieces))

    def pinv(self) -> 'TNFMatrix':
        if self.rows == self.cols:
            try:
                return self.inv()
            except Exception:
                pass
        return _tnf_general_pinv(self)

    def eigenvals(self):
        lam = sp.Symbol('lambda')
        return sp.roots(self.charpoly(lam).as_expr(), lam)

    def generalized_eigenspace_basis(self, eigenvalue, order: int | None = None):
        if self.rows != self.cols:
            raise ValueError('Generalized eigenspaces require a square matrix.')
        if order is None:
            order = self.rows
        identity = TNFMatrix(self.rows, self.cols, tuple(tuple(sp.Integer(1) if i == j else sp.Integer(0) for j in range(self.cols)) for i in range(self.rows)))
        shifted = self - eigenvalue * identity
        power = identity
        for _ in range(order):
            power = shifted @ power
        return [sp.Matrix(vec) for vec in _tnf_nullspace_basis(power)]

    def eigenvects(self):
        if self.rows != self.cols:
            raise ValueError('Eigenvectors require a square matrix.')
        identity = TNFMatrix(self.rows, self.cols, tuple(tuple(sp.Integer(1) if i == j else sp.Integer(0) for j in range(self.cols)) for i in range(self.rows)))
        out = []
        for eigenvalue, multiplicity in self.eigenvals().items():
            shifted = self - eigenvalue * identity
            basis = [sp.Matrix(vec) for vec in _tnf_nullspace_basis(shifted)]
            if not basis:
                basis = self.generalized_eigenspace_basis(eigenvalue, multiplicity)
            out.append((eigenvalue, multiplicity, basis))
        return out

    def jordan_form(self):
        n = self.rows
        if self.rows != self.cols:
            raise ValueError('Jordan form requires a square matrix.')
        try:
            basis_columns = []
            jordan_blocks = []
            for eigenvalue, multiplicity in self.eigenvals().items():
                chains = _tnf_jordan_chains_for_eigenvalue(self, eigenvalue, multiplicity)
                for chain in chains:
                    basis_columns.extend(sp.Matrix(vec) for vec in chain)
                    jordan_blocks.append(_tnf_jordan_block(eigenvalue, len(chain)))
            if len(basis_columns) == n:
                P = sp.Matrix.hstack(*basis_columns) if basis_columns else sp.eye(n)
                if P.det() != 0:
                    J = sp.diag(*jordan_blocks) if jordan_blocks else sp.zeros(n)
                    return P, J
        except Exception:
            pass
        return self.to_sympy().jordan_form()

    def charpoly(self, lam=None):
        sym = sp.Symbol('lambda') if lam is None else lam
        return sp.PurePoly(_tnf_charpoly_expr(self, sym), sym, domain='EX')

    def pfaffian(self):
        if self.rows != self.cols:
            raise ValueError('Pfaffian requires a square matrix.')
        if self.rows % 2 == 1:
            return sp.Integer(0)
        if any(_tnf_simplify_expr(self.entries[i][j] + self.entries[j][i]) != 0 for i in range(self.rows) for j in range(self.cols)):
            raise ValueError('Pfaffian is defined for skew-symmetric matrices.')
        return _tnf_pfaffian_recursive(self.entries)

    def _matmul_tnf(self, other: 'TNFMatrix') -> 'TNFMatrix':
        if self.cols != other.rows:
            raise ValueError('Matrix shapes are not compatible for multiplication.')
        return tnf_build_matrix(self.rows, other.cols, lambda i, j: sum(self.entries[i][k] * other.entries[k][j] for k in range(self.cols)))

    def __matmul__(self, other) -> 'TNFMatrix':
        return self._matmul_tnf(as_tnf_matrix(other))

    def __mul__(self, other):
        if _is_scalar_like(other):
            scalar = tnf_scalar_to_sympy(other)
            return tnf_build_matrix(self.rows, self.cols, lambda i, j: self.entries[i][j] * scalar)
        return self._matmul_tnf(as_tnf_matrix(other))

    def __rmul__(self, other):
        if _is_scalar_like(other):
            scalar = tnf_scalar_to_sympy(other)
            return tnf_build_matrix(self.rows, self.cols, lambda i, j: scalar * self.entries[i][j])
        return as_tnf_matrix(other)._matmul_tnf(self)

    def __add__(self, other):
        other_mat = as_tnf_matrix(other)
        if self.shape != other_mat.shape:
            raise ValueError('Matrix shapes must match for addition.')
        return tnf_build_matrix(self.rows, self.cols, lambda i, j: self.entries[i][j] + other_mat.entries[i][j])

    def __sub__(self, other):
        other_mat = as_tnf_matrix(other)
        if self.shape != other_mat.shape:
            raise ValueError('Matrix shapes must match for subtraction.')
        return tnf_build_matrix(self.rows, self.cols, lambda i, j: self.entries[i][j] - other_mat.entries[i][j])

    def __neg__(self):
        return tnf_build_matrix(self.rows, self.cols, lambda i, j: -self.entries[i][j])

    def __truediv__(self, other):
        scalar = tnf_scalar_to_sympy(other)
        return tnf_build_matrix(self.rows, self.cols, lambda i, j: self.entries[i][j] / scalar)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, j = key
            return self.entries[i][j]
        if self.cols == 1:
            return self.entries[key][0]
        if self.rows == 1:
            return self.entries[0][key]
        return self.entries[key]

    def __len__(self) -> int:
        return self.rows if self.cols == 1 else self.rows * self.cols

    def __iter__(self):
        if self.cols == 1:
            for i in range(self.rows):
                yield self.entries[i][0]
            return
        if self.rows == 1:
            for j in range(self.cols):
                yield self.entries[0][j]
            return
        for i in range(self.rows):
            for j in range(self.cols):
                yield self.entries[i][j]

    def subs(self, *args, **kwargs) -> 'TNFMatrix':
        return tnf_build_matrix(self.rows, self.cols, lambda i, j: self.entries[i][j].subs(*args, **kwargs))

    def replace(self, *args, **kwargs) -> 'TNFMatrix':
        return tnf_build_matrix(self.rows, self.cols, lambda i, j: self.entries[i][j].replace(*args, **kwargs))

    def applyfunc(self, func) -> 'TNFMatrix':
        return tnf_build_matrix(self.rows, self.cols, lambda i, j: func(self.entries[i][j]))

    def equals(self, other) -> bool:
        other_mat = as_tnf_matrix(other)
        if self.shape != other_mat.shape:
            return False
        return all(is_equal(self.entries[i][j], other_mat.entries[i][j]) for i in range(self.rows) for j in range(self.cols))

    def __eq__(self, other) -> bool:
        try:
            return self.equals(other)
        except Exception:
            return False

    def __sympy__(self):
        return self.to_sympy()

    def __repr__(self) -> str:
        return f'TNFMatrix({repr(self.to_sympy())})'


@dataclass(frozen=True)
class TNFTensorArray:
    @classmethod
    def zeros(cls, *shape: int) -> 'TNFTensorArray':
        if not shape:
            return cls((), (sp.Integer(0),))
        total = 1
        for s in shape:
            total *= s
        return cls(tuple(shape), tuple(sp.Integer(0) for _ in range(total)))

    shape: tuple[int, ...]
    entries: tuple[sp.Expr, ...]

    def rank(self) -> int:
        return len(self.shape)

    @classmethod
    def from_sympy(cls, array) -> 'TNFTensorArray':
        shape, entries = _coerce_nested_tensor(array)
        return cls(shape, entries)

    def to_sympy(self) -> sp.MutableDenseNDimArray:
        if not self.shape:
            arr = sp.MutableDenseNDimArray.zeros()
            arr[()] = self.entries[0]
            return arr
        arr = sp.MutableDenseNDimArray.zeros(*self.shape)
        for flat_idx, idx in enumerate(tnf_iter_indices(self.shape)):
            arr[idx] = self.entries[flat_idx]
        return arr

    def _flat_index(self, key) -> int:
        if not self.shape:
            return 0
        if not isinstance(key, tuple):
            key = (key,)
        flat = 0
        stride = 1
        for axis in range(len(self.shape) - 1, -1, -1):
            flat += key[axis] * stride
            stride *= self.shape[axis]
        return flat

    def __getitem__(self, key):
        return self.entries[self._flat_index(key)]

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 1

    def __iter__(self):
        return iter(self.entries)

    def subs(self, *args, **kwargs) -> 'TNFTensorArray':
        return tnf_map_array(self, lambda e: e.subs(*args, **kwargs))

    def replace(self, *args, **kwargs) -> 'TNFTensorArray':
        return tnf_map_array(self, lambda e: e.replace(*args, **kwargs))

    def applyfunc(self, func) -> 'TNFTensorArray':
        return tnf_map_array(self, func)

    def __neg__(self):
        return TNFTensorArray(self.shape, tuple(-e for e in self.entries))

    def __add__(self, other):
        return _binary_tensor_array_op(self, other, lambda a, b: a + b)

    def __sub__(self, other):
        return _binary_tensor_array_op(self, other, lambda a, b: a - b)

    def __mul__(self, other):
        if not _is_scalar_like(other):
            raise TypeError('TNFTensorArray only supports scalar multiplication.')
        scalar = tnf_scalar_to_sympy(other)
        return TNFTensorArray(self.shape, tuple(_tnf_simplify_expr(entry * scalar) for entry in self.entries))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        scalar = tnf_scalar_to_sympy(other)
        return TNFTensorArray(self.shape, tuple(_tnf_simplify_expr(entry / scalar) for entry in self.entries))

    def permutedims(self, perm):
        perm = tuple(perm)
        new_shape = tuple(self.shape[i] for i in perm)
        reverse = tuple(perm.index(i) for i in range(len(perm)))
        return tnf_build_array(new_shape, lambda out_idx: self[tuple(out_idx[reverse[i]] for i in range(len(perm)))])

    def __eq__(self, other) -> bool:
        try:
            other_arr = as_tnf_array(other)
            return self.shape == other_arr.shape and all(is_equal(a, b) for a, b in zip(self.entries, other_arr.entries))
        except Exception:
            return False

    def __sympy__(self):
        return self.to_sympy()

    def __repr__(self) -> str:
        return f'TNFTensorArray(shape={self.shape})'


def as_tnf_matrix(value) -> TNFMatrix:
    if isinstance(value, TNFMatrix):
        return value
    if isinstance(value, TNFTensorArray):
        if len(value.shape) != 2:
            raise TypeError('Only rank-2 TNFTensorArray objects can be viewed as TNFMatrix instances.')
        rows, cols = value.shape
        return TNFMatrix(rows, cols, tuple(tuple(value[(i, j)] for j in range(cols)) for i in range(rows)))
    return TNFMatrix.from_sympy(sp.Matrix(value))


def tnf_matrix_to_sympy(value) -> sp.Matrix:
    return value.to_sympy() if isinstance(value, TNFMatrix) else sp.Matrix(value)


def tnf_array_to_sympy(value) -> sp.MutableDenseNDimArray:
    return value.to_sympy() if isinstance(value, TNFTensorArray) else sp.MutableDenseNDimArray(value)


def as_tnf_array(value) -> TNFTensorArray:
    return value if isinstance(value, TNFTensorArray) else TNFTensorArray.from_sympy(value)


def tnf_scalar_to_sympy(value) -> sp.Expr:
    return value.to_sympy() if isinstance(value, TNFScalarAtom) else sp.sympify(value)


def _binary_tensor_array_op(left, right, op):
    left_arr = as_tnf_array(left)
    right_arr = as_tnf_array(right)
    if left_arr.shape != right_arr.shape:
        raise ValueError('Tensor array shapes must match.')
    return TNFTensorArray(left_arr.shape, tuple(_tnf_simplify_expr(op(a, b)) for a, b in zip(left_arr.entries, right_arr.entries)))


def tnf_column_from_entries(entries: Iterable[sp.Expr]) -> TNFMatrix:
    entries = tuple(_tnf_simplify_expr(e) for e in entries)
    return TNFMatrix(len(entries), 1, tuple((e,) for e in entries))


def tnf_identity_tensor(dim: int, rank: int) -> TNFTensorArray:
    return tnf_build_array((dim,) * rank, lambda idx: sp.Integer(1) if len(set(idx)) == 1 else sp.Integer(0))



