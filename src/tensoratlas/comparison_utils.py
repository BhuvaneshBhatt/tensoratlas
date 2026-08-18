from __future__ import annotations

from typing import Any

import sympy as sp

from .simplification_policy import simplify_object
from .normal_forms import TNFMatrix, TNFTensorArray, as_tnf_array, as_tnf_matrix, tnf_iter_indices
from .symbolic_decision import is_equal, is_zero


def scalar_equal(left: Any, right: Any, *, mode: str = "safe", normalize: bool = True) -> bool:
    l = simplify_object(left, level="normal") if normalize else sp.sympify(left)
    r = simplify_object(right, level="normal") if normalize else sp.sympify(right)
    return is_equal(l, r, mode=mode)


def scalar_is_zero(value: Any, *, mode: str = "safe", normalize: bool = True) -> bool:
    v = simplify_object(value, level="normal") if normalize else sp.sympify(value)
    return is_zero(v, mode=mode)


def matrix_equal(left: Any, right: Any, *, mode: str = "safe", normalize: bool = True) -> bool:
    left_mat = as_tnf_matrix(left)
    right_mat = as_tnf_matrix(right)
    if left_mat.shape != right_mat.shape:
        return False
    for row in range(left_mat.rows):
        for col in range(left_mat.cols):
            raw_a = left_mat[row, col]
            raw_b = right_mat[row, col]
            if raw_a == raw_b or str(raw_a) == str(raw_b):
                continue
            a = simplify_object(raw_a, level="normal") if normalize else raw_a
            b = simplify_object(raw_b, level="normal") if normalize else raw_b
            if a == b or str(a) == str(b):
                continue
            if not is_equal(a, b, mode=mode):
                return False
    return True


def matrix_is_zero(value: Any, *, mode: str = "safe", normalize: bool = True) -> bool:
    matrix = as_tnf_matrix(value)
    for row in range(matrix.rows):
        for col in range(matrix.cols):
            entry = simplify_object(matrix[row, col], level="normal") if normalize else matrix[row, col]
            if not is_zero(entry, mode=mode):
                return False
    return True


def tensor_equal(left: Any, right: Any, *, mode: str = "safe", normalize: bool = True) -> bool:
    left_arr = as_tnf_array(left)
    right_arr = as_tnf_array(right)
    if left_arr.shape != right_arr.shape:
        return False
    for idx in tnf_iter_indices(left_arr.shape):
        a = simplify_object(left_arr[idx], level="normal") if normalize else left_arr[idx]
        b = simplify_object(right_arr[idx], level="normal") if normalize else right_arr[idx]
        if not is_equal(a, b, mode=mode):
            return False
    return True


def tensor_is_zero(value: Any, *, mode: str = "safe", normalize: bool = True) -> bool:
    array = as_tnf_array(value)
    for idx in tnf_iter_indices(array.shape):
        entry = simplify_object(array[idx], level="normal") if normalize else array[idx]
        if not is_zero(entry, mode=mode):
            return False
    return True
