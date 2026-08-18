from __future__ import annotations

import time

from tensoratlas import TensorObject, TensorBasis, IndexedTensor, indices, normalize_indexed_expression, indexed_equal


def main() -> None:
    basis = TensorBasis("V", "tangent", dimension=3)
    tensor = TensorObject("A", basis, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], variance="ud")
    i, j = indices("^i,_j")
    expr = IndexedTensor(tensor, (i, j))

    start = time.perf_counter()
    for _ in range(100):
        normalize_indexed_expression(expr)
        indexed_equal(expr, expr)
    elapsed = time.perf_counter() - start
    print({"indexed_nf_s": round(elapsed, 4)})


if __name__ == "__main__":
    main()
