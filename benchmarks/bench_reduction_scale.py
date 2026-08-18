from __future__ import annotations

import time

from tensoratlas import TensorIndex, IndexBundle, IndexedTensor, IndexedTensorExpr, tensor_graph, tensor_reduce


def main() -> None:
    bundle = IndexBundle("V", 4)
    i = TensorIndex("i", "up", bundle)
    j = TensorIndex("j", "down", bundle)
    k = TensorIndex("k", "up", bundle)
    l = TensorIndex("l", "down", bundle)
    a = IndexedTensor("A", (i, j))
    b = IndexedTensor("B", (k, l))
    expr = IndexedTensorExpr("tensor_product", (a, b))
    start = time.perf_counter()
    for _ in range(100):
        tensor_graph(expr)
        tensor_reduce(expr)
    elapsed = time.perf_counter() - start
    print({"reduction_scale_s": round(elapsed, 4)})


if __name__ == "__main__":
    main()
