from __future__ import annotations

from tensoratlas import IndexBundle, TensorIndex, IndexedTensor, IndexedTensorExpr, tensor_graph, tensor_reduce
from benchmarks._common import run_case, print_report


def planner_workload() -> dict[str, int]:
    bundle = IndexBundle('V', 5)
    iu = lambda n: TensorIndex(f'i{n}', 'u', bundle)
    il = lambda n: TensorIndex(f'i{n}', 'l', bundle)
    factors = [IndexedTensor(f'T{n}', (iu(n), il(n + 1))) for n in range(6)]
    expr = factors[0]
    for factor in factors[1:]:
        expr = IndexedTensorExpr('tensor_product', (expr, factor))
    for _ in range(30):
        tensor_graph(expr)
        tensor_reduce(expr)
    return {"factor_count": len(factors)}


def main() -> None:
    case = run_case("contraction_planner", planner_workload, metadata={"factor_count": 6})
    print_report(case)


if __name__ == '__main__':
    main()
