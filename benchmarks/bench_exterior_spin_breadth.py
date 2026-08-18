from __future__ import annotations

import time

from tensoratlas import clifford_algebra, gamma_generators, clifford_reduce


def main() -> None:
    cl = clifford_algebra(5, (5, 0, 0), basis_labels=("1", "2", "3", "4", "5"))
    gens = gamma_generators(cl)
    expr = sum(gens[i] * gens[i] for i in range(5))
    expr += gens[4] * gens[3] * gens[2] * gens[1] * gens[0]
    start = time.perf_counter()
    out = clifford_reduce(expr, cl)
    elapsed = time.perf_counter() - start
    print({"elapsed_seconds": elapsed, "result": str(out)})


if __name__ == "__main__":
    main()
