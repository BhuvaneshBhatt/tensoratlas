from __future__ import annotations

import time
import sympy as sp

from tensoratlas import coordinate_chart, tensor_from_components, tensor_roundtrip_structured, tensor_interop_report


def main() -> None:
    chart = coordinate_chart("Euclidean", "Cartesian", 3)
    x, y, z = chart.symbols()
    tensor = tensor_from_components(
        chart,
        [
            [[x, 0, y], [0, z, 0], [y, 0, x + z]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[z, y, x], [y, x, z], [x, z, y]],
        ],
        "uuu",
        symmetry_metadata={"symmetric": ((0, 2),)},
        domain_metadata={"benchmark": True},
    )
    start = time.perf_counter()
    for _ in range(25):
        tensor_roundtrip_structured(tensor)
        tensor_interop_report(tensor)
    elapsed = time.perf_counter() - start
    print({"interop_roundtrip_s": round(elapsed, 4)})


if __name__ == "__main__":
    main()
