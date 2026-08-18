from __future__ import annotations

import time

from tensoratlas.abstract_tensor import (
    abstract_to_indexed,
    canonical_tensor_expression,
    fully_symmetric_head,
    indexed_to_abstract,
    leaf,
)
from tensoratlas.contracts.bridges import check_bridge_contract
from tensoratlas.contracts.normal_forms import check_normal_form_contract


def _sample_tensor_expression():
    head = fully_symmetric_head("BenchT", 2)
    a = head.index_types[0].dummy_name
    b = head.index_types[1].dummy_name
    return leaf(head(a, b) + head(b, a))


def run() -> dict[str, object]:
    expr = _sample_tensor_expression()

    t0 = time.perf_counter()
    nf = check_normal_form_contract(
        expr,
        normalize=canonical_tensor_expression,
    )
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    bridge = check_bridge_contract(
        expr,
        forward=abstract_to_indexed,
        backward=indexed_to_abstract,
        canonical_key=lambda x: canonical_tensor_expression(x),
    )
    t3 = time.perf_counter()

    return {
        "normal_form_elapsed": t1 - t0,
        "bridge_elapsed": t3 - t2,
        "normal_form_idempotent": nf.idempotent,
        "normal_form_stable_under_equivalence": nf.stable_under_equivalence,
        "bridge_preserves_canonical_key": bridge.preserves_canonical_key,
    }


if __name__ == "__main__":
    print(run())
