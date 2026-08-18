from __future__ import annotations

import json
import time
from typing import Callable, Any


def run_case(name: str, fn: Callable[[], Any], *, repeat: int = 1, warmup: int = 0, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    for _ in range(max(warmup, 0)):
        fn()
    start = time.perf_counter()
    for _ in range(max(repeat, 1)):
        fn()
    elapsed = time.perf_counter() - start
    payload = {
        "name": name,
        "repeat": max(repeat, 1),
        "elapsed_s": round(elapsed, 6),
        "per_iter_s": round(elapsed / max(repeat, 1), 6),
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def print_report(*cases: dict[str, Any]) -> None:
    print(json.dumps({"benchmarks": list(cases)}, indent=2, sort_keys=True))
