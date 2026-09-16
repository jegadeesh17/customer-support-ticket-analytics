"""Measure real single-row inference latency for the ML tier.

Loads each model bundle once per process (matching the `@lru_cache` warm-process
pattern already used in `src/inference.py`), then times N repeated single-row
predictions per model to produce real p50/p95/p99 numbers. This exists to check
the latency budget table in docs/SPEC.md (section 4) against measured reality,
since no benchmark code previously backed those numbers.

Usage:
    python scripts/benchmark_inference.py [--iterations 200] [--warmup 10]

Requires the trained model bundles to already exist under models/ (run
src/train_models.py first if they are missing).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.constants import DEFAULT_INFERENCE_ROW
from src.inference import predict_classification, predict_regression, predict_satisfaction

TASKS = {
    "classification": predict_classification,
    "regression": predict_regression,
    "satisfaction": predict_satisfaction,
}


def _percentile(samples_ms: list[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation), sorted ascending input."""
    if not samples_ms:
        return float("nan")
    k = max(0, min(len(samples_ms) - 1, int(round(pct / 100 * (len(samples_ms) - 1)))))
    return samples_ms[k]


def benchmark_task(name: str, fn, payload: dict, iterations: int, warmup: int) -> dict:
    # Warm-up: triggers model unpickling via lru_cache in src/inference.py so the
    # timed loop only measures steady-state (already-loaded) prediction latency,
    # simulating a warm long-lived process such as the deployed Cloud Run container.
    for _ in range(warmup):
        fn(payload)

    samples_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(payload)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    samples_ms.sort()
    return {
        "task": name,
        "n": iterations,
        "p50_ms": round(_percentile(samples_ms, 50), 3),
        "p95_ms": round(_percentile(samples_ms, 95), 3),
        "p99_ms": round(_percentile(samples_ms, 99), 3),
        "min_ms": round(samples_ms[0], 3),
        "max_ms": round(samples_ms[-1], 3),
        "mean_ms": round(statistics.mean(samples_ms), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200, help="Timed predictions per model.")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warm-up predictions per model.")
    args = parser.parse_args()

    payload = dict(DEFAULT_INFERENCE_ROW)

    results = []
    for name, fn in TASKS.items():
        try:
            results.append(benchmark_task(name, fn, payload, args.iterations, args.warmup))
        except FileNotFoundError as exc:
            print(f"[skip] {name}: {exc}")

    if not results:
        print("No models available to benchmark. Run src/train_models.py first.")
        return

    header = f"{'task':<15}{'n':>6}{'p50_ms':>10}{'p95_ms':>10}{'p99_ms':>10}{'min_ms':>10}{'max_ms':>10}{'mean_ms':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['task']:<15}{r['n']:>6}{r['p50_ms']:>10}{r['p95_ms']:>10}{r['p99_ms']:>10}"
            f"{r['min_ms']:>10}{r['max_ms']:>10}{r['mean_ms']:>10}"
        )


if __name__ == "__main__":
    main()
