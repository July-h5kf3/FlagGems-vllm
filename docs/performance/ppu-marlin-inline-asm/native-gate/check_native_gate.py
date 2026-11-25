"""Check paired native Marlin measurements for the recorded PPU workload."""

import argparse
import json
import math
import statistics
from pathlib import Path


def checked_rows(paths, candidate_sha, shapes):
    by_precision = {}
    hashes = {}
    for path in paths:
        run = json.loads(path.read_text())
        if not run["completed"] or run["rounds"] != 3 or run["rep_ms"] != 100:
            raise ValueError(f"Incomplete or incompatible measurement in {path.name}")
        for precision, group in run["results"].items():
            native = group["ready"]["vendor"]
            candidate = group["ready"]["candidate"]
            weight_hash = native["weights_sha256"]
            if not (
                native["uuid"] == candidate["uuid"] == run["uuid"].removeprefix("GPU-")
                and weight_hash == candidate["weights_sha256"]
                and hashes.setdefault(precision, weight_hash) == weight_hash
                and candidate["source_sha256"] == candidate_sha
                and candidate["backend"]
                == "flaggems_vllm.runtime.backend._thead.fused.fused_marlin_moe"
                and "PPUDeepGemmExperts" in native["backend"]
                and (
                    native["backend"].endswith(".PPUDeepGemmExperts")
                    if precision in ("fp8", "int8")
                    else run["native_w4"]
                    and native["backend"].endswith(precision + "-W4A16")
                )
            ):
                raise ValueError(
                    f"Device, source, weights or backend mismatch in {path.name}"
                )
            rows = by_precision.setdefault(precision, {})
            for row in group["rows"]:
                shape = tuple(row["shape"])
                if shapes.get(shape) != row["count"] or shape in rows:
                    raise ValueError(
                        f"Unexpected count or duplicate: {precision} {shape}"
                    )
                medians = {}
                for side in ("vendor", "candidate"):
                    trials = row["trials_ms"][side]
                    if len(trials) != 3 or not all(
                        math.isfinite(t) and t > 0 for t in trials
                    ):
                        raise ValueError(f"Invalid trials for {side} at {shape}")
                    medians[side] = statistics.median(trials)
                valid = all(
                    0 <= rms < 0.01 and 0 <= peak < 0.02
                    for rms, peak in (row["errors"], row["vendor_oracle_errors"])
                )
                ratio = medians["vendor"] / medians["candidate"] if valid else None
                if valid != row["numeric_passed"] or ratio != row["speedup"]:
                    raise ValueError(f"Accuracy or speedup inconsistent at {shape}")
                rows[shape] = row
    return by_precision, hashes


def verify(paths, retries, shapes, candidate_sha, required):
    groups, hashes = checked_rows(paths, candidate_sha, shapes)
    retried, retry_hashes = checked_rows(retries, candidate_sha, shapes)
    for precision, rows in retried.items():
        if retry_hashes[precision] != hashes.get(precision):
            raise ValueError("Retry used different weights")
        for shape, replacement in rows.items():
            previous = groups[precision][shape]
            if (
                previous["numeric_passed"]
                or previous["input_sha256"] != replacement["input_sha256"]
            ):
                raise ValueError(f"Retry must match an invalid original row at {shape}")
            groups[precision][shape] = replacement
    if not set(required).issubset(groups):
        raise ValueError("Missing a required precision")
    total = sum(shapes.values())
    results = {}
    for precision, rows in groups.items():
        if set(rows) != set(shapes) or not all(
            r["numeric_passed"] for r in rows.values()
        ):
            raise ValueError(f"Incomplete valid coverage for {precision}")
        weighted = (
            sum(rows[s]["speedup"] * count for s, count in shapes.items()) / total
        )
        results[precision] = weighted
    return results, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--retry", type=Path, action="append", default=[])
    parser.add_argument("--require", action="append", choices=["fp8", "int8"])
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    shapes = {
        tuple(s["shape"]): s["count"] for s in json.loads(args.shapes.read_text())
    }
    if (
        len(shapes) != 53
        or sum(shapes.values()) != 10105
        or any(
            shape[1:] != (256, 4096, 256, 6) or count < 1
            for shape, count in shapes.items()
        )
    ):
        raise ValueError("Expected the complete PPU Marlin 53-shape workload")
    required = args.require or ["fp8", "int8"]
    results, _ = verify(args.runs, args.retry, shapes, args.candidate_sha256, required)
    for precision, ratio in results.items():
        print(
            f"{precision}: {len(shapes)} shapes, {sum(shapes.values())} calls, {ratio:.6f}x"
        )
    return int(any(results[p] < 1.3 for p in required))


if __name__ == "__main__":
    raise SystemExit(main())
