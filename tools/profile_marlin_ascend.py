# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Profile all PR 741 shapes, or only M=16384 with --max-only."""
import csv
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch
import torch_npu

# Initialize backend adapters before entering profiler scopes.
import flaggems_vllm  # noqa: F401

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "u", root / "benchmark/marlin_ascend_utils.py"
)
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)

ww = u.weights(256, 4096, 256, torch.bfloat16)
cases = [
    (m, calls, *u.inputs(m, 256, 4096, 6))
    for m, calls in u.PR5140_TRACE
    if "--max-only" not in sys.argv or m == 16384
]
for m, calls, x, p, ids in cases:
    for _ in range(5):
        u.baseline(x, ww, p, ids)
        u.gems_call(x, ww, p, ids)
torch.npu.synchronize()
results = {}
for label, fn, launches in [("baseline", u.baseline, 5), ("candidate", u.gems_call, 8)]:
    out = root / "work" / ("marlin_profile/" + label)
    schedule = torch_npu.profiler.schedule(
        wait=0, warmup=0, active=len(cases), repeat=1
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=schedule,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out)),
    ) as prof:
        for m, calls, x, p, ids in cases:
            with torch.profiler.record_function("marlin_M_" + str(m)):
                for _ in range(5):
                    fn(x, ww, p, ids)
                torch.npu.synchronize()
            prof.step()
    csv_path = max(out.rglob("kernel_details.csv"), key=lambda p: p.stat().st_mtime)
    records = list(csv.DictReader(csv_path.open()))
    expected = sum(
        5 * (5 if label == "baseline" else 1 if m * 6 <= 64 else 8) for m, *_ in cases
    )
    if len(records) != expected:
        raise RuntimeError(
            f"{label}: expected {expected} launches, found {len(records)}"
        )
    rows = []
    cursor = 0
    for index, (m, calls, *_) in enumerate(cases):
        launches = 5 if label == "baseline" else 1 if m * 6 <= 64 else 8
        part = records[cursor : cursor + 5 * launches]
        cursor += 5 * launches
        timings = [
            sum(
                float(r["Duration(us)"])
                for r in part[i * launches : (i + 1) * launches]
            )
            for i in range(5)
        ]
        rows.append(
            dict(
                m=m,
                calls=calls,
                median_kernel_us=statistics.median(timings),
                samples_us=timings,
            )
        )
    results[label] = rows
    print("PROFILE_DONE", label, len(records), flush=True)
rows = []
for b, c in zip(results["baseline"], results["candidate"]):
    rows.append(
        dict(
            m=b["m"],
            calls=b["calls"],
            baseline_kernel_us=b["median_kernel_us"],
            candidate_kernel_us=c["median_kernel_us"],
            speedup=b["median_kernel_us"] / c["median_kernel_us"],
        )
    )
result = dict(
    method="torch_npu.profiler kernel_details.csv; sum of all kernel durations per invocation; median of 5",
    rows=rows,
)
result["weighted_speedup"] = sum(
    r["calls"] * r["baseline_kernel_us"] for r in rows
) / sum(r["calls"] * r["candidate_kernel_us"] for r in rows)
(root / "work/marlin-profiler.json").write_text(json.dumps(result, indent=2) + "\n")
print("PROFILER_WEIGHTED", result["weighted_speedup"])
