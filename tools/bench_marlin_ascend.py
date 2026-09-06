"""Paired operator and NPUGraph screening, preserving every PR 741 trace row."""

import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "marlin_ascend_utils", ROOT / "benchmark/marlin_ascend_utils.py"
)
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="all")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--output", default="work/marlin-trace.json")
    args = ap.parse_args()
    selected = None if args.m == "all" else set(map(int, args.m.split(",")))
    trace = [(m, c) for m, c in u.PR5140_TRACE if selected is None or m in selected]
    ww = u.weights(256, 4096, 256, torch.bfloat16)
    rows = []
    for m, calls in trace:
        x, p, ids = u.inputs(m, 256, 4096, 6)
        candidate = lambda: u.gems_call(x, ww, p, ids)
        baseline = lambda: u.baseline(x, ww, p, ids)
        print("CASE", m, "compiling/validating", flush=True)
        ref = baseline()
        got = candidate()
        torch.npu.synchronize()
        torch.testing.assert_close(got, ref, rtol=0.02, atol=0.02)
        max_abs = float((got - ref).abs().max())
        for _ in range(5):
            baseline()
            candidate()
        torch.npu.synchronize()
        bg = torch.npu.NPUGraph()
        cg = torch.npu.NPUGraph()
        with torch.npu.graph(bg):
            baseline_output = baseline()
        with torch.npu.graph(cg):
            candidate_output = candidate()
        # Replays, including validation, exercise the same captured buffers.
        bg.replay()
        cg.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            candidate_output, baseline_output, rtol=0.02, atol=0.02
        )
        bu = []
        cu = []
        bgu = []
        cgu = []
        for j in range(args.pairs):
            if j % 2:
                cu.append(u.bench(candidate, args.iters))
                bu.append(u.bench(baseline, args.iters))
                cgu.append(u.bench(cg.replay, args.iters))
                bgu.append(u.bench(bg.replay, args.iters))
            else:
                bu.append(u.bench(baseline, args.iters))
                cu.append(u.bench(candidate, args.iters))
                bgu.append(u.bench(bg.replay, args.iters))
                cgu.append(u.bench(cg.replay, args.iters))
        bm, cm, bgm, cgm = map(statistics.median, (bu, cu, bgu, cgu))
        row = dict(
            m=m,
            calls=calls,
            baseline_us=bm,
            candidate_us=cm,
            speedup=bm / cm,
            baseline_graph_us=bgm,
            candidate_graph_us=cgm,
            graph_speedup=bgm / cgm,
            max_abs=max_abs,
            operator_pairs=[bu, cu],
            graph_pairs=[bgu, cgu],
        )
        rows.append(row)
        payload = dict(
            geometry=dict(
                E=256,
                K=4096,
                N=256,
                top_k=6,
                group_size=128,
                dtype="bfloat16",
                weights="uint4b8",
                router_dtype="float32",
            ),
            timing="NPU events; operator and explicit NPUGraph; profiler is separate",
            rows=rows,
        )
        payload["operator_weighted_speedup"] = sum(
            r["calls"] * r["baseline_us"] for r in rows
        ) / sum(r["calls"] * r["candidate_us"] for r in rows)
        payload["graph_weighted_speedup"] = sum(
            r["calls"] * r["baseline_graph_us"] for r in rows
        ) / sum(r["calls"] * r["candidate_graph_us"] for r in rows)
        payload["all_shapes_1_3x"] = len(rows) == 53 and all(
            r["graph_speedup"] >= 1.3 and r["speedup"] >= 1.3 for r in rows
        )
        dest = ROOT / args.output
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2))
        print(
            json.dumps({k: v for k, v in row.items() if not k.endswith("_pairs")}),
            flush=True,
        )
        del bg, cg, baseline_output, candidate_output
    print(
        "TRACE_DONE",
        len(rows),
        "weighted",
        payload["operator_weighted_speedup"],
        payload["graph_weighted_speedup"],
        flush=True,
    )


if __name__ == "__main__":
    main()
