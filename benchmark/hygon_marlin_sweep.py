# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Upstream-shape sweep against native vLLM BF16 with bounded setup memory."""

import argparse
import hashlib
import inspect
import json
import os
from functools import partial
from pathlib import Path
from statistics import median

import torch
import triton
import vllm
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

import flaggems_vllm
from benchmark.hygon_marlin_vllm import errors
from benchmark.hygon_vllm_reference import ensure_large_weight_addressing
from tests.marlin_moe_hygon_reference import make_case, reference

GEOMETRIES = [
    (8, 4096, 14336, 2),
    (256, 7168, 2048, 8),
    (512, 4096, 1024, 10),
    (256, 4096, 2048, 6),
]


def make_bank(m, e, k, n, topk, q, dtype=torch.bfloat16):
    """Allocate final tensors once; generate at most one expert's scratch."""
    result, ref1, ref2 = None, None, None
    for expert in range(e):
        one, first, second = make_case(m=0, e=1, k=k, n=n, topk=1, q=q, dtype=dtype)
        if result is None:
            result = dict(one)
            for name in ("w1", "w2", "w1_scale", "w2_scale"):
                tensor = one[name]
                result[name] = torch.empty(
                    (e, *tensor.shape[1:]), device=tensor.device, dtype=tensor.dtype
                )
            ref1 = torch.empty(
                (e, *first.shape[1:]), device=first.device, dtype=first.dtype
            )
            ref2 = torch.empty(
                (e, *second.shape[1:]), device=second.device, dtype=second.dtype
            )
        for name in ("w1", "w2", "w1_scale", "w2_scale"):
            result[name][expert : expert + 1].copy_(one[name])
        ref1[expert : expert + 1].copy_(first)
        ref2[expert : expert + 1].copy_(second)
        del one, first, second
    result["hidden_states"] = (torch.randn((m, k), device="cuda") * 0.2).to(dtype)
    routing = torch.randn((m, e), device="cuda")
    scores, ids = torch.topk(routing, topk, dim=-1)
    result["topk_ids"] = ids
    result["topk_weights"] = torch.softmax(scores, dim=-1)
    return result, ref1, ref2


def prepare_native_int8(args, ref1, ref2):
    """Shared channel-scale subset; keep temporary memory to one expert."""
    weights, scales = [], []
    for index, reference_weight in ((1, ref1), (2, ref2)):
        packed, grouped = args[f"w{index}"], args[f"w{index}_scale"]
        signed = torch.empty_like(packed, dtype=torch.int8)
        channel = grouped[..., 0].contiguous()
        for expert in range(packed.shape[0]):
            grouped[expert].copy_(channel[expert, :, None].expand_as(grouped[expert]))
            signed[expert].copy_((packed[expert].to(torch.int16) - 128).to(torch.int8))
            reference_weight[expert].copy_(
                (signed[expert].float() * channel[expert, :, None].float()).to(
                    reference_weight.dtype
                )
            )
        weights.append(signed)
        scales.append(channel)
    return weights, scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=int, choices=(0, 1, 2, 6), required=True)
    parser.add_argument("--geometry", type=int, choices=range(4), required=True)
    parser.add_argument("--tokens", default="1,4,8,16,32,64,128,256")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--bank-tokens", type=int, default=256)
    parser.add_argument("--baseline", choices=("bf16", "fp16", "int8"))
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    options = parser.parse_args()
    dtype = getattr(torch, options.dtype)
    if options.baseline is None:
        options.baseline = "bf16" if dtype == torch.bfloat16 else "fp16"
    if options.baseline != "int8":
        assert options.baseline == ("bf16" if dtype == torch.bfloat16 else "fp16")
    tokens = [int(value) for value in options.tokens.split(",")]
    assert options.repeat > 0 and all(value > 0 for value in tokens)
    e, k, n, topk = GEOMETRIES[options.geometry]
    source = Path(inspect.getfile(flaggems_vllm.fused_marlin_moe))
    print(
        "HYGON_ENV "
        + json.dumps(
            dict(
                torch=torch.__version__,
                triton=triton.__version__,
                vllm=vllm.__version__,
                device=torch.cuda.get_device_name(),
                visible_devices=os.environ.get("HIP_VISIBLE_DEVICES"),
                seed=752,
                implementation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                native_function=fused_experts.__module__ + ".fused_experts",
                baseline=options.baseline,
                bank_tokens=max(max(tokens), options.bank_tokens),
            )
        ),
        flush=True,
    )
    torch.manual_seed(752)
    bank, w1, w2 = make_bank(
        max(max(tokens), options.bank_tokens), e, k, n, topk, options.q, dtype=dtype
    )
    assert (
        bank["hidden_states"].dtype == dtype and w1.dtype == dtype and w2.dtype == dtype
    )
    native_weights, native_kwargs = (w1, w2), {}
    if options.baseline == "int8":
        assert options.q == 1, "INT8 baseline requires q=1"
        native_weights, native_scales = prepare_native_int8(bank, w1, w2)
        native_kwargs = dict(
            use_int8_w8a16=True, w1_scale=native_scales[0], w2_scale=native_scales[1]
        )
    address_metadata = ensure_large_weight_addressing(native_weights)
    print("HYGON_VLLM_ADDRESS " + json.dumps(address_metadata), flush=True)
    print(
        "HYGON_BANK_READY "
        + json.dumps(
            dict(
                q=options.q,
                geometry=options.geometry,
                allocated_bytes=torch.cuda.memory_allocated(),
            )
        ),
        flush=True,
    )
    for m in tokens:
        args = dict(bank)
        for name in ("hidden_states", "topk_ids", "topk_weights"):
            args[name] = bank[name][:m]
        run = partial(flaggems_vllm.fused_marlin_moe, **args)
        native = partial(
            fused_experts,
            args["hidden_states"],
            native_weights[0],
            native_weights[1],
            args["topk_weights"],
            args["topk_ids"],
            **native_kwargs,
        )
        active_experts = args["topk_ids"].unique().numel()
        expected = reference(args, w1, w2)
        op_error, vllm_error = errors(run(), expected), errors(native(), expected)
        for check in (op_error, vllm_error):
            assert check["relative_rms"] < 0.01 and check["relative_peak"] < 0.02, check
        samples = []
        for iteration in range(options.repeat):
            # Alternate timing order to reduce a fixed first-run advantage.
            if iteration % 2:
                op_ms = triton.testing.do_bench(run, warmup=100, rep=200)
                base_ms = triton.testing.do_bench(native, warmup=100, rep=200)
            else:
                base_ms = triton.testing.do_bench(native, warmup=100, rep=200)
                op_ms = triton.testing.do_bench(run, warmup=100, rep=200)
            samples.append(
                dict(vllm_ms=base_ms, operator_ms=op_ms, speedup=base_ms / op_ms)
            )
        print(
            "HYGON_SWEEP "
            + json.dumps(
                dict(
                    q=options.q,
                    baseline=options.baseline,
                    dtype=options.dtype,
                    shape=(m, e, k, n, topk),
                    active_experts=active_experts,
                    samples=samples,
                    median_speedup=median(row["speedup"] for row in samples),
                    operator_error=op_error,
                    vllm_error=vllm_error,
                    allocated_bytes=torch.cuda.memory_allocated(),
                )
            ),
            flush=True,
        )
        del args, run, native, expected


if __name__ == "__main__":
    main()
