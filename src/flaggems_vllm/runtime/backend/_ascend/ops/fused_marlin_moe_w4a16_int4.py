# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Forward W4A16 INT4 MoE for Ascend 910B."""

from typing import Any, Callable, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _offsets(Counts, Offsets, E: tl.constexpr, EP: tl.constexpr, BM: tl.constexpr):
    ei = tl.arange(0, EP)
    count = tl.load(Counts + ei, ei < E, other=0)
    padded = tl.cdiv(count, BM) * BM
    end = tl.cumsum(padded)
    tl.store(Offsets + ei, end - padded, ei < E)
    tl.store(Offsets + E, tl.sum(padded, 0))


@triton.jit
def _pack_tiles(
    X,
    Routes,
    Counts,
    Offsets,
    Experts,
    Inv,
    A,
    E: tl.constexpr,
    EP: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
    BM: tl.constexpr,
):
    tile = tl.program_id(0)
    ei = tl.arange(0, EP)
    ends = tl.load(Offsets + ei + 1, ei < E, other=2147483647)
    ex = tl.sum((tile * BM >= ends).to(tl.int32), 0)
    tl.store(Experts + tile, tl.where(ex < E, ex, -1))
    if ex < E:
        begin = tl.load(Offsets + ex)
        count = tl.load(Counts + ex)
        rs = tile * BM - begin + tl.arange(0, BM)
        route = tl.load(Routes + ex * R + rs, rs < count, other=0)
        tl.store(Inv + route, tile * BM + tl.arange(0, BM), rs < count)


@triton.jit
def _cast_ids(Input, Output, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(Input + i, i < TOTAL, other=0).to(tl.int32)
    tl.store(Output + i, v, i < TOTAL)


def _run(x, w1, w2, s1, s2, topk_weights, topk_ids):
    """Return routed SwiGLU MoE for symmetric uint4b8, group size 128."""
    if x.dtype != torch.bfloat16 or x.device.type != "npu":
        raise NotImplementedError("The Ascend implementation supports BF16 activations")
    tensors = (x, w1, w2, s1, s2, topk_weights, topk_ids)
    if any((not a.is_contiguous() or a.device != x.device for a in tensors)):
        raise NotImplementedError("All inputs must be contiguous on the same NPU")
    if x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("Expected rank-2 activations and rank-3 weights")
    m, k = x.shape
    e, n2, kp = w1.shape
    n = n2 // 2
    if min(e, k, n) <= 0 or e > 256 or k > 4096 or n > 4096 or e * 2 * n * k >= 2**31:
        raise NotImplementedError(
            "Geometry exceeds the tested Ascend indexing and UB limits"
        )
    if k % 128 or n % 128 or n2 % 2 or (kp != k // 2) or (w2.shape != (e, k, n // 2)):
        raise ValueError("Invalid packed INT4 weight geometry")
    if w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise NotImplementedError("Weights must contain uint8 nibble pairs")
    if (
        s1.shape != (e, 2 * n, k // 128)
        or s2.shape != (e, k, n // 128)
        or s1.dtype != x.dtype
        or (s2.dtype != x.dtype)
    ):
        raise ValueError("Invalid group128 scale shape or dtype")
    if (
        topk_ids.ndim != 2
        or topk_ids.shape[0] != m
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError("Invalid routing shape")
    if (
        topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_weights.dtype != torch.float32
    ):
        raise ValueError("Routing requires integer IDs and FP32 weights")
    t = topk_ids.shape[1]
    if t < 1:
        raise ValueError("top_k must be positive")
    if x.requires_grad:
        raise NotImplementedError("Forward inference only")
    if m == 0:
        return torch.empty((m, k), device=x.device, dtype=x.dtype)
    if m * t <= 64:
        from flaggems_vllm.runtime.backend._ascend.ops.marlin_w4a16.small import (
            run as small_moe,
        )

        return small_moe(x, w1, w2, s1, s2, topk_weights, topk_ids)
    out = torch.empty((m, k), device=x.device, dtype=x.dtype)
    from flaggems_vllm.runtime.backend._ascend.ops.marlin_w4a16.custom_mixed import (
        gemm as custom_mixed,
    )
    from flaggems_vllm.runtime.backend._ascend.ops.marlin_w4a16.custom_routes import (
        routes as custom_routes,
    )
    from flaggems_vllm.runtime.backend._ascend.ops.marlin_w4a16.vector_stages import (
        combine as combine,
    )
    from flaggems_vllm.runtime.backend._ascend.ops.marlin_w4a16.vector_stages import (
        pack as pack,
    )
    from flaggems_vllm.runtime.backend._ascend.ops.marlin_w4a16.vector_stages import (
        silu as silu,
    )

    r = m * t
    routes = torch.empty((e, r), device=x.device, dtype=torch.int32)
    counts = torch.empty((e,), device=x.device, dtype=torch.int32)
    custom_routes(topk_ids, routes, counts)
    bm, _ = (128 if m >= 8192 else 64 if m >= 1024 else 32 if m > 32 else 16, 64)
    padded = triton.cdiv(r + e * (bm - 1), bm) * bm
    if max(e * r, padded * k, padded * 2 * n) >= 2**31 or r >= 2**24:
        raise NotImplementedError(
            "Geometry exceeds 32-bit addressing or exact routing-index limits"
        )
    offsets = torch.empty((e + 1,), device=x.device, dtype=torch.int32)
    experts = torch.empty((padded // bm,), device=x.device, dtype=torch.int32)
    inv = torch.empty((r,), device=x.device, dtype=torch.int32)
    packed_x = torch.empty((padded, k), device=x.device, dtype=x.dtype)
    h = torch.empty((padded, 2 * n), device=x.device, dtype=x.dtype)
    a = torch.empty((padded, n), device=x.device, dtype=x.dtype)
    z = torch.empty((padded, k), device=x.device, dtype=x.dtype)
    _offsets[1,](counts, offsets, e, triton.next_power_of_2(e), bm)
    _pack_tiles[padded // bm,](
        x,
        routes,
        counts,
        offsets,
        experts,
        inv,
        packed_x,
        e,
        triton.next_power_of_2(e),
        r,
        k,
        t,
        bm,
    )
    pack(x, routes, counts, offsets, experts, packed_x, bm, t)
    custom_mixed(packed_x, w1, s1, experts, h, bm, 256 if m <= 32 else 128)
    silu(h, a, offsets)
    custom_mixed(a, w2, s2, experts, z, bm, 256 if m <= 32 else 128)
    combine(z, topk_weights, out, inv)
    return out


def fused_marlin_moe_w4a16_int4(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias1: Optional[torch.Tensor],
    bias2: Optional[torch.Tensor],
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    apply_router_weight_on_input: bool = False,
    global_num_experts: int = -1,
    activation: Any = None,
    activation_func: Optional[Callable] = None,
    moe_sum: Optional[Callable] = None,
    expert_map: Optional[torch.Tensor] = None,
    input_global_scale1: Optional[torch.Tensor] = None,
    input_global_scale2: Optional[torch.Tensor] = None,
    global_scale1: Optional[torch.Tensor] = None,
    global_scale2: Optional[torch.Tensor] = None,
    g_idx1: Optional[torch.Tensor] = None,
    g_idx2: Optional[torch.Tensor] = None,
    sort_indices1: Optional[torch.Tensor] = None,
    sort_indices2: Optional[torch.Tensor] = None,
    w1_zeros: Optional[torch.Tensor] = None,
    w2_zeros: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    intermediate_cache13: Optional[torch.Tensor] = None,
    intermediate_cache2: Optional[torch.Tensor] = None,
    is_k_full: bool = True,
    output: Optional[torch.Tensor] = None,
    input_dtype: Optional[torch.dtype] = None,
    inplace: bool = False,
    clamp_limit: Optional[float] = None,
    group_size: int = 128,
) -> torch.Tensor:
    """Ascend BF16/uint4b8 specialization; unsupported options raise explicitly."""
    from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT4B8

    if quant_type_id != QUANT_TYPE_UINT4B8 or group_size != 128:
        raise NotImplementedError(
            "Only symmetric uint4b8 with group_size=128 is supported"
        )
    if (
        activation not in (None, "silu")
        or apply_router_weight_on_input
        or inplace
        or not is_k_full
    ):
        raise NotImplementedError(
            "Only out-of-place SiLU with output router weights is supported"
        )
    options = (
        bias1,
        bias2,
        activation_func,
        moe_sum,
        expert_map,
        input_global_scale1,
        input_global_scale2,
        global_scale1,
        global_scale2,
        g_idx1,
        g_idx2,
        sort_indices1,
        sort_indices2,
        w1_zeros,
        w2_zeros,
        workspace,
        intermediate_cache13,
        intermediate_cache2,
        output,
        input_dtype,
        clamp_limit,
    )
    if any(v is not None for v in options):
        raise NotImplementedError(
            "Bias, extra quantization metadata, callbacks and caller-owned workspaces are unsupported"
        )
    if global_num_experts not in (-1, w1.shape[0]):
        raise NotImplementedError("Expert parallel mappings are unsupported")
    if hidden_states.device.type != "npu" or hidden_states.dtype != torch.bfloat16:
        raise NotImplementedError("Ascend BF16 activations are required")
    if topk_ids.device != hidden_states.device or not topk_ids.is_contiguous():
        raise NotImplementedError("Routing IDs must be contiguous on the input NPU")
    if topk_ids.dtype == torch.int64 and topk_ids.numel() > 0:
        ids32 = torch.empty(topk_ids.shape, device=topk_ids.device, dtype=torch.int32)
        _cast_ids[(triton.cdiv(topk_ids.numel(), 1024),)](
            topk_ids, ids32, topk_ids.numel(), 1024
        )
        topk_ids = ids32
    return _run(hidden_states, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids)
