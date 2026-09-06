# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Test-only references for the vLLM-Ascend W4A16 primitive call chain."""

import torch
import torch_npu

PR5140_TRACE = (
    (1, 172),
    (2, 172),
    (4, 172),
    (8, 172),
    (16, 172),
    (24, 172),
    (32, 172),
    (40, 172),
    (48, 172),
    (56, 172),
    (64, 172),
    (72, 172),
    (80, 172),
    (88, 172),
    (96, 172),
    (104, 172),
    (112, 172),
    (120, 172),
    (128, 172),
    (136, 172),
    (144, 172),
    (152, 172),
    (160, 172),
    (168, 172),
    (176, 172),
    (184, 172),
    (192, 172),
    (200, 172),
    (208, 172),
    (216, 172),
    (224, 172),
    (232, 172),
    (240, 172),
    (248, 172),
    (256, 172),
    (272, 172),
    (288, 172),
    (304, 172),
    (320, 172),
    (336, 172),
    (352, 172),
    (368, 172),
    (384, 172),
    (400, 172),
    (416, 172),
    (432, 172),
    (448, 172),
    (464, 172),
    (480, 172),
    (496, 344),
    (512, 344),
    (2048, 43),
    (16384, 946),
)
MODEL_GEOMETRY = (256, 4096, 256, 6)


def weights(e, k, n, dtype):
    torch.manual_seed(7)
    result = []
    for ni, ki in [(2 * n, k), (k, n)]:
        w = torch.randint(0, 256, (e, ni, ki // 2), device="npu", dtype=torch.uint8)
        s = torch.rand((e, ni, ki // 128), device="npu", dtype=dtype) * 0.03
        native = []
        for ei in range(e):
            q = w[ei].to(torch.int32)
            q = torch.stack((q & 15, q >> 4), dim=-1).reshape(ni, ki) - 8
            native.append(torch_npu.npu_convert_weight_to_int4pack(q.T.contiguous()))
        wp = torch.stack(native)
        sn = s.transpose(1, 2).contiguous()
        result.append((w, s, wp, sn, torch.zeros_like(sn)))
    return result


def baseline(x, ww, p, ids):
    e = ww[0][0].shape[0]
    a, idx, counts, _ = torch_npu.npu_moe_init_routing_v2(
        x,
        ids.to(torch.int32),
        expert_num=e,
        active_num=x.shape[0] * ids.shape[1],
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        row_idx_type=0,
    )
    for j in range(2):
        _, _, w, s, z = ww[j]
        a = torch_npu.npu_grouped_matmul(
            x=[a],
            weight=[w],
            antiquant_scale=[s],
            antiquant_offset=[z],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=counts,
            output_dtype=x.dtype,
        )[0]
        if j == 0:
            a = torch_npu.npu_swiglu(a)
    return torch_npu.npu_moe_token_unpermute(a, idx, probs=p)


def reference(x, ww, p, ids):
    x = x.cpu()
    p = p.cpu()
    ids = ids.cpu()
    y = torch.zeros_like(x, dtype=torch.float32)
    decoded = []
    for w, s, *_ in ww:
        w = w.cpu().to(torch.int32)
        s = s.cpu()
        q = (
            torch.stack((w & 15, w >> 4), dim=-1).reshape(
                *w.shape[:-1], w.shape[-1] * 2
            )
            - 8
        )
        decoded.append(
            (q.float() * s.float().repeat_interleave(128, dim=-1)).to(x.dtype).float()
        )
    for m in range(x.shape[0]):
        for t in range(ids.shape[1]):
            e = int(ids[m, t])
            h = (x[m].float() @ decoded[0][e].T).to(x.dtype).float()
            a, b = h.chunk(2)
            a = (torch.nn.functional.silu(a) * b).to(x.dtype).float()
            z = (a @ decoded[1][e].T).to(x.dtype).float()
            y[m] += z * p[m, t]
    return y.to(x.dtype)


def bench(fn, iters):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    a = torch.npu.Event(enable_timing=True)
    b = torch.npu.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / iters


def graph_bench(fn, iters):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        _ = fn()
    return bench(graph.replay, iters)


def gems_call(x, ww, p, ids):
    import flaggems_vllm
    from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT4B8

    return flaggems_vllm.fused_marlin_moe_w4a16_int4(
        x,
        ww[0][0],
        ww[1][0],
        None,
        None,
        ww[0][1],
        ww[1][1],
        p,
        ids,
        QUANT_TYPE_UINT4B8,
    )


def inputs(m, e, k, t, seed=7):
    torch.manual_seed(seed + m)
    x = torch.randn((m, k), device="npu", dtype=torch.bfloat16) * 0.1
    ids = torch.rand((m, e), device="npu").topk(t, -1).indices.to(torch.int32)
    p = torch.softmax(torch.randn((m, t), device="npu"), -1)
    return x, p, ids
