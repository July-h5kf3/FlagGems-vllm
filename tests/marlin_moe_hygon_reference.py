# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Test/benchmark-only input generation and independent PyTorch oracle."""

import torch
import torch.nn.functional as F


def make_case(
    m=3, e=4, k=128, n=128, topk=2, q=0, dtype=torch.float16, group=None, device="cuda"
):
    if group is None:
        group = 32 if q == 6 else 128

    def weight(out, red):
        g = red if group == -1 else group
        if q in (0, 6):
            codes = torch.randint(
                0, 16, (e, out, red), device=device, dtype=torch.uint8
            )
            packed = codes[..., ::2] | (codes[..., 1::2] << 4)
            if q == 0:
                values = codes.float() - 8
            else:
                lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device)
                values = lut[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
        elif q == 1:
            packed = torch.randint(
                0, 256, (e, out, red), device=device, dtype=torch.uint8
            )
            values = packed.float() - 128
        else:
            # Include signed zero, subnormals and large finite E4M3FN values.
            packed = torch.randint(
                0, 254, (e, out, red), device=device, dtype=torch.uint8
            )
            packed = torch.where(packed == 127, 0, packed).to(torch.uint8)
            values = packed.view(torch.float8_e4m3fn).float()
        if q == 6:
            scales = torch.randint(
                121, 126, (e, out, red // g), device=device, dtype=torch.uint8
            )
            sf = torch.exp2(scales.float() - 127)
        else:
            factor = 0.02 if q == 0 else 0.001
            scales = (
                torch.rand((e, out, red // g), device=device) * factor + factor
            ).to(dtype)
            sf = scales.float()
        ref = (values * sf.repeat_interleave(g, -1)).to(dtype)
        return packed, scales, ref

    w1, s1, ref1 = weight(2 * n, k)
    w2, s2, ref2 = weight(k, n)
    a = (torch.randn((m, k), device=device) * 0.2).to(dtype)
    ids = torch.randint(0, e, (m, topk), device=device)
    tw = torch.softmax(torch.randn((m, topk), device=device), -1)
    args = dict(
        hidden_states=a,
        w1=w1,
        w2=w2,
        bias1=None,
        bias2=None,
        w1_scale=s1,
        w2_scale=s2,
        topk_weights=tw,
        topk_ids=ids,
        quant_type_id=q,
        group_size=group,
    )
    return args, ref1, ref2


def reference(args, w1, w2):
    a, ids, weights = (args[k] for k in ("hidden_states", "topk_ids", "topk_weights"))
    m, k = a.shape
    topk = ids.shape[1]
    first_weight = args.get("apply_router_weight_on_input", False)
    result = torch.zeros((m * topk, k), dtype=a.dtype, device=a.device)
    flat_ids = ids.flatten()
    for expert in range(w1.shape[0]):
        routes = torch.where(flat_ids == expert)[0]
        x = a[routes // topk].float()
        gateup = x @ w1[expert].float().T
        rw = weights.flatten()[routes, None]
        if first_weight:
            gateup = gateup * rw
        gate, up = gateup.to(a.dtype).float().chunk(2, -1)
        act = (F.silu(gate) * up).to(a.dtype)
        out = act.float() @ w2[expert].float().T
        if not first_weight:
            out = out * rw
        result[routes] = out.to(a.dtype)
    return result.view(m, topk, k).float().sum(1).to(a.dtype)
