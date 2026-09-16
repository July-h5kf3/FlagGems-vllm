# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

import flaggems_vllm

from .marlin_moe_hygon_reference import make_case, reference

pytestmark = pytest.mark.skipif(
    flaggems_vllm.vendor_name != "hygon", reason="Hygon backend coverage"
)


@pytest.mark.parametrize("q", [0, 1, 2, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape",
    [
        (0, 3, 128, 128, 2),
        (1, 4, 128, 256, 2),
        (9, 5, 256, 128, 3),
        (32, 4, 128, 256, 2),
    ],
)
@pytest.mark.parametrize("router_input", [False, True])
def test_hygon_marlin_correctness(q, dtype, shape, router_input):
    torch.manual_seed(752)
    args, w1, w2 = make_case(*shape, q=q, dtype=dtype)
    args["apply_router_weight_on_input"] = router_input
    expected = reference(args, w1, w2)
    actual = flaggems_vllm.fused_marlin_moe(**args)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("group", [-1, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hygon_marlin_fp8_groups(group, dtype):
    args, w1, w2 = make_case(m=11, q=2, group=group, dtype=dtype)
    args["w1"] = args["w1"].view(torch.float8_e4m3fn)
    args["w2"] = args["w2"].view(torch.float8_e4m3fn)
    actual = flaggems_vllm.fused_marlin_moe(**args)
    torch.testing.assert_close(actual, reference(args, w1, w2), atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("mode", ["output", "inplace", "strided", "skewed"])
def test_hygon_marlin_buffers(mode):
    args, w1, w2 = make_case(m=19)
    if mode == "skewed":
        args["topk_ids"].fill_(2)
    if mode == "strided":
        for key in ("w1", "w2", "w1_scale", "w2_scale"):
            x = args[key]
            args[key] = x.transpose(1, 2).contiguous().transpose(1, 2)
    expected = reference(args, w1, w2)
    if mode == "output":
        args["output"] = torch.empty_like(args["hidden_states"])
    if mode == "inplace":
        args["inplace"] = True
    result = flaggems_vllm.fused_marlin_moe(**args)
    torch.testing.assert_close(result, expected, atol=2e-3, rtol=2e-2)
    if mode in ("output", "inplace"):
        assert result is args["output" if mode == "output" else "hidden_states"]


@pytest.mark.parametrize(
    "change,exc",
    [
        ({"quant_type_id": 9}, NotImplementedError),
        ({"group_size": 64}, NotImplementedError),
        ({"activation": "relu"}, NotImplementedError),
        ({"is_k_full": False}, NotImplementedError),
        ({"global_num_experts": 9}, NotImplementedError),
    ],
)
def test_hygon_marlin_invalid(change, exc):
    args, _, _ = make_case()
    args.update(change)
    with pytest.raises(exc):
        flaggems_vllm.fused_marlin_moe(**args)


@pytest.mark.parametrize("q", [0, 1, 2, 6])
@pytest.mark.parametrize("inference", [False, True])
def test_hygon_marlin_weight_mutation(q, inference):
    with torch.inference_mode(inference):
        args, w1, w2 = make_case(m=2, q=q)
        before = flaggems_vllm.fused_marlin_moe(**args)
        # Same storage, different contents: cached layouts must be invalidated.
        args["w1"].zero_()
        if q == 0:
            w1.fill_(-8)
            w1.mul_(args["w1_scale"].repeat_interleave(128, -1))
        elif q == 1:
            w1.fill_(-128)
            w1.mul_(args["w1_scale"].repeat_interleave(128, -1))
        else:
            w1.zero_()
        after = flaggems_vllm.fused_marlin_moe(**args)
        torch.testing.assert_close(after, reference(args, w1, w2), atol=2e-3, rtol=2e-2)
        assert not torch.equal(before, after)


@pytest.mark.parametrize(
    "kind",
    ["output_shape", "output_alias", "grad", "activation_layout", "both_outputs"],
)
def test_hygon_marlin_validation(kind):
    args, _, _ = make_case()
    if kind == "output_shape":
        args["output"] = torch.empty((1, 128), device="cuda", dtype=torch.float16)
    elif kind == "output_alias":
        args["output"] = args["w1_scale"].view(-1)[:384].view(3, 128)
    elif kind == "grad":
        args["hidden_states"].requires_grad_()
    elif kind == "activation_layout":
        args["hidden_states"] = args["hidden_states"].T.contiguous().T
    else:
        args["output"] = torch.empty_like(args["hidden_states"])
        args["inplace"] = True
    with pytest.raises((ValueError, NotImplementedError)):
        flaggems_vllm.fused_marlin_moe(**args)


@pytest.mark.parametrize("q", [0, 2])
@pytest.mark.parametrize("split", [False, True])
def test_hygon_marlin_graph(q, split):
    args, w1, w2 = make_case(m=2 if split else 18, k=4096 if split else 128, q=q)
    expected = reference(args, w1, w2)
    flaggems_vllm.fused_marlin_moe(**args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = flaggems_vllm.fused_marlin_moe(**args)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("q", [2, 6])
def test_hygon_marlin_decode_special(q):
    import triton
    import triton.language as tl

    from flaggems_vllm.runtime.backend._hygon.fused.fused_marlin_moe import _decode

    @triton.jit
    def decode_kernel(W, S, Out, Q: tl.constexpr):
        k = tl.arange(0, 256)
        value = _decode(
            W,
            S,
            0,
            tl.full((256,), 0, tl.int32),
            k,
            1,
            256,
            256,
            256,
            1,
            8,
            8,
            1,
            Q,
            32,
            tl.float32,
        )
        tl.store(Out + k, value)

    codes = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
    if q == 2:
        w = codes
        scales = torch.ones(8, device="cuda")
        expected = codes.view(torch.float8_e4m3fn).float()
    else:
        codes = codes % 16
        w = codes[::2] | (codes[1::2] << 4)
        scales = torch.tensor(
            [0, 1, 120, 127, 128, 200, 254, 255], device="cuda", dtype=torch.uint8
        )
        lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
        values = lut[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
        sf = torch.exp2(scales.double() - 127).float()
        sf[-1] = float("nan")
        expected = values * sf.repeat_interleave(32)
    out = torch.empty(256, device="cuda", dtype=torch.float32)
    decode_kernel[(1,)](w, scales, out, q)
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)
    zero = expected == 0
    assert torch.equal(torch.signbit(out[zero]), torch.signbit(expected[zero]))


def test_hygon_marlin_fp8_fp32_scales():
    args, w1, w2 = make_case(m=2, q=2)
    args["w1_scale"] = args["w1_scale"].float()
    args["w2_scale"] = args["w2_scale"].float()
    out = flaggems_vllm.fused_marlin_moe(**args)
    torch.testing.assert_close(out, reference(args, w1, w2), atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("q", [2, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hygon_marlin_partial_tiles(q, dtype):
    args, w1, w2 = make_case(m=9, e=5, k=160, n=96, topk=3, q=q, dtype=dtype, group=32)
    args["topk_ids"] = args["topk_ids"].int()
    args["topk_weights"] = args["topk_weights"].to(dtype)
    actual = flaggems_vllm.fused_marlin_moe(**args)
    torch.testing.assert_close(actual, reference(args, w1, w2), atol=2e-3, rtol=2e-2)


def test_hygon_marlin_topk_one():
    args, w1, w2 = make_case(m=17, e=1, topk=1)
    actual = flaggems_vllm.fused_marlin_moe(**args)
    torch.testing.assert_close(actual, reference(args, w1, w2), atol=2e-3, rtol=2e-2)


def test_hygon_marlin_scale_mutation():
    args, w1, w2 = make_case(m=19)
    flaggems_vllm.fused_marlin_moe(**args)
    args["w1_scale"].mul_(2)
    w1.mul_(2)
    actual = flaggems_vllm.fused_marlin_moe(**args)
    torch.testing.assert_close(actual, reference(args, w1, w2), atol=2e-3, rtol=2e-2)


def test_hygon_marlin_stream_change():
    args, w1, w2 = make_case(m=19, q=6)
    flaggems_vllm.fused_marlin_moe(**args)
    expected = reference(args, w1, w2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = flaggems_vllm.fused_marlin_moe(**args)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("q", [0, 1, 2, 6])
@pytest.mark.parametrize("m", [1, 10])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("router_input", [False, True])
def test_hygon_marlin_split_k(q, m, dtype, router_input):
    torch.manual_seed(752)
    args, w1, w2 = make_case(m=m, e=4, k=4096, n=128, q=q, dtype=dtype)
    args["apply_router_weight_on_input"] = router_input
    actual = flaggems_vllm.fused_marlin_moe(**args)
    torch.testing.assert_close(actual, reference(args, w1, w2), atol=2e-3, rtol=2e-2)


def test_hygon_marlin_mxfp4_all_scales_bf16():
    import triton
    import triton.language as tl

    from flaggems_vllm.runtime.backend._hygon.fused.fused_marlin_moe import _decode

    @triton.jit
    def decode_all(W, S, Out):
        e = tl.program_id(0)
        k = tl.arange(0, 32)
        v = _decode(
            W,
            S,
            e,
            tl.full((32,), 0, tl.int32),
            k,
            1,
            32,
            16,
            16,
            1,
            1,
            1,
            1,
            6,
            32,
            tl.bfloat16,
        )
        tl.store(Out + e * 32 + k, v)

    codes = (torch.arange(32, device="cuda", dtype=torch.uint8) % 16).repeat(256, 1)
    weights = codes[:, ::2] | (codes[:, 1::2] << 4)
    scales = torch.arange(256, device="cuda", dtype=torch.int32).to(torch.uint8)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    values = lut[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
    sf = torch.exp2(scales.double() - 127).float()
    sf[-1] = float("nan")
    expected = (values * sf[:, None]).to(torch.bfloat16)
    out = torch.empty_like(expected)
    decode_all[(256,)](weights, scales, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)
    zero = expected == 0
    assert torch.equal(torch.signbit(out[zero]), torch.signbit(expected[zero]))


def test_hygon_marlin_fp16_subnormal_input():
    args, _, _ = make_case(m=1, e=1, topk=1, q=0)
    args["hidden_states"].zero_()
    args["hidden_states"][0, 0] = 2**-15
    args["hidden_states"][0, 1] = 1
    args["w1"].fill_(0x88)
    args["w1"][0, :128, 0] = 0x89
    args["w1"][0, 128:, 0] = 0x98
    args["w1_scale"].fill_(1)
    args["w1_scale"][0, 128:].fill_(128)
    args["w2"].fill_(0x99)
    args["w2_scale"].fill_(1)
    args["topk_weights"].fill_(1)
    actual = flaggems_vllm.fused_marlin_moe(**args)
    # gate=2^-15, up=128; the FP16 activation rounds to 2^-9.
    # The down projection adds 128 copies, yielding exactly 1/4.
    torch.testing.assert_close(actual, torch.full_like(actual, 0.25), rtol=0, atol=0)


@pytest.mark.parametrize("q", [0, 1])
def test_hygon_marlin_large_expert_stride(q):
    args, w1, w2 = make_case(m=9, e=2, q=q)
    args["topk_ids"] = torch.ones_like(args["topk_ids"], dtype=torch.int32)
    original = args["w1"]
    # Allocate a storage hole without initializing it; expert 1 starts past 2 GiB.
    wide = torch.empty_strided(
        original.shape,
        (2**31 + 256, original.stride(1), original.stride(2)),
        dtype=original.dtype,
        device=original.device,
    )
    wide.copy_(original)
    args["w1"] = wide
    actual = flaggems_vllm.fused_marlin_moe(**args)
    torch.testing.assert_close(actual, reference(args, w1, w2), atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hygon_marlin_mxfp4_cached_all_scales(dtype):
    import triton
    import triton.language as tl

    from flaggems_vllm.runtime.backend._hygon.fused.fused_marlin_moe import (
        _cached_transpose,
        _decode,
    )

    @triton.jit
    def decode_cached(W, S, Out, DTYPE: tl.constexpr):
        e = tl.program_id(0)
        k = tl.arange(0, 32)
        v = _decode(
            W,
            S,
            e,
            tl.full((32,), 0, tl.int32),
            k,
            1,
            32,
            32,
            1,
            1,
            1,
            1,
            1,
            7,
            32,
            DTYPE,
        )
        tl.store(Out + e * 32 + k, v)

    codes = (torch.arange(32, device="cuda", dtype=torch.uint8) % 16).repeat(256, 1)
    weights = (codes[:, ::2] | (codes[:, 1::2] << 4)).view(256, 1, 16)
    scales = (
        torch.arange(256, device="cuda", dtype=torch.int32)
        .to(torch.uint8)
        .view(256, 1, 1)
    )
    w = _cached_transpose(weights, 1)
    s = _cached_transpose(scales, 2)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    values = lut[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
    sf = torch.exp2(scales.flatten().double() - 127).float()
    sf[-1] = float("nan")
    expected = (values * sf[:, None]).to(dtype)
    out = torch.empty_like(expected)
    decode_cached[(256,)](
        w, s, out, tl.float16 if dtype == torch.float16 else tl.bfloat16
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)
    zero = expected == 0
    assert torch.equal(torch.signbit(out[zero]), torch.signbit(expected[zero]))


@pytest.mark.parametrize("q", [0, 1, 2, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("router_input", [False, True])
def test_hygon_marlin_dense_skewed_routes(q, dtype, router_input):
    torch.manual_seed(752)
    args, w1, w2 = make_case(m=65, e=4, k=128, n=128, q=q, dtype=dtype)
    # Several row tiles and a partial final tile for one expert; others empty.
    args["topk_ids"].zero_()
    args["apply_router_weight_on_input"] = router_input
    actual = flaggems_vllm.fused_marlin_moe(**args)
    torch.testing.assert_close(actual, reference(args, w1, w2), atol=2e-3, rtol=2e-2)
