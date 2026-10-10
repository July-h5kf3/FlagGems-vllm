# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import math

import pytest
import torch

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close, gems_assert_equal

DESCALE_BLOCK = 128

pytestmark = [
    pytest.mark.flash_attn_varlen_func_w8a8_int8,
    pytest.mark.skipif(
        flaggems_vllm.vendor_name not in ("hygon", "thead", "ascend"),
        reason="Hygon/PPU/Ascend-only API",
    ),
]


def _inputs(lengths, heads, dim, broadcast_scales=False, scale_length=None):
    torch.manual_seed(sum(lengths) + heads + dim)
    quant = torch.randint(
        -127,
        128,
        (sum(lengths), heads, dim),
        device=flaggems_vllm.device,
        dtype=torch.int8,
    )
    max_length = max(lengths) if scale_length is None else scale_length
    num_scale_blocks = -(-max_length // DESCALE_BLOCK)
    scales = (
        torch.rand((len(lengths), heads, num_scale_blocks), device=flaggems_vllm.device)
        * 0.015
        + 0.002
    )
    if broadcast_scales:
        scales = scales[:, :, :1].expand_as(scales)
    ref = torch.empty(quant.shape, device=flaggems_vllm.device, dtype=torch.float32)
    offset = 0
    for b, length in enumerate(lengths):
        for start in range(0, length, DESCALE_BLOCK):
            end = min(start + DESCALE_BLOCK, length)
            ref[offset + start : offset + end] = (
                quant[offset + start : offset + end].float()
                * scales[b, :, start // DESCALE_BLOCK, None]
            )
        offset += length
    cu = torch.tensor(
        [0] + list(itertools.accumulate(lengths)),
        device=flaggems_vllm.device,
        dtype=torch.int32,
    )
    return quant, scales, ref, cu


def _reference(q, k, v, qlens, klens, causal, window=(-1, -1), cap=0, alibi=None):
    outputs, lses = [], []
    q_offset = k_offset = 0
    for b, (nq, nk) in enumerate(zip(qlens, klens)):
        qi = q[q_offset : q_offset + nq].transpose(0, 1)
        ki = (
            k[k_offset : k_offset + nk]
            .transpose(0, 1)
            .repeat_interleave(q.shape[1] // k.shape[1], 0)
        )
        vi = (
            v[k_offset : k_offset + nk]
            .transpose(0, 1)
            .repeat_interleave(q.shape[1] // v.shape[1], 0)
        )
        scores = qi @ ki.transpose(1, 2) * q.shape[-1] ** -0.5
        if cap:
            scores = cap * (scores / cap).tanh()
        m = torch.arange(nq, device=q.device) + nk - nq
        n = torch.arange(nk, device=q.device)
        if alibi is not None:
            slope = alibi if alibi.ndim == 1 else alibi[b]
            scores -= slope[:, None, None] * (m[:, None] - n).abs()
        mask = torch.ones((nq, nk), device=q.device, dtype=torch.bool)
        if causal:
            mask &= n <= m[:, None]
        if window[0] >= 0:
            mask &= n >= m[:, None] - window[0]
        if window[1] >= 0:
            mask &= n <= m[:, None] + window[1]
        scores = scores.masked_fill(~mask, -torch.inf)
        outputs.append((scores.softmax(-1).nan_to_num() @ vi).transpose(0, 1))
        lse = scores.logsumexp(-1)
        lses.append(lse.masked_fill(~mask.any(-1), torch.inf))
        q_offset += nq
        k_offset += nk
    return torch.cat(outputs), torch.cat(lses, dim=1)


def _run_case(
    qlens,
    klens,
    dim=64,
    heads=4,
    kvheads=4,
    causal=False,
    dtype=torch.bfloat16,
    paged=False,
    strided=False,
    window=(-1, -1),
    cap=0,
    alibi=None,
    broadcast_scales=False,
    extra_cache_pages=0,
    max_query_bound=None,
):
    q, qs, qr, cuq = _inputs(qlens, heads, dim, broadcast_scales, max_query_bound)
    k, ks, kr, cuk = _inputs(klens, kvheads, dim, broadcast_scales)
    v, vs, vr, _ = _inputs(klens, kvheads, dim, broadcast_scales)
    # Use different V data/scales to detect accidentally reusing K or its scale.
    v = -v
    vs = (vs[:, :, :1] * 1.7).expand_as(vs) if broadcast_scales else vs * 1.7
    vr = -vr * 1.7
    kwargs = dict(cu_seqlens_k=cuk)
    if paged:
        page_size = 16
        pages = (max(klens) + page_size - 1) // page_size
        table = (
            torch.randperm(len(klens) * pages, device=flaggems_vllm.device)
            .to(torch.int32)
            .reshape(len(klens), pages)
        )
        kc = torch.empty(
            (table.numel() + extra_cache_pages, page_size, kvheads, dim),
            device=flaggems_vllm.device,
            dtype=torch.int8,
        )
        vc = torch.empty_like(kc)
        if extra_cache_pages:
            kc.fill_(127)
            vc.fill_(-127)
        offset = 0
        for b, length in enumerate(klens):
            for start in range(0, length, page_size):
                count = min(page_size, length - start)
                kc[table[b, start // page_size].long(), :count] = k[
                    offset + start : offset + start + count
                ]
                vc[table[b, start // page_size].long(), :count] = v[
                    offset + start : offset + start + count
                ]
            offset += length
        if extra_cache_pages:
            for b, length in enumerate(klens):
                # Entries outside seqused_k are not valid cache addresses.
                table[b, -(-length // page_size) :] = kc.shape[0] + 1
        k, v = kc, vc
        kwargs = dict(
            seqused_k=torch.tensor(
                klens, device=flaggems_vllm.device, dtype=torch.int32
            ),
            block_table=table,
        )
    out = torch.empty(q.shape, device=flaggems_vllm.device, dtype=dtype)
    if strided:

        def _padded(x):
            storage = torch.empty(
                (*x.shape[:-2], x.shape[-2] * 2, x.shape[-1]),
                device=x.device,
                dtype=x.dtype,
            )
            result = storage[..., ::2, :]
            result.copy_(x)
            return result

        q, k, v, out = [_padded(x) for x in (q, k, v, out)]
        qs, ks, vs = [
            x.transpose(1, 2).contiguous().transpose(1, 2) for x in (qs, ks, vs)
        ]
    result, lse = flaggems_vllm.flash_attn_varlen_func(
        q,
        k,
        v,
        max(qlens) if max_query_bound is None else max_query_bound,
        cuq,
        max(klens),
        **kwargs,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        causal=causal,
        return_softmax_lse=True,
        out=out,
        window_size=window,
        softcap=cap,
        alibi_slopes=alibi,
    )
    ref, ref_lse = _reference(qr, kr, vr, qlens, klens, causal, window, cap, alibi)
    gems_assert_close(
        result.float(), ref, dtype=(result.float()).dtype, atol=0.025, rtol=0.025
    )
    gems_assert_close(lse, ref_lse, dtype=(lse).dtype, atol=2e-05, rtol=2e-05)
    output = flaggems_vllm.flash_attn_varlen_func(
        q,
        k,
        v,
        max(qlens) if max_query_bound is None else max_query_bound,
        cuq,
        max(klens),
        **kwargs,
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
        causal=causal,
        out=out,
        window_size=window,
        softcap=cap,
        alibi_slopes=alibi,
    )
    gems_assert_close(
        output.float(), ref, dtype=output.float().dtype, atol=0.025, rtol=0.025
    )
    return qr, kr, vr, cuq, cuk, result


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "qlens,klens",
    [([1], [1]), ([17, 129], [33, 257]), ([131, 0, 3], [1, 0, 0]), ([128], [256])],
)
def test_packed(dim, dtype, causal, qlens, klens):
    _run_case(qlens, klens, dim=dim, dtype=dtype, causal=causal)


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_paged_gqa(dim, causal):
    _run_case([17, 129], [145, 257], dim=dim, kvheads=2, causal=causal, paged=True)


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("paged", [False, True])
def test_strided(paged):
    _run_case([17, 129], [145, 257], kvheads=1, paged=paged, strided=True)


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize(
    "window,cap,alibi",
    [
        ((31, 0), 0, None),
        ((-1, 17), 3, None),
        ((32, 9), 0, "head"),
        ((-1, -1), 4, "batch"),
    ],
)
def test_score_modifiers(window, cap, alibi):
    slopes = (
        None
        if alibi is None
        else torch.rand(
            (4,) if alibi == "head" else (2, 4), device=flaggems_vllm.device
        )
        * 0.1
    )
    _run_case([17, 129], [145, 257], window=window, cap=cap, alibi=slopes)


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.skipif(
    flaggems_vllm.vendor_name == "ascend",
    reason="The existing generic FP16/BF16 path is not numerically validated on Ascend",
)
def test_bf16_baseline():
    q, k, v, cuq, cuk, result = _run_case([17, 129], [33, 257], causal=True)
    baseline = flaggems_vllm.flash_attn_varlen_func(
        q.bfloat16(), k.bfloat16(), v.bfloat16(), 129, cuq, 257, cuk, causal=True
    )
    gems_assert_close(result, baseline, dtype=(result).dtype, atol=0.03, rtol=0.03)


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("zero_scale", [False, True])
def test_default_output_and_broadcast_scales(zero_scale):
    q = torch.full((3, 2, 64), -128, device=flaggems_vllm.device, dtype=torch.int8)
    k = torch.full((129, 2, 64), 127, device=flaggems_vllm.device, dtype=torch.int8)
    v = -k
    cuq = torch.tensor([0, 3], device=flaggems_vllm.device, dtype=torch.int32)
    cuk = torch.tensor([0, 129], device=flaggems_vllm.device, dtype=torch.int32)
    scale = torch.full(
        (1, 2, 1), 0.0 if zero_scale else 0.01, device=flaggems_vllm.device
    )
    result = flaggems_vllm.flash_attn_varlen_func_w8a8_int8(
        q,
        k,
        v,
        3,
        cuq,
        129,
        cuk,
        q_descale=scale,
        k_descale=scale.expand(1, 2, 2),
        v_descale=scale.expand(1, 2, 2),
    )
    expected = torch.full_like(result, 0.0 if zero_scale else -1.27)
    gems_assert_equal(result, expected)


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("entry", ["top_level", "ops"])
@pytest.mark.skipif(
    flaggems_vllm.vendor_name == "ascend",
    reason="The existing generic FP16/BF16 path is not numerically validated on Ascend",
)
def test_public_float_route(dtype, entry):
    q, _, _, cuq = _inputs([17, 129], 4, 64)
    k, _, _, cuk = _inputs([33, 257], 4, 64)
    q, k = q.to(dtype) * 0.01, k.to(dtype) * 0.01
    v = -k
    op = (
        flaggems_vllm if entry == "top_level" else flaggems_vllm.ops
    ).flash_attn_varlen_func
    out = torch.empty_like(q)
    actual, lse = op(
        q, k, v, 129, cuq, 257, cuk, causal=True, out=out, return_softmax_lse=True
    )
    expected, expected_lse = _reference(
        q.float(), k.float(), v.float(), [17, 129], [33, 257], True
    )
    gems_assert_close(
        actual.float(), expected, dtype=(actual.float()).dtype, atol=0.01, rtol=0.01
    )
    gems_assert_close(lse, expected_lse, dtype=(lse).dtype, atol=2e-05, rtol=2e-05)


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("qlens,klens", [([512], [1024]), ([513, 1025], [257, 2049])])
def test_long_sequences(dim, causal, qlens, klens):
    _run_case(qlens, klens, dim=dim, causal=causal)


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize(
    "seed,query_length,kv_length,is_cancellation",
    [
        (0, 512, 512, False),
        (1, 512, 512, False),
        (2, 512, 512, False),
        (3, 512, 512, False),
        (0, 1024, 2048, True),
        (0, 2048, 4096, True),
    ],
    ids=["random-0", "random-1", "random-2", "random-3", "cancel-short", "cancel-long"],
)
def test_probability_quantization_accuracy(
    seed, query_length, kv_length, is_cancellation
):
    if is_cancellation:
        # Opposite V rows expose probability rounding when two logits almost agree.
        score_gap = 0.000244
        value_scale = 1.7
        device = flaggems_vllm.device
        q = torch.full((query_length, 32, 128), 127, dtype=torch.int8, device=device)
        q[:, :, 0] = 1
        k = torch.full(
            (kv_length // 16, 16, 8, 128), -127, dtype=torch.int8, device=device
        )
        k[0, :2] = 0
        k[0, 1, :, 0] = 1
        v = torch.zeros_like(k)
        v[0, 0] = 127
        v[0, 1] = -127
        qs = torch.ones((1, 32, 1), device=device).expand(1, 32, query_length // 128)
        ks = torch.full((1, 8, 1), score_gap * math.sqrt(128), device=device).expand(
            1, 8, kv_length // 128
        )
        vs = torch.full((1, 8, 1), value_scale, device=device).expand_as(ks)
        cuq = torch.tensor([0, query_length], dtype=torch.int32, device=device)
        used = torch.tensor([kv_length], dtype=torch.int32, device=device)
        table = torch.arange(kv_length // 16, dtype=torch.int32, device=device).reshape(
            1, -1
        )
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=device)
        actual = flaggems_vllm.flash_attn_varlen_func(
            q,
            k,
            v,
            query_length,
            cuq,
            kv_length,
            seqused_k=used,
            block_table=table,
            q_descale=qs,
            k_descale=ks,
            v_descale=vs,
            causal=True,
            out=out,
        )
        expected = torch.full_like(
            actual, -127 * value_scale * math.tanh(score_gap / 2), dtype=torch.float32
        )
        gems_assert_close(
            actual.float(), expected, dtype=torch.float32, atol=0.025, rtol=0.025
        )
        return
    torch.manual_seed(seed)
    tensors, descales, references = [], [], []
    for heads in (16, 8, 8):
        x = torch.randn(
            (512, heads, 128), device=flaggems_vllm.device, dtype=torch.bfloat16
        )
        scale = x.float().abs().amax((0, 2)) / 127
        quant = (x.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        tensors.append(quant)
        descales.append(scale[None, :, None].expand(1, heads, 4))
        references.append(quant.float() * scale[:, None])
    cu = torch.tensor([0, 512], dtype=torch.int32, device=flaggems_vllm.device)
    out, lse = flaggems_vllm.flash_attn_varlen_func(
        *tensors,
        512,
        cu,
        512,
        cu,
        causal=True,
        return_softmax_lse=True,
        q_descale=descales[0],
        k_descale=descales[1],
        v_descale=descales[2],
    )
    expected, expected_lse = _reference(*references, [512], [512], True)
    gems_assert_close(
        out.float(), expected, dtype=(out.float()).dtype, atol=0.025, rtol=0.025
    )
    gems_assert_close(lse, expected_lse, dtype=(lse).dtype, atol=2e-05, rtol=2e-05)


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("heads,kvheads", [(8, 2), (16, 1), (6, 2)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [(-1, -1), (17, 3)])
def test_paged_short_query_gqa(dim, heads, kvheads, causal, window):
    _run_case(
        [1, 4, 8, 16],
        [1, 17, 129, 257],
        dim=dim,
        heads=heads,
        kvheads=kvheads,
        causal=causal,
        paged=True,
        window=window,
        cap=5.0,
        alibi=torch.linspace(0.01, 0.1, heads, device=flaggems_vllm.device),
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("heads,kvheads", [(8, 2), (16, 1)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("broadcast_scales", [False, True])
@pytest.mark.parametrize(
    "qlens,klens",
    [([129, 513], [257, 1025]), ([129, 129], [4097, 17])],
)
def test_paged_long_query_gqa(
    dim, heads, kvheads, causal, broadcast_scales, qlens, klens
):
    _run_case(
        qlens,
        klens,
        dim=dim,
        heads=heads,
        kvheads=kvheads,
        causal=causal,
        paged=True,
        broadcast_scales=broadcast_scales,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("broadcast_scales", [False, True])
@pytest.mark.parametrize(
    "qlens,klens",
    [([1, 2, 513, 1], [129, 257, 1025, 17]), ([1024, 1], [2048, 1024])],
)
def test_paged_mixed_query_grid(dim, causal, broadcast_scales, qlens, klens):
    _run_case(
        qlens,
        klens,
        dim=dim,
        heads=8,
        kvheads=2,
        causal=causal,
        paged=True,
        broadcast_scales=broadcast_scales,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_long_query_strided(dtype):
    _run_case(
        [129, 257],
        [257, 513],
        dim=128,
        heads=8,
        kvheads=2,
        causal=True,
        paged=True,
        strided=True,
        dtype=dtype,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
def test_paged_long_query_empty_kv():
    _run_case(
        [129, 0, 3], [0, 0, 0], dim=128, heads=8, kvheads=2, causal=True, paged=True
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("strided", [False, True])
def test_paged_unused_cache_and_table_slots(strided):
    _run_case(
        [257, 1, 1],
        [129, 513, 65],
        dim=128,
        heads=8,
        kvheads=2,
        causal=True,
        paged=True,
        strided=strided,
        extra_cache_pages=37,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
def test_paged_shared_cache_workspace_fallback():
    qlens, klens = [129, 5, 1], [512, 512, 512]
    q, qs, qr, cuq = _inputs(qlens, 8, 128)
    k, ks, kr, _ = _inputs([512], 2, 128)
    v, vs, vr, _ = _inputs([512], 2, 128)
    v, vs, vr = -v, vs * 1.7, -vr * 1.7
    # All requests share one physical KV cache; duplicating it for each request
    # would exceed the original physical-cache workspace budget.
    table = torch.arange(32, dtype=torch.int32, device=flaggems_vllm.device).expand(
        3, -1
    )
    actual, lse = flaggems_vllm.flash_attn_varlen_func(
        q,
        k.reshape(32, 16, 2, 128),
        v.reshape(32, 16, 2, 128),
        max(qlens),
        cuq,
        max(klens),
        seqused_k=torch.tensor(klens, dtype=torch.int32, device=flaggems_vllm.device),
        block_table=table,
        causal=True,
        return_softmax_lse=True,
        q_descale=qs,
        k_descale=ks.expand(3, -1, -1),
        v_descale=vs.expand(3, -1, -1),
    )
    expected, expected_lse = _reference(
        qr, kr.repeat(3, 1, 1), vr.repeat(3, 1, 1), qlens, klens, True
    )
    gems_assert_close(
        actual.float(), expected, dtype=(actual.float()).dtype, atol=0.025, rtol=0.025
    )
    gems_assert_close(lse, expected_lse, dtype=(lse).dtype, atol=2e-05, rtol=2e-05)


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize(
    "qlens,klens",
    [
        ([0, 1, 16, 17, 513, 2, 0], [0, 33, 257, 19, 1025, 65, 0]),
        ([17, 17, 513], [33, 65, 1025]),
    ],
)
def test_paged_worklist_request_classes(qlens, klens):
    _run_case(qlens, klens, dim=128, heads=8, kvheads=2, causal=True, paged=True)


@pytest.mark.flash_attn_varlen_func_w8a8_int8
def test_paged_worklist_without_long_queries():
    _run_case(
        [0, 1, 2, 4],
        [0, 33, 65, 17],
        dim=128,
        heads=8,
        kvheads=2,
        causal=True,
        paged=True,
        max_query_bound=128,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [(-1, -1), (17, 3)])
def test_paged_mask_phase_boundaries(causal, window):
    _run_case(
        [0, 1, 2, 4] * 8,
        [0, 63, 64, 65, 127, 128, 129, 255] * 4,
        dim=128,
        heads=32,
        kvheads=8,
        causal=causal,
        paged=True,
        window=window,
        extra_cache_pages=11,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("broadcast_scales", [False, True])
@pytest.mark.parametrize("query_len", [4101, 8193])
def test_paged_large_query_tile(causal, broadcast_scales, query_len):
    _run_case(
        [query_len],
        [257],
        dim=128,
        heads=8,
        kvheads=2,
        causal=causal,
        paged=True,
        broadcast_scales=broadcast_scales,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("value", [-127, 127])
@pytest.mark.parametrize("q_scale_factor", [0.03, 0.1])
def test_paged_long_query_constant_v(value, q_scale_factor):
    # Constant V makes accumulation drift visible even when QK is nearly uniform.
    length, heads, kvheads, dim = 8192, 8, 2, 128
    q, qs, _, cuq = _inputs([length], heads, dim, broadcast_scales=True)
    k, ks, _, _ = _inputs([length], kvheads, dim, broadcast_scales=True)
    v = torch.full_like(k, value)
    vs = torch.ones((1, kvheads, 1), device=flaggems_vllm.device).expand(
        1, kvheads, length // DESCALE_BLOCK
    )
    table = torch.arange(length // 16, dtype=torch.int32, device=flaggems_vllm.device)[
        None, :
    ]
    result = flaggems_vllm.flash_attn_varlen_func(
        q,
        k.reshape(-1, 16, kvheads, dim),
        v.reshape(-1, 16, kvheads, dim),
        length,
        cuq,
        length,
        seqused_k=torch.tensor(
            [length], dtype=torch.int32, device=flaggems_vllm.device
        ),
        block_table=table,
        causal=True,
        q_descale=qs * q_scale_factor,
        k_descale=ks,
        v_descale=vs,
    )
    expected = torch.full(
        result.shape, value, dtype=torch.float32, device=flaggems_vllm.device
    )
    gems_assert_close(
        result.float(), expected, dtype=(result.float()).dtype, atol=0.025, rtol=0.025
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("batch", [16, 32, 64])
@pytest.mark.parametrize("broadcast_scales", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_length", [129, 1024])
def test_paged_single_query_gqa(batch, broadcast_scales, causal, kv_length):
    _run_case(
        [1] * batch,
        [kv_length + index % 5 for index in range(batch)],
        heads=32,
        kvheads=8,
        dim=128,
        paged=True,
        causal=causal,
        broadcast_scales=broadcast_scales,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("broadcast_scales", [False, True])
def test_paged_gqa_softcap_alibi(broadcast_scales):
    torch.manual_seed(779)
    slopes = torch.rand((2, 32), device=flaggems_vllm.device) * 0.1
    _run_case(
        [129, 513],
        [257, 1025],
        heads=32,
        kvheads=8,
        dim=128,
        paged=True,
        causal=True,
        cap=4,
        alibi=slopes,
        broadcast_scales=broadcast_scales,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("broadcast_scales", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_small_batch_decode_split_kv(batch, dtype, broadcast_scales, causal):
    _run_case(
        [1] * batch,
        [513 + 129 * (index % 3) for index in range(batch)],
        heads=32,
        kvheads=8,
        dim=128,
        dtype=dtype,
        paged=True,
        causal=causal,
        broadcast_scales=broadcast_scales,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
def test_small_batch_decode_split_kv_empty_request():
    _run_case(
        [1, 1],
        [0, 513],
        heads=32,
        kvheads=8,
        dim=128,
        paged=True,
        causal=True,
        broadcast_scales=True,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("cap,with_alibi", [(4, False), (0, True), (4, True)])
def test_small_batch_decode_split_kv_modifiers(cap, with_alibi):
    torch.manual_seed(779)
    slopes = (
        torch.rand((2, 32), device=flaggems_vllm.device) * 0.1 if with_alibi else None
    )
    _run_case(
        [1, 1],
        [513, 1025],
        heads=32,
        kvheads=8,
        dim=128,
        paged=True,
        causal=True,
        cap=cap,
        alibi=slopes,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("batch", [16, 32, 64])
@pytest.mark.parametrize("kv_length", [129, 513, 2048])
@pytest.mark.parametrize("broadcast_scales", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_paged_two_query_gqa(batch, kv_length, broadcast_scales, causal):
    _run_case(
        [2] * batch,
        [kv_length + index % 5 for index in range(batch)],
        heads=32,
        kvheads=8,
        dim=128,
        paged=True,
        causal=causal,
        broadcast_scales=broadcast_scales,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
def test_small_batch_decode_split_kv_strided():
    _run_case(
        [1, 1],
        [513, 1025],
        heads=16,
        kvheads=4,
        dim=128,
        dtype=torch.float16,
        paged=True,
        strided=True,
        causal=True,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("max_query_bound", [None, 4096])
def test_reordered_causal_gqa_ragged_empty_masked(dtype, max_query_bound):
    _run_case(
        [513, 129, 0],
        [1025, 65, 0],
        heads=32,
        kvheads=8,
        dim=128,
        dtype=dtype,
        paged=True,
        causal=True,
        strided=True,
        max_query_bound=max_query_bound,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("batch", [33, 65])
def test_reordered_worklist_many_requests(batch):
    _run_case(
        [129, 257] + [1] * (batch - 2),
        [513, 1025] + [65] * (batch - 2),
        heads=32,
        kvheads=8,
        dim=128,
        paged=True,
        causal=True,
        broadcast_scales=True,
        max_query_bound=1024,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("kv_length", [15, 16, 17, 63, 64, 65, 129])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_gqa_small_kv_boundaries(kv_length, causal, dtype):
    _run_case(
        [513],
        [kv_length],
        heads=32,
        kvheads=8,
        dim=128,
        paged=True,
        causal=causal,
        dtype=dtype,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("query_length", [129, 255, 511, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_folded_causal_prefill_partial_query_tiles(query_length, dtype):
    _run_case(
        [query_length, query_length - 7],
        [query_length + 17, query_length + 3],
        heads=32,
        kvheads=8,
        dim=128,
        paged=True,
        causal=True,
        dtype=dtype,
    )
