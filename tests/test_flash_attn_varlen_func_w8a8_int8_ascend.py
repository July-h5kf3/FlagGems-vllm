# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public-entry coverage for Ascend mixed Cube/Vector attention."""

import importlib
import math

import pytest
import torch

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close, gems_assert_equal
from tests.test_flash_attn_varlen_func_w8a8_int8 import _inputs, _reference

pytestmark = [
    pytest.mark.flash_attn_varlen_func_w8a8_int8,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "ascend", reason="Ascend-only API"),
]


@pytest.fixture
def launches(monkeypatch):
    inline = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.fused.attention"
    )
    if not inline._supported_device(torch.device(flaggems_vllm.device)):
        pytest.skip("mixed attention is validated for physical 910B4")
    records = []

    class Spy:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def invoke(*args, **kwargs):
                artifact_key = args[-4]
                assert isinstance(artifact_key, tuple) and artifact_key
                assert all(
                    len(value) == 64 and int(value, 16) >= 0 for value in artifact_key
                )
                records.append((args[-2], args[-3]))  # mode, KV width
                return launch(*args, **kwargs)

            return invoke

    head = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.fused.attention"
    )
    monkeypatch.setattr(head, "launch_cube_tle", Spy(head.launch_cube_tle))
    monkeypatch.setattr(
        head, "launch_hybrid_small_tle", Spy(head.launch_hybrid_small_tle)
    )
    monkeypatch.setattr(
        head, "launch_hybrid_large_tle", Spy(head.launch_hybrid_large_tle)
    )
    monkeypatch.setattr(head, "launch_grouped_tle", Spy(head.launch_grouped_tle))
    monkeypatch.setattr(head, "launch_packed_tle", Spy(head.launch_packed_tle))
    return records


def _case(qlens, klens, broadcast, padding=0, heads=32, kvheads=8):
    q, qs, qr, cuq = _inputs(qlens, heads, 128, broadcast)
    k, ks, kr, _ = _inputs(klens, kvheads, 128, broadcast)
    v, vs, vr, _ = _inputs(klens, kvheads, 128, broadcast)
    v, vr = v.flip(-1), vr.flip(-1)
    pages = (max(klens) + 15) // 16 + padding
    table = (
        torch.randperm(len(qlens) * pages, device=q.device)
        .to(torch.int32)
        .reshape(len(qlens), pages)
    )
    kc = torch.zeros(
        (table.numel(), 16, kvheads, 128), dtype=torch.int8, device=q.device
    )
    vc = torch.zeros_like(kc)
    offset = 0
    for b, length in enumerate(klens):
        for start in range(0, length, 16):
            count = min(16, length - start)
            physical = table[b, start // 16].long()
            kc[physical, :count] = k[offset + start : offset + start + count]
            vc[physical, :count] = v[offset + start : offset + start + count]
        offset += length
    args = (q, kc, vc, max(qlens), cuq, max(klens))
    kwargs = dict(
        block_table=table,
        seqused_k=torch.tensor(klens, dtype=torch.int32, device=q.device),
        q_descale=qs,
        k_descale=ks,
        v_descale=vs,
    )
    return args, kwargs, (qr.cpu(), kr.cpu(), vr.cpu())


CASES = [
    ([1, 1], [129, 259], 0),
    ([1, 1], [2049, 4097], 0),
    ([7, 7], [0, 2049], 0),
    ([1, 1], [513, 1025], 0),
    ([7, 7], [0, 1025], 0),
    ([32, 32], [259, 129], 0),
    ([129], [259], 3),
    ([129, 3, 0], [17, 259, 0], None),
]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("qlens,klens,mode", CASES)
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_public_inline_paths(launches, qlens, klens, mode, broadcast, causal):
    # Extra columns must not require descales for padding beyond max_seqlen_k.
    args, kwargs, refs = _case(qlens, klens, broadcast, padding=32)
    out = torch.full(
        args[0].shape, float("nan"), dtype=torch.bfloat16, device=args[0].device
    )
    actual = flaggems_vllm.flash_attn_varlen_func(
        *args, **kwargs, out=out, causal=causal
    )
    assert actual is out
    expected, _ = _reference(*refs, qlens, klens, causal)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    width = 128 if not broadcast else 256
    if broadcast and len(set(qlens)) == 1 and max(qlens) <= 8:
        minimum = 256 if max(qlens) <= 4 else 512
        width = (1024 if max(klens) >= 2048 else 512) if max(klens) > minimum else 256
    expected_modes = (
        [1, 4 if max(qlens) >= 1024 else 2]
        if mode is None and broadcast
        else [0 if mode is None else mode]
    )
    assert launches == [(value, width) for value in expected_modes]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("entry", ["generic", "specialized"])
def test_default_output_and_empty_kv(launches, entry):
    qlens, klens = [1, 1], [0, 0]
    args, kwargs, _ = _case(qlens, klens, False)
    fn = (
        flaggems_vllm.flash_attn_varlen_func
        if entry == "generic"
        else flaggems_vllm.flash_attn_varlen_func_w8a8_int8
    )
    actual = fn(*args, **kwargs)
    assert actual.dtype == torch.bfloat16
    gems_assert_equal(actual, torch.zeros_like(actual))
    assert launches == [(0, 128)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize(
    "option", ["lse", "window", "scale", "softcap", "alibi", "fp16", "strided"]
)
def test_unsupported_features_keep_public_fallback(launches, option):
    qlens, klens = [3, 1], [17, 33]
    args, kwargs, refs = _case(qlens, klens, True)
    refs = list(refs)
    window, cap, alibi = (-1, -1), 0, None
    if option == "lse":
        kwargs["return_softmax_lse"] = True
    elif option == "window":
        window = (7, 1)
        kwargs["window_size"] = window
    elif option == "scale":
        kwargs["softmax_scale"] = 0.07
        refs[0] = refs[0] * (0.07 / (128**-0.5))
    elif option == "softcap":
        cap = 5.0
        kwargs["softcap"] = cap
    elif option == "alibi":
        alibi = torch.linspace(0.01, 0.1, 32, device="cpu")
        kwargs["alibi_slopes"] = alibi.to(args[0].device)
    out = torch.empty(
        args[0].shape,
        dtype=torch.float16 if option == "fp16" else torch.bfloat16,
        device=args[0].device,
    )
    if option == "strided":
        storage = torch.empty(
            (*out.shape[:-2], out.shape[-2] * 2, 128),
            dtype=out.dtype,
            device=out.device,
        )
        out = storage[..., ::2, :]
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kwargs, out=out)
    expected, lse = _reference(
        *refs, qlens, klens, False, window=window, cap=cap, alibi=alibi
    )
    if option == "lse":
        actual, actual_lse = actual
        gems_assert_close(
            actual_lse.cpu(),
            lse,
            dtype=(actual_lse.cpu()).dtype,
            atol=2e-05,
            rtol=2e-05,
        )
    assert actual is out
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert not launches


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("invalid", ["missing_scale", "short_scale", "wrong_dtype"])
def test_invalid_descales_retain_validation(launches, invalid):
    args, kwargs, _ = _case([1, 1], [259, 129], False)
    if invalid == "missing_scale":
        kwargs["q_descale"] = None
        error = TypeError
    elif invalid == "short_scale":
        kwargs["k_descale"] = kwargs["k_descale"][:, :, :1].contiguous()
        error = ValueError
    else:
        kwargs["v_descale"] = kwargs["v_descale"].half()
        error = TypeError
    with pytest.raises(error):
        flaggems_vllm.flash_attn_varlen_func(*args, **kwargs)
    assert not launches


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize(
    "changed", ["used_dtype", "table_dtype", "cuq_dtype", "kv_shape", "cpu_scale"]
)
def test_inline_guard_rejects_invalid_metadata(launches, changed):
    # Exercise admission checks directly: invalid caller metadata must never be
    # passed to a device kernel merely to test that the admission check works.
    inline = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.fused.attention"
    )
    args, kwargs, _ = _case([1, 1], [259, 129], False)
    q, k, v, maxq, cuq, maxk = args
    table, used = kwargs["block_table"], kwargs["seqused_k"]
    qs, ks, vs = kwargs["q_descale"], kwargs["k_descale"], kwargs["v_descale"]
    if changed == "used_dtype":
        used = used.long()
    elif changed == "table_dtype":
        table = table.long()
    elif changed == "cuq_dtype":
        cuq = cuq.long()
    elif changed == "kv_shape":
        v = v[:-1]
    else:
        ks = ks.cpu()
    assert (
        inline.try_run(q, k, v, maxq, cuq, maxk, table, used, qs, ks, vs, None, False)
        is None
    )
    assert not launches


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize(
    "heads,kvheads,qlens,expected_modes",
    [
        (4, 1, [1, 1], [0]),
        (8, 2, [32, 31], None),
        (16, 4, [129, 3, 0], None),
        (64, 16, [129], [3]),
        (40, 8, [129], [3]),
        (8, 8, [128], [3]),
        (16, 1, [128], [3]),
    ],
)
@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_dynamic_head_counts(
    launches, heads, kvheads, qlens, expected_modes, broadcast, causal
):
    klens = [259 if length else 0 for length in qlens]
    args, kwargs, refs = _case(
        qlens, klens, broadcast, padding=17, heads=heads, kvheads=kvheads
    )
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kwargs, causal=causal)
    expected, _ = _reference(*refs, qlens, klens, causal)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    modes = (
        ([1, 4 if max(qlens) >= 1024 else 2] if broadcast else [0])
        if expected_modes is None
        else expected_modes
    )
    width = 128 if not broadcast else 256
    if broadcast and len(set(qlens)) == 1 and max(qlens) <= 8:
        minimum = 256 if max(qlens) <= 4 else 512
        width = (1024 if max(klens) >= 2048 else 512) if max(klens) > minimum else 256
    assert launches == [(mode, width) for mode in modes]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("qlens", [[4, 1], [129, 1]])
def test_nonuniform_other_group_preserves_fallback(launches, qlens):
    # Single-head code supports other ratios; grouped/hybrid code must not
    # silently assume ratio four for these nonuniform calls.
    klens = [33, 17]
    args, kwargs, refs = _case(qlens, klens, True, heads=6, kvheads=2)
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kwargs)
    expected, _ = _reference(*refs, qlens, klens, False)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert not launches


@pytest.mark.flash_attn_varlen_func_w8a8_int8
def test_probability_precision_short_context_prefill(launches):
    # Regression exposed during LSE validation. Keep the existing tolerance:
    # the current N256 approximation must not silently lose low-probability
    # contributions in a short-context, long-query noncausal call.
    qlens, klens = [512], [259]
    args, kwargs, refs = _case(qlens, klens, True, padding=17)
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kwargs, causal=False)
    expected, _ = _reference(*refs, qlens, klens, False)
    assert launches == [(3, 256)]
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize(
    "qlens,klens",
    [
        ([1024, 1, 4], [259, 1024, 17]),
        ([1025, 5, 0], [17, 259, 0]),
        ([1024, 127, 2], [0, 259, 1024]),
        ([1025, 33, 3], [1024, 129, 0]),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_hybrid_head_boundaries(launches, qlens, klens, causal):
    args, kwargs, refs = _case(qlens, klens, True, padding=17)
    out = torch.full(
        args[0].shape, float("nan"), device=args[0].device, dtype=torch.bfloat16
    )
    actual = flaggems_vllm.flash_attn_varlen_func(
        *args, **kwargs, out=out, causal=causal
    )
    assert actual is out
    expected, _ = _reference(*refs, qlens, klens, causal)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert launches == [(1, 256), (4, 256)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("qlens", [[128, 1], [129, 5]])
@pytest.mark.parametrize("causal", [False, True])
def test_small_mixed_keeps_grouped_tile(launches, qlens, causal):
    klens = [259, 17]
    args, kwargs, refs = _case(qlens, klens, True, padding=17)
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kwargs, causal=causal)
    expected, _ = _reference(*refs, qlens, klens, causal)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert launches == [(1, 256), (2, 256)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("q_factor", [0.125, 1.0, 8.0])
@pytest.mark.parametrize("causal", [False, True])
def test_head_probability_flat_and_peaked(launches, q_factor, causal):
    qlens, klens = [512], [259]
    args, kwargs, refs = _case(qlens, klens, True, padding=17)
    kwargs["q_descale"] = kwargs["q_descale"] * q_factor
    kwargs["v_descale"] = (kwargs["v_descale"][:, :, :1] * 4).expand_as(
        kwargs["v_descale"]
    )
    refs = (refs[0] * q_factor, refs[1], refs[2] * 4)
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kwargs, causal=causal)
    expected, _ = _reference(*refs, qlens, klens, causal)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert launches == [(3, 256)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("qlens", [[5, 5], [8, 8], [9, 9], [16, 16], [17, 5, 1]])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("block_k", [False, True])
def test_grouped_fixed_probability(launches, qlens, causal, block_k):
    klens = [259] * len(qlens)
    args, kwargs, refs = _case(qlens, klens, True, padding=17)
    kwargs["q_descale"] = kwargs["q_descale"] * 4
    kwargs["v_descale"] = (kwargs["v_descale"][:, :, :1] * 4).expand_as(
        kwargs["v_descale"]
    )
    refs = (refs[0] * 4, refs[1], refs[2] * 4)
    if block_k:
        factors = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        kwargs["k_descale"] = kwargs["k_descale"][:, :, :1] * factors.to(args[0].device)
        key_reference = refs[1].clone()
        offset = 0
        for length in klens:
            key_reference[offset : offset + length] *= factors[
                torch.arange(length) // 128
            ][:, None, None]
            offset += length
        refs = (refs[0], key_reference, refs[2])
    out = torch.full(
        args[0].shape, float("nan"), device=args[0].device, dtype=torch.bfloat16
    )
    actual = flaggems_vllm.flash_attn_varlen_func(
        *args, **kwargs, out=out, causal=causal
    )
    expected, _ = _reference(*refs, qlens, klens, causal)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    width = 256
    assert launches == ([(1, 256), (2, 256)] if qlens == [17, 5, 1] else [(0, width)])


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize(
    "qlens,klens", [([1, 1], [128, 256]), ([8, 8], [128, 256]), ([8, 8], [256, 512])]
)
@pytest.mark.parametrize("causal", [False, True])
def test_short_kv_retains_n256(launches, qlens, klens, causal):
    args, kwargs, refs = _case(qlens, klens, True, padding=17)
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kwargs, causal=causal)
    expected, _ = _reference(*refs, qlens, klens, causal)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert launches == [(0, 256)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("qlen", [1, 7])
@pytest.mark.parametrize("causal", [False, True])
def test_long_kv_varying_key_blocks(launches, qlen, causal):
    qlens, klens = [qlen, qlen], [2051, 17]
    args, kwargs, refs = _case(qlens, klens, True, padding=17, heads=8, kvheads=2)
    factors = torch.linspace(0.5, 2.0, (max(klens) + 127) // 128)
    kwargs["k_descale"] = kwargs["k_descale"][:, :, :1] * factors.to(args[0].device)
    key_reference = refs[1].clone()
    offset = 0
    for length in klens:
        key_reference[offset : offset + length] *= factors[torch.arange(length) // 128][
            :, None, None
        ]
        offset += length
    refs = (refs[0], key_reference, refs[2])
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kwargs, causal=causal)
    expected, _ = _reference(*refs, qlens, klens, causal)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert launches == [(0, 1024)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize(
    "query_length,key_length,score_gap,value_scale,mixed_keys",
    [
        (1, 2048, 11.5, 0.06, False),
        (1, 512, 11.5, 0.1, False),
        (1, 2048, 8.0, 1.0, True),
        (8, 2048, 8.0, 1.0, True),
    ],
)
def test_probability_cancellation(
    launches, query_length, key_length, score_gap, value_scale, mixed_keys
):
    device = flaggems_vllm.device
    pages = (key_length + 15) // 16
    input_scale = (score_gap / (128 * 127 * 127 * (128**-0.5))) ** 0.5
    query = torch.full((query_length, 32, 128), 127, dtype=torch.int8, device=device)
    key_cache = torch.zeros((pages, 16, 8, 128), dtype=torch.int8, device=device)
    if mixed_keys:
        key_cache.reshape(-1, 8, 128)[1::2] = 1
        small_mass = (key_length // 2 - 1) * math.exp(-score_gap) + (
            key_length // 2
        ) * math.exp(-score_gap * 126 / 127)
        first_value = max(-128, min(127, round(-127 * small_mass)))
    else:
        first_value = -1
    key_cache[0, 0] = 127
    value_cache = torch.full_like(key_cache, 127)
    value_cache[0, 0] = first_value
    cu_query = torch.tensor([0, query_length], dtype=torch.int32, device=device)
    block_table = torch.arange(pages, dtype=torch.int32, device=device)[None, :]
    used_key = torch.tensor([key_length], dtype=torch.int32, device=device)
    query_descale = torch.full(
        (1, 32, 1), input_scale, dtype=torch.float32, device=device
    )
    key_descale = torch.full(
        (1, 8, 1), input_scale, dtype=torch.float32, device=device
    ).expand(1, 8, (key_length + 127) // 128)
    value_descale = torch.full(
        (1, 8, 1), value_scale, dtype=torch.float32, device=device
    ).expand_as(key_descale)
    query_reference = query.cpu().float() * query_descale.cpu()[0, :, 0][None, :, None]
    key_reference = (
        key_cache.cpu().reshape(-1, 8, 128)[:key_length].float()
        * key_descale.cpu()[0, :, 0][None, :, None]
    )
    value_reference = (
        value_cache.cpu().reshape(-1, 8, 128)[:key_length].float()
        * value_descale.cpu()[0, :, 0][None, :, None]
    )
    actual = flaggems_vllm.flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        query_length,
        cu_query,
        key_length,
        block_table=block_table,
        seqused_k=used_key,
        q_descale=query_descale,
        k_descale=key_descale,
        v_descale=value_descale,
        causal=True,
    )
    expected, _ = _reference(
        query_reference,
        key_reference,
        value_reference,
        [query_length],
        [key_length],
        True,
    )
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("query_length", [1, 8])
@pytest.mark.parametrize("head_mode", ["all", "even", "odd"])
@pytest.mark.parametrize(
    "key_lengths,peak_positions",
    [
        ([4096, 2049, 1025, 3073], [2048, 1024, 0, 2048]),
        ([1537, 513, 0, 17], [1024, 0, None, 0]),
    ],
)
def test_late_peaks_and_mixed_heads(
    launches, query_length, head_mode, key_lengths, peak_positions
):
    query_lengths = [query_length] * 4
    args, kwargs, _ = _case(query_lengths, key_lengths, True, padding=17)
    query, key_cache, value_cache, max_query, cu_query, max_key = args
    table_cpu = kwargs["block_table"].cpu()
    input_scale = (8 / (128 * 127 * 127 * (128**-0.5))) ** 0.5
    key_cpu = torch.zeros(key_cache.shape, dtype=torch.int8)
    value_cpu = torch.zeros(value_cache.shape, dtype=torch.int8)
    key_reference, value_reference = [], []
    for batch, (length, peak) in enumerate(zip(key_lengths, peak_positions)):
        tokens = torch.arange(length)
        physical_pages = table_cpu[batch, tokens // 16].long()
        logical_key = torch.zeros((length, 8, 128), dtype=torch.int8)
        if length:
            logical_key[1::2] = 1
            logical_key[peak] = 127
            mass = (length - length // 2 - 1) * math.exp(-8) + (length // 2) * math.exp(
                -8 * 126 / 127
            )
            choices = range(64, max(65, min(127, int(127 / max(mass, 1e-10))) + 1))
            background = min(
                choices, key=lambda value: abs(value * mass - round(value * mass))
            )
            logical_value = torch.full_like(logical_key, background)
            logical_value[peak] = max(-128, min(127, round(-background * mass)))
            key_cpu[physical_pages, tokens % 16] = logical_key
            value_cpu[physical_pages, tokens % 16] = logical_value
        else:
            logical_value = torch.empty((0, 8, 128), dtype=torch.int8)
        key_reference.append(logical_key.float() * input_scale)
        value_reference.append(logical_value.float() * 8)
    key_cache.copy_(key_cpu)
    value_cache.copy_(value_cpu)
    kwargs["q_descale"] = torch.full(
        (4, 32, 1), input_scale, dtype=torch.float32, device=query.device
    )
    kwargs["k_descale"] = torch.full(
        (4, 8, 1), input_scale, dtype=torch.float32, device=query.device
    ).expand(4, 8, (max_key + 127) // 128)
    kwargs["v_descale"] = torch.full(
        (4, 8, 1), 8.0, dtype=torch.float32, device=query.device
    ).expand_as(kwargs["k_descale"])
    query.zero_()
    if head_mode == "all":
        query.fill_(127)
    elif head_mode == "even":
        query[:, 0::2] = 127
    else:
        query[:, 1::2] = 127
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kwargs, causal=True)
    expected, _ = _reference(
        query.cpu().float() * input_scale,
        torch.cat(key_reference),
        torch.cat(value_reference),
        query_lengths,
        key_lengths,
        True,
    )
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize(
    "qlens,klens,causal,declared_max,expected_launches",
    [
        ([128], [3135], True, None, [(3, 256)]),
        ([128], [3136], True, None, [(3, 512)]),
        ([128], [2047], False, None, [(3, 256)]),
        ([128], [2048], False, None, [(3, 512)]),
        ([257], [4097], True, None, [(3, 512)]),
        ([128, 128], [4096, 17], True, None, [(5, 256), (5, 512)]),
        ([128, 128], [4096, 17], False, None, [(5, 256), (5, 512)]),
        ([128, 128], [4096, 4096], True, None, [(5, 256), (5, 512)]),
        ([129, 129], [17, 0], True, 4096, [(5, 256), (5, 512)]),
        ([129, 129], [17, 0], False, 4096, [(5, 256), (5, 512)]),
    ],
)
def test_wide_head_dispatch_and_partitions(
    launches, monkeypatch, qlens, klens, causal, declared_max, expected_launches
):
    torch.manual_seed(9182)
    padding = 17
    if declared_max is not None:
        padding += (declared_max + 15) // 16 - (max(klens) + 15) // 16
    args, kwargs, refs = _case(qlens, klens, True, padding=padding)
    if declared_max is not None:
        args = (*args[:5], declared_max)
        for name in ("k_descale", "v_descale"):
            kwargs[name] = kwargs[name][:, :, :1].expand(
                *kwargs[name].shape[:2], (declared_max + 127) // 128
            )
    raw_output = torch.full(
        (args[0].shape[0] + 2, *args[0].shape[1:]),
        13.0,
        dtype=torch.bfloat16,
        device=args[0].device,
    )
    out = raw_output[1:-1]
    out.fill_(float("nan"))
    original_empty = torch.empty
    workspaces = []

    def guarded_empty(*shape, **options):
        if (
            len(shape) == 1
            and isinstance(shape[0], int)
            and options.get("dtype") == torch.uint8
        ):
            raw = original_empty(shape[0] + 128, **options)
            raw.fill_(0xA5)
            workspaces.append(raw)
            return raw[64:-64]
        return original_empty(*shape, **options)

    monkeypatch.setattr(torch, "empty", guarded_empty)
    actual = flaggems_vllm.flash_attn_varlen_func(
        *args, **kwargs, out=out, causal=causal
    )
    assert actual is out
    assert (
        torch.all(raw_output[0] == 13.0).item()
        and torch.all(raw_output[-1] == 13.0).item()
    )
    assert workspaces
    for raw in workspaces:
        assert (
            torch.all(raw[:64] == 0xA5).item() and torch.all(raw[-64:] == 0xA5).item()
        )
    assert launches == expected_launches
    expected, _ = _reference(*refs, qlens, klens, causal)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("query_length", [128, 257])
@pytest.mark.parametrize(
    "gap,value_scale,mixed_keys", [(11.5, 0.06, False), (8.0, 1.0, True)]
)
def test_wide_head_probability_cancellation(
    launches, monkeypatch, query_length, gap, value_scale, mixed_keys
):
    # The original short-query fixture supplies a single scale block. Expand
    # that constant view to satisfy the unchanged public long-query contract.
    original = flaggems_vllm.flash_attn_varlen_func

    def expanded_scale_case(*args, **kwargs):
        for name, length in (
            ("q_descale", args[3]),
            ("k_descale", args[5]),
            ("v_descale", args[5]),
        ):
            if kwargs[name].shape[-1] == 1:
                kwargs[name] = kwargs[name].expand(
                    *kwargs[name].shape[:2], (length + 127) // 128
                )
        return original(*args, **kwargs)

    monkeypatch.setattr(flaggems_vllm, "flash_attn_varlen_func", expanded_scale_case)
    test_probability_cancellation(
        launches, query_length, 4096, gap, value_scale, mixed_keys
    )
    assert launches == [(3, 512)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("heads,kvheads", [(8, 8), (40, 8), (16, 1)])
def test_partitioned_head_group_ratios(launches, heads, kvheads):
    torch.manual_seed(9183)
    qlens, klens = [128, 128], [4096, 17]
    args, kwargs, refs = _case(
        qlens, klens, True, padding=17, heads=heads, kvheads=kvheads
    )
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kwargs, causal=True)
    expected, _ = _reference(*refs, qlens, klens, True)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert launches == [(5, 256), (5, 512)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("klens", [[4097], [4097, 17]])
@pytest.mark.parametrize("causal", [False, True])
def test_wide_head_strided_qk_block_scales(launches, klens, causal):
    torch.manual_seed(9184)
    qlens = [257] * len(klens)
    args, kwargs, refs = _case(qlens, klens, True, padding=17)
    for name, step in (("q_descale", 2), ("k_descale", 3)):
        original = kwargs[name]
        factors = (
            torch.arange(original.shape[2], device=original.device, dtype=torch.float32)
            % 5
            + 1
        ) / 2
        values = original * factors[None, None, :]
        padded = torch.full(
            (*original.shape[:2], original.shape[2] * step),
            float("nan"),
            dtype=torch.float32,
            device=original.device,
        )
        padded[:, :, ::step] = values
        kwargs[name] = padded[:, :, ::step]
    qcpu = args[0].cpu().float()
    kcpu = args[1].cpu().float()
    table = kwargs["block_table"].cpu().long()
    qs, ks = kwargs["q_descale"].cpu(), kwargs["k_descale"].cpu()
    qrefs, krefs = [], []
    qoffset = 0
    for batch, (qlen, klen) in enumerate(zip(qlens, klens)):
        qscale = (
            qs[batch]
            .index_select(1, torch.arange(qlen, device="cpu") // 128)
            .transpose(0, 1)
        )
        qrefs.append(qcpu[qoffset : qoffset + qlen] * qscale[:, :, None])
        logical_k = kcpu[table[batch, : (klen + 15) // 16]].reshape(-1, 8, 128)[:klen]
        kscale = (
            ks[batch]
            .index_select(1, torch.arange(klen, device="cpu") // 128)
            .transpose(0, 1)
        )
        krefs.append(logical_k * kscale[:, :, None])
        qoffset += qlen
    expected, _ = _reference(
        torch.cat(qrefs), torch.cat(krefs), refs[2], qlens, klens, causal
    )
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kwargs, causal=causal)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert launches == ([(3, 512)] if len(klens) == 1 else [(5, 256), (5, 512)])


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize(
    "batch,klens,heads,kvheads,broadcast,mode",
    [
        (8, [1024] * 8, 32, 8, True, 6),
        (7, [1024] * 7, 32, 8, True, 0),
        (8, [1023] * 8, 32, 8, True, 0),
        (8, [1024] * 8, 32, 8, False, 0),
        (16, [4097, 17, 0, 1024] * 4, 32, 8, True, 6),
        (16, [1024] * 16, 16, 4, True, 6),
        (4, [2049, 1024, 17, 0], 64, 16, True, 6),
        (16, [1024] * 16, 24, 6, True, 0),
    ],
)
def test_packed_decode_dispatch_and_outputs(
    launches, monkeypatch, batch, klens, heads, kvheads, broadcast, mode
):
    torch.manual_seed(3111)
    qlens = [1] * batch
    args, kwargs, refs = _case(
        qlens, klens, broadcast, padding=17, heads=heads, kvheads=kvheads
    )
    raw_output = torch.full(
        (batch + 2, heads, 128), 13.0, dtype=torch.bfloat16, device=args[0].device
    )
    output = raw_output[1:-1]
    output.fill_(float("nan"))
    original = torch.empty
    guards = []

    def empty(*shape, **options):
        if (
            len(shape) == 1
            and isinstance(shape[0], int)
            and options.get("dtype") == torch.uint8
        ):
            raw = original(shape[0] + 128, **options)
            raw.fill_(0xA5)
            guards.append(raw)
            return raw[64:-64]
        return original(*shape, **options)

    monkeypatch.setattr(torch, "empty", empty)
    actual = flaggems_vllm.flash_attn_varlen_func(
        *args, **kwargs, out=output, causal=True
    )
    expected, _ = _reference(*refs, qlens, klens, True)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert (
        actual is output
        and torch.all(raw_output[0] == 13.0).item()
        and torch.all(raw_output[-1] == 13.0).item()
    )
    for raw in guards:
        assert (
            torch.all(raw[:64] == 0xA5).item() and torch.all(raw[-64:] == 0xA5).item()
        )
    assert launches == [(mode, 128 if not broadcast else 512)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize(
    "klen,gap,vscale,mixed",
    [(1024, 11.5, 0.1, False), (2048, 8.0, 1.0, True), (4097, 8.0, 4.0, True)],
)
def test_packed_decode_cancellation_multiple_groups(
    launches, monkeypatch, klen, gap, vscale, mixed
):
    original = flaggems_vllm.flash_attn_varlen_func

    def batched(q, k, v, mq, cu, mk, **kw):
        batch = 16
        q = q.repeat(batch, 1, 1)
        cu = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
        kw["block_table"] = kw["block_table"].expand(batch, -1).contiguous()
        kw["seqused_k"] = kw["seqused_k"].expand(batch).contiguous()
        for name in ("q_descale", "k_descale", "v_descale"):
            kw[name] = kw[name].expand(batch, -1, -1)
        result = original(q, k, v, mq, cu, mk, **kw)
        gems_assert_equal(result, result[:1].expand_as(result))
        return result[:1]

    monkeypatch.setattr(flaggems_vllm, "flash_attn_varlen_func", batched)
    test_probability_cancellation(None, 1, klen, gap, vscale, mixed)
    assert launches == [(6, 512)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("batch", [8, 16])
def test_packed_decode_strided_scales(launches, batch):
    torch.manual_seed(3112)
    klens = [1301, 17, 0, 1025] * (batch // 4)
    args, kw, refs = _case([1] * batch, klens, True, padding=17)
    qs = kw["q_descale"]
    qstorage = torch.full(
        (batch, 64, 1), float("nan"), dtype=torch.float32, device=qs.device
    )
    qstorage[:, ::2] = qs
    kw["q_descale"] = qstorage[:, ::2]
    ks = kw["k_descale"]
    blocks = ks.shape[2]
    factors = (torch.arange(blocks, dtype=torch.float32, device=ks.device) % 5 + 1) / 2
    kstorage = torch.full(
        (batch, 8, blocks * 3), float("nan"), dtype=torch.float32, device=ks.device
    )
    kstorage[:, :, ::3] = ks * factors[None, None, :]
    kw["k_descale"] = kstorage[:, :, ::3]
    signs = torch.tensor([1.0, -1.0, 2.0, -0.5, 1.0, -2.0, 0.5, -1.0], device=ks.device)
    kw["v_descale"] = (kw["v_descale"][:, :, :1] * signs[None, :, None]).expand_as(ks)
    qref = args[0].cpu().float() * kw["q_descale"].cpu()[:, :, 0, None]
    kc = args[1].cpu().float()
    table = kw["block_table"].cpu().long()
    kscpu = kw["k_descale"].cpu()
    keys = []
    values = []
    offset = 0
    for b, length in enumerate(klens):
        logical = kc[table[b, : (length + 15) // 16]].reshape(-1, 8, 128)[:length]
        scale = (
            kscpu[b]
            .index_select(1, torch.arange(length, device="cpu") // 128)
            .transpose(0, 1)
        )
        keys.append(logical * scale[:, :, None])
        values.append(refs[2][offset : offset + length] * signs.cpu()[None, :, None])
        offset += length
    expected, _ = _reference(
        qref, torch.cat(keys), torch.cat(values), [1] * batch, klens, True
    )
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kw, causal=True)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert launches == [(6, 512)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("poison", [1515870810, -1010580541])
def test_hybrid_vector_block_scales_and_workspace(
    launches, monkeypatch, causal, poison
):
    torch.manual_seed(111801)
    qlens, klens = [32, 4, 3, 2, 1, 0], [259, 17, 129, 257, 0, 0]
    args, kwargs, refs = _case(qlens, klens, True, padding=32, heads=16, kvheads=4)
    query_storage = torch.empty((6, 32, 1), dtype=torch.float32, device=args[0].device)
    query_scale = query_storage[:, ::2]
    query_scale.copy_(kwargs["q_descale"][:, :, :1])
    kwargs["q_descale"] = query_scale
    blocks = 3
    key_storage = torch.empty(
        (6, 4, blocks * 2), dtype=torch.float32, device=args[0].device
    )
    key_scale = key_storage[:, :, ::2]
    factors = 0.75 + torch.arange(blocks, device=args[0].device) * 0.125
    key_scale.copy_(kwargs["k_descale"][:, :, :1] * factors)
    kwargs["k_descale"] = key_scale
    key_reference = refs[1].clone()
    offset = 0
    for length in klens:
        for start in range(0, length, 128):
            key_reference[offset + start : offset + min(length, start + 128)] *= (
                0.75 + (start // 128) * 0.125
            )
        offset += length
    expected, _ = _reference(refs[0], key_reference, refs[2], qlens, klens, causal)
    head = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.fused.attention"
    )
    original = head.launch_hybrid_small_tle

    class PoisonWorkspace:
        def __getitem__(self, grid):
            def invoke(*kernel_args, **options):
                kernel_args[9].fill_(poison)
                return original[grid](*kernel_args, **options)

            return invoke

    monkeypatch.setattr(head, "launch_hybrid_small_tle", PoisonWorkspace())
    guarded = torch.full(
        (args[0].numel() + 32,), 7, dtype=torch.bfloat16, device=args[0].device
    )
    output = guarded[16:-16].view(args[0].shape)
    actual = flaggems_vllm.flash_attn_varlen_func(
        *args, **kwargs, causal=causal, out=output
    )
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    gems_assert_close(
        guarded[:16], torch.full_like(guarded[:16], 7), dtype=(guarded[:16]).dtype
    )
    gems_assert_close(
        guarded[-16:], torch.full_like(guarded[-16:], 7), dtype=(guarded[-16:]).dtype
    )
    assert launches == [(1, 256), (2, 256)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "qlens,klens",
    [([128, 128], [0, 259]), ([129], [129]), ([257], [129]), ([256], [257])],
)
def test_n128_vector_boundaries(launches, qlens, klens, causal):
    torch.manual_seed(10101)
    args, kwargs, refs = _case(qlens, klens, False, padding=17, heads=16, kvheads=1)
    # Preserve scalar values while exercising non-contiguous descale views.
    for name in ("q_descale", "k_descale", "v_descale"):
        value = kwargs[name]
        storage = torch.empty(
            (*value.shape[:-1], value.shape[-1] * 2),
            dtype=value.dtype,
            device=value.device,
        )
        storage[..., ::2].copy_(value)
        kwargs[name] = storage[..., ::2]
    guard = torch.full(
        (args[0].numel() + 256,), 13, dtype=torch.bfloat16, device=args[0].device
    )
    out = guard[128:-128].view(args[0].shape)
    actual = flaggems_vllm.flash_attn_varlen_func(
        *args, **kwargs, causal=causal, out=out
    )
    expected, _ = _reference(*refs, qlens, klens, causal)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert actual.data_ptr() == out.data_ptr()
    assert torch.all(guard[:128] == 13) and torch.all(guard[-128:] == 13)
    assert launches == [(3, 128)]


@pytest.mark.flash_attn_varlen_func_w8a8_int8
@pytest.mark.parametrize("qlen", [128, 129, 512])
@pytest.mark.parametrize("factor", [0.125, 8.0])
@pytest.mark.parametrize("causal", [False, True])
def test_n128_vector_flat_and_peaked(launches, qlen, factor, causal):
    torch.manual_seed(10102)
    qlens, klens = [qlen], [259]
    args, kwargs, refs = _case(qlens, klens, False, padding=17)
    kwargs["q_descale"] *= factor
    kwargs["v_descale"] *= 4
    refs = (refs[0] * factor, refs[1], refs[2] * 4)
    actual = flaggems_vllm.flash_attn_varlen_func(*args, **kwargs, causal=causal)
    expected, _ = _reference(*refs, qlens, klens, causal)
    gems_assert_close(
        actual.cpu().float(),
        expected,
        dtype=(actual.cpu().float()).dtype,
        atol=0.025,
        rtol=0.025,
    )
    assert launches == [(3, 128)]
