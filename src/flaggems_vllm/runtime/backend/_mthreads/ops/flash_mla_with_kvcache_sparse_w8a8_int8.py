# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Callable, Optional

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_sparse_int8_mla"),
    key=["ROWS", "H", "K", "ROPE", "SPLITS"],
)
@triton.jit
def sparse_int8_mla(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Lengths,
    Partial,
    Stats,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    split = tl.program_id(2)
    dims = tl.arange(0, 512)
    columns = tl.arange(0, BN)
    query = tl.load(
        Q + (row * H + heads[:, None]) * 512 + dims[None, :], heads[:, None] < H, 0
    )
    qscale = tl.load(QS + row * H + heads, heads < H, 0)
    if ROPE:
        rdims = tl.arange(0, 64)
        qrope = tl.load(
            QR + (row * H + heads[:, None]) * 64 + rdims[None, :], heads[:, None] < H, 0
        )
    length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), K) if HAS_LENGTH else K
    maximum = tl.full((BH,), float("-inf"), tl.float32)
    denominator = tl.full((BH,), 0.0, tl.float32)
    accumulator = tl.full((BH, 512), 0.0, tl.float32)
    for begin in range(split * CHUNK, tl.minimum((split + 1) * CHUNK, length), BN):
        offsets = begin + columns
        tokens = tl.load(Indices + row * K + offsets, offsets < length, -1)
        valid = (offsets < length) & (tokens >= 0) & (tokens < TOKENS)
        safe_tokens = tl.where(valid, tokens, 0)
        keys = tl.load(
            KV + safe_tokens[None, :].to(tl.int64) * 512 + dims[:, None],
            valid[None, :],
            0,
        )
        kscale = tl.load(KS + safe_tokens, valid, 0)
        scores = tl.dot(query, keys).to(tl.float32)
        if ROPE:
            krope = tl.load(
                KR + safe_tokens[None, :].to(tl.int64) * 64 + rdims[:, None],
                valid[None, :],
                0,
            )
            scores += tl.dot(qrope, krope).to(tl.float32)
        scores *= qscale[:, None] * kscale[None, :] * SCALE
        scores = tl.where(valid[None, :], scores, float("-inf"))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == float("-inf"), 0.0, next_max)
        correction = tl.exp(maximum - safe_max)
        probability = tl.exp(scores - safe_max[:, None])
        denominator = denominator * correction + tl.sum(probability, 1)
        weights = probability * kscale[None, :]
        pscale = tl.maximum(tl.max(weights, 1) / 127.0, 1.0e-30)
        normalized = weights / pscale[:, None]
        high = tl.floor(normalized + 0.5).to(tl.int8)
        residual = (normalized - high.to(tl.float32)) * 254.0
        low = tl.minimum(tl.maximum(tl.floor(residual + 0.5), -127.0), 127.0).to(
            tl.int8
        )
        values = tl.trans(keys)
        product = tl.dot(high, values).to(tl.float32)
        product += tl.dot(low, values).to(tl.float32) * (1.0 / 254.0)
        accumulator = accumulator * correction[:, None] + product * pscale[:, None]
        maximum = next_max
    base = (row * SPLITS + split) * H + heads
    tl.store(
        Partial + base[:, None] * 512 + dims[None, :], accumulator, heads[:, None] < H
    )
    tl.store(Stats + base * 2, maximum, heads < H)
    tl.store(Stats + base * 2 + 1, denominator, heads < H)


@libentry()
@triton.jit
def combine_sparse_int8(
    Partial,
    Stats,
    Sink,
    Output,
    LSE,
    H: tl.constexpr,
    SQ: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    splits = tl.arange(0, BLOCK_SPLITS)
    dims = tile * 128 + tl.arange(0, 128)
    base = (row * SPLITS + splits) * H + head
    maxima = tl.load(Stats + base * 2, splits < SPLITS, float("-inf"))
    sums = tl.load(Stats + base * 2 + 1, splits < SPLITS, 0.0)
    maximum = tl.max(maxima, 0)
    safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
    weights = tl.exp(maxima - safe_max)
    denominator = tl.sum(weights * sums, 0)
    logsum = tl.where(denominator > 0, safe_max + tl.log(denominator), float("inf"))
    partial = tl.load(
        Partial + base[:, None] * 512 + dims[None, :], splits[:, None] < SPLITS, 0.0
    )
    values = tl.sum(partial * weights[:, None], 0) / tl.where(
        denominator > 0, denominator, 1.0
    )
    if HAS_SINK:
        sink = tl.load(Sink + head)
        values *= tl.where(denominator > 0, 1.0 / (1.0 + tl.exp(sink - logsum)), 0.0)
    tl.store(Output + (row * H + head) * 512 + dims, values)
    if tile == 0:
        tl.store(LSE + (row // SQ * H + head) * SQ + row % SQ, logsum)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_sparse_int8_qk"),
    key=["ROWS", "H", "K", "ROPE"],
)
@triton.jit
def sparse_int8_qk(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Indices,
    Lengths,
    Scores,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    offsets = tl.program_id(2) * BN + tl.arange(0, BN)
    length = tl.minimum(tl.maximum(tl.load(Lengths + row), 0), K) if HAS_LENGTH else K
    tokens = tl.load(Indices + row * K + offsets, offsets < length, -1)
    valid = (offsets < length) & (tokens >= 0) & (tokens < TOKENS)
    tokens = tl.where(valid, tokens, 0)
    dims = tl.arange(0, 512)
    query = tl.load(
        Q + (row * H + heads[:, None]) * 512 + dims[None, :], heads[:, None] < H, 0
    )
    keys = tl.load(
        KV + tokens[None, :].to(tl.int64) * 512 + dims[:, None], valid[None, :], 0
    )
    scores = tl.dot(query, keys).to(tl.float32)
    if ROPE:
        rdims = tl.arange(0, 64)
        query_rope = tl.load(
            QR + (row * H + heads[:, None]) * 64 + rdims[None, :], heads[:, None] < H, 0
        )
        key_rope = tl.load(
            KR + tokens[None, :].to(tl.int64) * 64 + rdims[:, None], valid[None, :], 0
        )
        scores += tl.dot(query_rope, key_rope)
    qscale = tl.load(QS + row * H + heads, heads < H, 0)
    kscale = tl.load(KS + tokens, valid, 0)
    scores = scores * qscale[:, None] * kscale[None, :] * SCALE
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tl.store(
        Scores + (row * H + heads[:, None]) * K + offsets[None, :],
        scores,
        (heads[:, None] < H) & (offsets[None, :] < K),
    )


@libentry()
@triton.jit
def sparse_int8_probabilities(
    Scores,
    KS,
    Indices,
    High,
    Low,
    Scales,
    Sink,
    LSE,
    H: tl.constexpr,
    SQ: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    scores = tl.load(
        Scores + (row * H + head) * K + offsets, offsets < K, float("-inf")
    )
    maximum = tl.max(scores, 0)
    safe_max = tl.where(maximum == float("-inf"), 0.0, maximum)
    probability = tl.exp(scores - safe_max)
    denominator = tl.sum(probability, 0)
    lse = tl.where(denominator > 0, safe_max + tl.log(denominator), float("inf"))
    tokens = tl.load(Indices + row * K + offsets, offsets < K, -1)
    valid = (offsets < K) & (tokens >= 0) & (tokens < TOKENS)
    kscale = tl.load(KS + tl.where(valid, tokens, 0), valid, 0)
    weights = probability * kscale
    pscale = tl.maximum(tl.max(weights, 0) / 127.0, 1.0e-30)
    normalized = weights * tl.div_rn(1.0, pscale)
    high = tl.floor(normalized + 0.5).to(tl.int8)
    low = tl.minimum(
        tl.maximum(tl.floor((normalized - high.to(tl.float32)) * 254.0 + 0.5), -127.0),
        127.0,
    ).to(tl.int8)
    pscale /= tl.where(denominator > 0, denominator, 1.0)
    if HAS_SINK:
        sink = tl.load(Sink + head)
        pscale *= tl.where(denominator > 0, 1.0 / (1.0 + tl.exp(sink - lse)), 0.0)
    tl.store(High + (row * H + head) * K + offsets, high, offsets < K)
    tl.store(Low + (row * H + head) * K + offsets, low, offsets < K)
    tl.store(Scales + row * H + head, pscale)
    tl.store(LSE + (row // SQ * H + head) * SQ + row % SQ, lse)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_sparse_int8_pv"), key=["ROWS", "H", "K"]
)
@triton.jit
def sparse_int8_pv(
    High,
    Low,
    Scales,
    KV,
    Indices,
    Output,
    ROWS: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    TOKENS: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    BV: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.program_id(1) * BH + tl.arange(0, BH)
    dims = tl.program_id(2) * BV + tl.arange(0, BV)
    columns = tl.arange(0, BN)
    high_sum = tl.full((BH, BV), 0, tl.int32)
    low_sum = tl.full((BH, BV), 0, tl.int32)
    for begin in range(tl.cdiv(K, BN)):
        offsets = begin * BN + columns
        tokens = tl.load(Indices + row * K + offsets, offsets < K, -1)
        valid = (offsets < K) & (tokens >= 0) & (tokens < TOKENS)
        values = tl.load(
            KV + tl.where(valid, tokens, 0)[:, None].to(tl.int64) * 512 + dims[None, :],
            valid[:, None],
            0,
        )
        high = tl.load(
            High + (row * H + heads[:, None]) * K + offsets[None, :],
            (heads[:, None] < H) & (offsets[None, :] < K),
            0,
        )
        low = tl.load(
            Low + (row * H + heads[:, None]) * K + offsets[None, :],
            (heads[:, None] < H) & (offsets[None, :] < K),
            0,
        )
        high_sum += tl.dot(high, values)
        low_sum += tl.dot(low, values)
    scale = tl.load(Scales + row * H + heads, heads < H, 0)
    output = (high_sum.to(tl.float32) + low_sum.to(tl.float32) * (1.0 / 254.0)) * scale[
        :, None
    ]
    tl.store(
        Output + (row * H + heads[:, None]) * 512 + dims[None, :],
        output,
        heads[:, None] < H,
    )


def prepare_flash_mla_with_kvcache_sparse_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[
    Callable[[], tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]
]:
    """Prepare contiguous page64 INT8 MLA; scales must be positive and finite.

    RoPE is divided by content scales. Calls through a prepared handle must be
    serialized on one stream; tensor contents may change between executions.
    """
    if q_nope.ndim != 4 or q_nope.shape[1] not in (1, 2) or q_nope.shape[-1] != 512:
        raise NotImplementedError("requires Q [batch, 1 or 2, heads, 512]")
    batch, queries, heads, _ = q_nope.shape
    if heads not in (0, 16, 32) and (heads < 64 or heads % 64):
        raise NotImplementedError("requires 16, 32, or a multiple of 64 query heads")
    if (
        k_nope.ndim not in (3, 4)
        or k_nope.shape[1] != 64
        or k_nope.shape[-1] != 512
        or (k_nope.ndim == 4 and k_nope.shape[2] != 1)
    ):
        raise NotImplementedError("requires KV [pages,64,512] or [pages,64,1,512]")
    if q_nope.dtype != torch.int8 or k_nope.dtype != torch.int8:
        raise TypeError("Q and KV content must be INT8")
    if (
        q_rope.dtype not in (torch.bfloat16, torch.float16)
        or k_rope.dtype != q_rope.dtype
    ):
        raise TypeError("RoPE must have matching BF16 or FP16 dtype")
    if q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32:
        raise TypeError("scales must be FP32")
    if (
        q_rope.ndim != 4
        or q_rope.shape[-1] not in (0, 64)
        or q_rope.shape[:-1] != q_nope.shape[:-1]
        or k_rope.shape != (*k_nope.shape[:-1], q_rope.shape[-1])
        or q_scale.shape != (*q_nope.shape[:-1], 1)
        or k_scale.shape != (*k_nope.shape[:-1], 1)
    ):
        raise ValueError("RoPE/scales must match content rows with width 0 or 64 / 1")
    if (
        indices.dtype != torch.int32
        or indices.ndim != 3
        or indices.shape[:2] != (batch, queries)
        or indices.shape[-1] > 65536
    ):
        raise ValueError("indices must be INT32 [batch, queries, topk <= 65536]")
    if topk_length is not None and (
        topk_length.shape != (batch, queries) or topk_length.dtype != torch.int32
    ):
        raise ValueError("topk_length must be INT32 [batch, queries]")
    if attn_sink is not None and (
        attn_sink.shape != (heads,) or attn_sink.dtype != torch.float32
    ):
        raise ValueError("attn_sink must be FP32 [heads]")
    if out is not None and (
        out.shape != (batch, queries, heads, 512) or out.dtype != q_rope.dtype
    ):
        raise ValueError("out must match output shape and RoPE dtype")
    inputs = (q_nope, q_rope, k_nope, k_rope, q_scale, k_scale, indices)
    optional = tuple(t for t in (topk_length, attn_sink) if t is not None)
    tensors = inputs + optional + ((out,) if out is not None else ())
    if q_nope.device.type != "musa" or any(t.device != q_nope.device for t in tensors):
        raise ValueError("all inputs must share a MUSA device")
    if any(not t.is_contiguous() for t in tensors):
        raise NotImplementedError("requires contiguous inputs and output")
    if (
        out is not None
        and out.numel()
        and any(
            out.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
            for t in inputs + optional
        )
    ):
        raise ValueError("out must not alias an input")
    if softmax_scale is not None and (
        isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (float, int))
    ):
        raise TypeError("softmax_scale must be a Python number")
    scale = (
        (512 + q_rope.shape[-1]) ** -0.5
        if softmax_scale is None
        else float(softmax_scale)
    )
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be positive and finite")
    rows, topk = batch * queries, indices.shape[-1]
    splits = max(
        1,
        min(
            32,
            triton.cdiv(topk, 256),
            triton.cdiv(256, max(1, rows * triton.cdiv(heads, 32))),
        ),
    )
    chunk = triton.cdiv(topk, splits * 64) * 64
    if rows * splits * heads * 2056 > 8 * 1024**3 or k_nope.shape[0] * 64 >= 2**31:
        raise NotImplementedError(
            "workspace exceeds 8 GiB or cache exceeds INT32 indices"
        )
    output = (
        torch.empty(q_nope.shape, dtype=q_rope.dtype, device=q_nope.device)
        if out is None
        else out
    )
    lse = torch.empty(
        (batch, heads, queries), dtype=torch.float32, device=q_nope.device
    )
    if topk >= 4096:
        if rows * heads * topk * 6 > 8 * 1024**3:
            raise NotImplementedError("long sparse attention workspace exceeds 8 GiB")
        scores = torch.empty(
            (rows, heads, topk), dtype=torch.float32, device=q_nope.device
        )
        high = torch.empty((rows, heads, topk), dtype=torch.int8, device=q_nope.device)
        low = torch.empty_like(high)
        scales = torch.empty((rows, heads), dtype=torch.float32, device=q_nope.device)

        def run_long() -> tuple[torch.Tensor, torch.Tensor]:
            if rows == 0 or heads == 0:
                return output, lse
            with runtime.torch_device_fn.device(q_nope.device):
                sparse_int8_qk[
                    lambda meta: (
                        rows,
                        triton.cdiv(heads, meta["BH"]),
                        triton.cdiv(topk, meta["BN"]),
                    )
                ](
                    *inputs[:6],
                    indices,
                    topk_length,
                    scores,
                    rows,
                    heads,
                    topk,
                    k_nope.shape[0] * 64,
                    q_rope.shape[-1],
                    scale,
                    topk_length is not None,
                )
                sparse_int8_probabilities[(rows, heads)](
                    scores,
                    k_scale,
                    indices,
                    high,
                    low,
                    scales,
                    attn_sink,
                    lse,
                    heads,
                    queries,
                    topk,
                    k_nope.shape[0] * 64,
                    attn_sink is not None,
                    triton.next_power_of_2(topk),
                    num_warps=8,
                )
                sparse_int8_pv[
                    lambda meta: (
                        rows,
                        triton.cdiv(heads, meta["BH"]),
                        512 // meta["BV"],
                    )
                ](
                    high,
                    low,
                    scales,
                    k_nope,
                    indices,
                    output,
                    rows,
                    heads,
                    topk,
                    k_nope.shape[0] * 64,
                )
            return output, lse

        return run_long, run_long()

    partial = torch.empty(
        (rows, splits, heads, 512), dtype=torch.float32, device=q_nope.device
    )
    stats = torch.empty(
        (rows, splits, heads, 2), dtype=torch.float32, device=q_nope.device
    )

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        if rows == 0 or heads == 0:
            return output, lse
        with runtime.torch_device_fn.device(q_nope.device):
            sparse_int8_mla[
                lambda meta: (rows, triton.cdiv(heads, meta["BH"]), splits)
            ](
                *inputs[:6],
                indices,
                topk_length,
                partial,
                stats,
                rows,
                heads,
                topk,
                k_nope.shape[0] * 64,
                q_rope.shape[-1],
                scale,
                topk_length is not None,
                splits,
                chunk,
            )
            combine_sparse_int8[(rows, heads, 4)](
                partial,
                stats,
                attn_sink,
                output,
                lse,
                heads,
                queries,
                splits,
                attn_sink is not None,
                triton.next_power_of_2(splits),
                num_warps=4,
            )
        return output, lse

    return run, run()


def flash_mla_with_kvcache_sparse_w8a8_int8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, output = prepare_flash_mla_with_kvcache_sparse_w8a8_int8(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_scale,
        k_scale,
        indices,
        softmax_scale=softmax_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        out=out,
    )
    return output
