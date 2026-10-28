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

import importlib
import importlib.util
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable, NamedTuple

import pytest
import torch

import flaggems_vllm
from flaggems_vllm.ops.flash_mla_with_kvcache_w8a8 import (
    HAS_TLE,
    get_mla_fp8_metadata,
    prepare_flash_mla_with_kvcache_fwd_w8a8_fp8,
)

from .accuracy_utils import gems_assert_close, gems_assert_equal, to_reference

OUTPUT_ATOL = 2e-3


class DenseW8A8Inputs(NamedTuple):
    q_nope: torch.Tensor
    q_rope: torch.Tensor
    k_nope: torch.Tensor
    k_rope: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    block_table: torch.Tensor
    lengths: torch.Tensor
    value_dim: int
    native_metadata: torch.Tensor | None
    native_splits: torch.Tensor | None
    host_lengths: tuple[int, ...]
    native_query: torch.Tensor | None = None
    native_cache: torch.Tensor | None = None
    benchmark_shape: tuple = ()


class PreparedCall(NamedTuple):
    run: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    output: torch.Tensor
    lse: torch.Tensor


class CapturedCall(NamedTuple):
    replay: Callable[[], None]
    output: torch.Tensor
    lse: torch.Tensor


def quantize_dense_fp8(x, *, power_of_two=False, dequantize=True):
    scale = x[..., :512].float().abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    if power_of_two:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    nope = (x[..., :512].float() / scale).to(torch.float8_e4m3fn)
    rope = (x[..., 512:].float() / scale).to(torch.bfloat16)
    if dequantize:
        native = (torch.cat((nope.float(), rope.float()), -1) * scale).to(
            torch.bfloat16
        )
    else:
        native = None
    return nope, rope, scale, native


def dense_fp8_paged_reference(
    q_nope,
    q_rope,
    k_nope,
    k_rope,
    q_scale,
    k_scale,
    block_table,
    lengths,
    *,
    softmax_scale=None,
    causal=False,
):
    batch, queries, heads, _ = q_nope.shape
    pages = block_table.shape[1]
    capacity = pages * 64
    scale = 576**-0.5 if softmax_scale is None else softmax_scale
    output = torch.empty(
        (*q_nope.shape[:-1], 512), dtype=q_rope.dtype, device=q_nope.device
    )
    lse = torch.empty(
        (batch, heads, queries), dtype=torch.float32, device=q_nope.device
    )
    # Convert before indexing so the reference also works without FP8 gather support.
    cache = torch.cat((k_nope.float(), k_rope.float()), -1).reshape(-1, 64, 576)
    cache_scale = k_scale.reshape(-1, 64)
    tokens = torch.arange(capacity, device=q_nope.device)
    query_tokens = torch.arange(queries, device=q_nope.device)
    for begin in range(0, batch, 4):
        end = min(begin + 4, batch)
        page_ids = block_table[begin:end].long()
        values = cache[page_ids].reshape(end - begin, capacity, 576)
        valid_token = tokens[None, :] < lengths[begin:end, None]
        values = values.masked_fill(~valid_token[:, :, None], 0)
        ks = cache_scale[page_ids].reshape(end - begin, capacity)
        ks = ks.masked_fill(~valid_token, 0)
        query = torch.cat((q_nope[begin:end].float(), q_rope[begin:end].float()), -1)
        scores = torch.einsum("bqhd,bnd->bqhn", query, values)
        scores = scores * q_scale[begin:end] * ks[:, None, None, :] * scale
        valid = valid_token[:, None, None, :]
        if causal:
            valid = valid & (
                tokens[None, None, None, :]
                < (
                    lengths[begin:end, None, None, None]
                    - queries
                    + query_tokens[None, :, None, None]
                    + 1
                )
            )
        scores = scores.masked_fill(~valid, -torch.inf)
        logsum = torch.logsumexp(scores, -1)
        has_tokens = valid.any(-1, keepdim=True)
        probabilities = torch.where(has_tokens, scores.softmax(-1), 0)
        weights = (probabilities * ks[:, None, None, :]).reshape(
            end - begin, queries, heads, pages, 64
        )
        p_scale = (weights.abs().amax(-1, keepdim=True) / 448.0).clamp_min(1e-30)
        # Per-page P quantization is invariant to the online softmax row scale.
        quantized = (weights / p_scale).to(torch.float8_e4m3fn).float() * p_scale
        result = torch.einsum(
            "bqhn,bnd->bqhd", quantized.flatten(-2), values[..., :512]
        )
        output[begin:end] = result.to(q_rope.dtype)
        lse[begin:end] = logsum.transpose(1, 2)
    return output, lse


def assert_dense_fp8_close(actual, reference, *, output_atol):
    if reference[0].device.type == "cpu":
        actual = tuple(tensor.cpu() for tensor in actual)
    gems_assert_close(actual[0], reference[0], torch.bfloat16, atol=output_atol)
    gems_assert_close(actual[1], reference[1], torch.float32, atol=2e-4)


def assert_dense_fp8_replay(run, reference, *, output_atol, capture):
    assert_dense_fp8_close(run(), reference, output_atol=output_atol)
    captured = capture(run)
    captured.output.fill_(float("nan"))
    captured.lse.fill_(float("nan"))
    captured.replay()
    flaggems_vllm.runtime.torch_device_fn.synchronize()
    assert_dense_fp8_close(
        (captured.output, captured.lse), reference, output_atol=output_atol
    )


def supported() -> bool:
    return (
        HAS_TLE
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] == 9
    )


@lru_cache(maxsize=1)
def dense_cuda_reference():
    source_path = os.environ.get("FLAGGEMS_FLASHMLA_REFERENCE_PATH")
    if not source_path:
        try:
            return importlib.import_module(
                "vllm.third_party.flashmla.flash_mla_interface"
            )
        except ImportError as error:
            raise RuntimeError(
                "Official vLLM CUDA FlashMLA is required; set "
                "FLAGGEMS_FLASHMLA_REFERENCE_PATH to its unmodified interface"
            ) from error
    source = Path(source_path).resolve()
    package_root = source.parents[2]
    created_parent = "vllm" not in sys.modules
    if created_parent:
        parent = ModuleType("vllm")
        parent.__path__ = [str(package_root)]
        sys.modules["vllm"] = parent
    for module_name in ("_flashmla_C", "_flashmla_extension_C"):
        qualified = f"vllm.{module_name}"
        if qualified in sys.modules:
            continue
        candidates = sorted(package_root.glob(f"{module_name}*.so"))
        if not candidates:
            raise RuntimeError(f"Missing CUDA reference extension: {qualified}")
        spec = importlib.util.spec_from_file_location(qualified, candidates[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    name = "flaggems_vllm_test_dense_flashmla_reference"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if created_parent:
        del sys.modules["vllm"]
    return module


def _dense_cuda_reference_or_skip():
    try:
        return dense_cuda_reference()
    except (ImportError, RuntimeError):
        pytest.skip(
            "vLLM CUDA FlashMLA reference is unavailable; set "
            "FLAGGEMS_FLASHMLA_REFERENCE_PATH to its unmodified interface"
        )


def run_cuda(q, k_cache, block_table, cache_seqlens, head_dim_v, **kwargs):
    reference = _dense_cuda_reference_or_skip()
    metadata, _ = reference.get_mla_metadata()
    return reference.flash_mla_with_kvcache(
        q, k_cache, block_table, cache_seqlens, head_dim_v, metadata, **kwargs
    )


def functional_reference(inputs: DenseW8A8Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    return run_cuda(
        inputs.native_query,
        inputs.native_cache,
        inputs.block_table,
        inputs.lengths,
        inputs.value_dim,
        causal=True,
    )


def prepare_baseline(inputs: DenseW8A8Inputs) -> DenseW8A8Inputs:
    from flash_mla_fp8 import get_mla_metadata

    metadata, splits = get_mla_metadata(
        inputs.lengths, inputs.q_nope.shape[1] * inputs.q_nope.shape[2], 1
    )
    return inputs._replace(native_metadata=metadata, native_splits=splits)


def run_baseline(inputs: DenseW8A8Inputs) -> tuple[torch.Tensor, torch.Tensor]:
    from flash_mla_fp8 import flash_mla_ckv_fp8_per_token

    return flash_mla_ckv_fp8_per_token(*inputs[:11], causal=True)


def prepare_gems(inputs: DenseW8A8Inputs) -> PreparedCall:
    handle, (output, lse) = prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
        *inputs[:9],
        causal=True,
        initial_cache_seqlens=inputs.host_lengths,
        max_cache_seqlens=inputs.host_lengths,
    )
    return PreparedCall(lambda: handle(out=output, lse=lse), output, lse)


def capture(run) -> CapturedCall:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, lse = run()
    torch.cuda.current_stream().wait_stream(stream)
    return CapturedCall(graph.replay, output, lse)


def validate_functional(
    inputs: DenseW8A8Inputs,
    actual: tuple[torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor],
) -> None:
    metadata, splits = get_mla_fp8_metadata(
        inputs.lengths, inputs.q_nope.shape[1] * inputs.q_nope.shape[2]
    )
    planned = flaggems_vllm.flash_mla_with_kvcache_w8a8(
        *inputs[:9], metadata, splits, causal=True
    )
    gems_assert_equal(planned[0], to_reference(actual[0]))
    gems_assert_equal(planned[1], to_reference(actual[1]))
    prepared = prepare_gems(inputs)
    assert_dense_fp8_replay(
        prepared.run, reference, output_atol=OUTPUT_ATOL, capture=capture
    )


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.parametrize(
    "batch,seqlen,varlen",
    [
        (4, 128, False),
        (4, 256, True),
        (2, 33280, False),
        (8, 33280, False),
        (16, 8192, False),
    ],
)
@pytest.mark.parametrize("seq_q", (1, 2))
@pytest.mark.parametrize("h_q", (16, 32, 64, 128))
def test_dense_decode(seq_q, h_q, batch, seqlen, varlen):
    if not supported():
        pytest.skip("FP8 MLA requires NVIDIA Hopper and TLE")
    d_qk = 576
    h_k = 1
    head_dim_v = 512
    page_block_size = 64

    torch.manual_seed(45)
    q = torch.randn(
        batch, seq_q, h_q, d_qk, dtype=torch.bfloat16, device=flaggems_vllm.device
    )
    max_pages_per_seq = math.ceil(seqlen / page_block_size) + 4
    total_pages = batch * max_pages_per_seq
    kv_cache = (
        torch.randn(
            total_pages,
            page_block_size,
            h_k,
            d_qk,
            dtype=torch.bfloat16,
            device=flaggems_vllm.device,
        )
        * 0.1
    )
    block_table = torch.arange(
        total_pages, dtype=torch.int32, device=flaggems_vllm.device
    ).view(batch, max_pages_per_seq)
    host_lengths = [seqlen] * batch
    if varlen:
        host_lengths[0] = seqlen // 2
        host_lengths[-1] = seqlen + page_block_size
    cache_seqlens = torch.tensor(
        host_lengths, dtype=torch.int32, device=flaggems_vllm.device
    )
    q_nope, q_rope, q_scale, native_q = quantize_dense_fp8(q)
    k_nope, k_rope, k_scale, native_cache = quantize_dense_fp8(kv_cache)
    inputs = DenseW8A8Inputs(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
        None,
        None,
        tuple(host_lengths),
        native_q,
        native_cache,
    )
    reference = tuple(to_reference(t) for t in functional_reference(inputs))
    actual = flaggems_vllm.flash_mla_with_kvcache_w8a8(*inputs[:9], causal=True)
    assert_dense_fp8_close(actual, reference, output_atol=OUTPUT_ATOL)
    validate_functional(inputs, actual, reference)
