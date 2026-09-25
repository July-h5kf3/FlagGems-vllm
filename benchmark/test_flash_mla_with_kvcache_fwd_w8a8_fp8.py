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

import pytest
import torch

from flaggems_vllm.ops.flash_mla import HAS_TLE_FLASH_MLA as HAS_TLE
from flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8 import (
    prepare_flash_mla_with_kvcache_fwd_w8a8_fp8,
    quantize_k_ckv_per_token,
    quantize_q_ckv_per_token,
)

from . import base
from .test_flash_mla_with_kvcache import (
    HAS_CUDA_FLASHMLA,
    FlashMLAWithKVCacheBenchmark,
    _cuda_wrapper,
)

_PREPARED = {}


def _vllm_bf16_q_fp8_kv(
    q,
    packed_cache,
    q_nope,
    q_rope,
    q_scale,
    k_lora,
    k_rope,
    k_scale,
    block_table,
    cache_seqlens,
    indices,
    lengths,
    head_dim_v,
    *,
    causal,
):
    del q_nope, q_rope, q_scale, k_lora, k_rope, k_scale, block_table, lengths
    assert causal
    # vLLM's dense FP8 kernel requires FP8 Q; its sparse kernel supports BF16 Q.
    return _cuda_wrapper(
        q,
        packed_cache,
        None,
        None,
        head_dim_v,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
    )


def _fp8(
    q,
    packed_cache,
    q_nope,
    q_rope,
    q_scale,
    k_lora,
    k_rope,
    k_scale,
    block_table,
    cache_seqlens,
    indices,
    lengths,
    head_dim_v,
    *,
    causal,
):
    del q, packed_cache, indices
    key = (int(q_nope.data_ptr()), int(k_lora.data_ptr()), lengths)
    prepared = _PREPARED.get(key)
    if prepared is None:
        # Retain only the active shape's KV descriptors and workspace.
        _PREPARED.clear()
        handle, (out, lse) = prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
            q_nope,
            q_rope,
            k_lora,
            k_rope,
            q_scale,
            k_scale,
            block_table,
            cache_seqlens,
            head_dim_v,
            causal=causal,
            initial_cache_seqlens=lengths,
            max_cache_seqlens=lengths,
        )
        prepared = (handle, out, lse)
        _PREPARED[key] = prepared
    handle, out, lse = prepared
    return handle(out=out, lse=lse)


class FlashMLAWithKVCacheFP8Benchmark(FlashMLAWithKVCacheBenchmark):
    def __init__(self):
        base.Benchmark.__init__(
            self,
            "flash_mla_with_kvcache_fwd_w8a8_fp8",
            _vllm_bf16_q_fp8_kv,
            [torch.bfloat16],
        )
        self.set_gems(_fp8)

    @staticmethod
    def get_performance_test_params():
        return [
            param
            for param in FlashMLAWithKVCacheBenchmark.get_performance_test_params()
            if param.topk == 0 and param.d_qk == 576
        ]

    def get_input_iter(self, dtype):
        for inputs in super().get_input_iter(dtype):
            args, kwargs = self.unpack_to_args_kwargs(inputs)
            reference, reference_lse = _vllm_bf16_q_fp8_kv(*args, **kwargs)
            output, lse = _fp8(*args, **kwargs)
            relative_l2 = (output.float() - reference.float()).norm() / (
                reference.float().norm().clamp_min(1e-12)
            )
            assert relative_l2.item() < 0.05
            torch.testing.assert_close(lse, reference_lse, atol=0.025, rtol=0.002)
            yield inputs

    @staticmethod
    def make_input(param):
        for (
            q,
            cache,
            block_table,
            cache_seqlens,
            head_dim_v,
            kwargs,
        ) in FlashMLAWithKVCacheBenchmark.make_input(param):
            # The BF16-Q/FP8-KV CUDA path has a fixed top-k for every request.
            cache_seqlens.fill_(param.seqlen)
            q_nope, q_rope, q_scale = quantize_q_ckv_per_token(q)
            k_lora, k_rope, k_scale = quantize_k_ckv_per_token(cache)
            packed_cache = torch.empty(
                (*cache.shape[:-1], 656), device=cache.device, dtype=torch.uint8
            )
            packed_cache[..., :512].copy_(k_lora.view(torch.uint8))
            packed_cache[..., 512:528].copy_(
                k_scale.expand(*k_scale.shape[:-1], 4).contiguous().view(torch.uint8)
            )
            packed_cache[..., 528:].copy_(
                cache[..., 512:].contiguous().view(torch.uint8)
            )
            page_offsets = torch.arange(
                cache.shape[1], device=cache.device, dtype=torch.int32
            )
            pages_per_request = param.seqlen // cache.shape[1]
            indices = (
                block_table[:, :pages_per_request, None] * cache.shape[1] + page_offsets
            ).reshape(q.shape[0], 1, -1)
            yield (
                q,
                packed_cache,
                q_nope,
                q_rope,
                q_scale,
                k_lora,
                k_rope,
                k_scale,
                block_table,
                cache_seqlens,
                indices,
                tuple(cache_seqlens.tolist()),
                head_dim_v,
                kwargs,
            )


@pytest.mark.skipif(
    not (HAS_TLE and HAS_CUDA_FLASHMLA and torch.cuda.is_available()),
    reason="requires Hopper, FlagTree TLE and vLLM FlashMLA CUDA",
)
@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_fp8
def test_flash_mla_with_kvcache_fwd_w8a8_fp8():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires an NVIDIA Hopper GPU")
    FlashMLAWithKVCacheFP8Benchmark().run()
