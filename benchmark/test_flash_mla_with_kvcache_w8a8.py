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

import math
import random
from dataclasses import dataclass
from itertools import product

import pytest
import torch

import flaggems_vllm
from tests.test_flash_mla_with_kvcache_w8a8 import SUPPORTED, quantize_dense_fp8

from . import base
from .test_blas_perf_parallel import ParallelBenchmarkMixin

try:
    from flash_mla_fp8 import flash_mla_ckv_fp8_per_token as cuda_flash_mla
    from flash_mla_fp8 import get_mla_metadata as cuda_get_mla_metadata

    HAS_CUDA_FP8_FLASHMLA = True
except ImportError:
    HAS_CUDA_FP8_FLASHMLA = False


@dataclass
class TestParam:
    __test__ = False

    batch: int
    seq_q: int
    seqlen: int
    h_q: int
    varlen: bool


def cuda_wrapper(
    q_nope,
    q_rope,
    k_nope,
    k_rope,
    q_scale,
    k_scale,
    block_table,
    cache_seqlens,
    head_dim_v,
    **kwargs,
):
    metadata, splits = cuda_get_mla_metadata(
        cache_seqlens, q_nope.shape[1] * q_nope.shape[2], 1
    )
    return cuda_flash_mla(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        q_scale,
        k_scale,
        block_table,
        cache_seqlens,
        head_dim_v,
        metadata,
        splits,
        **kwargs,
    )


class FlashMLAWithKVCacheW8A8Benchmark(ParallelBenchmarkMixin, base.Benchmark):
    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        if not HAS_CUDA_FP8_FLASHMLA:
            raise RuntimeError("The native CUDA flash_mla_fp8 baseline is required")
        super().__init__("flash_mla_with_kvcache_w8a8", cuda_wrapper, [torch.bfloat16])
        self.set_gems(flaggems_vllm.flash_mla_with_kvcache_w8a8)

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from self.make_input(TestParam(*shape), dtype)

    @staticmethod
    def make_input(param: TestParam, dtype):
        torch.manual_seed(0)
        lengths = [param.seqlen] * param.batch
        if param.varlen:
            # Preserve the native reference's seeded length stream when selecting a shard.
            generator = random.Random(0)
            for mean, heads, queries in product(
                (4096, 8192, 16384), (16, 32, 64, 128), (1, 2)
            ):
                lengths = [
                    max(int(generator.normalvariate(mean, mean / 2)), queries)
                    for _ in range(128)
                ]
                if (param.batch, param.seqlen, param.h_q, param.seq_q) == (
                    128,
                    mean,
                    heads,
                    queries,
                ):
                    break
            else:
                raise ValueError("Variable-length shape must be from the native matrix")
        else:
            pass
        padded_length = math.ceil(max(lengths) / 256) * 256
        device = flaggems_vllm.device
        cache_seqlens = torch.tensor(lengths, dtype=torch.int32, device=device)
        query = torch.randn(
            param.batch, param.seq_q, param.h_q, 576, dtype=dtype, device=device
        )
        table = torch.arange(
            param.batch * padded_length // 64, dtype=torch.int32, device=device
        ).view(param.batch, padded_length // 64)
        cache = torch.randn(table.numel(), 64, 1, 576, dtype=dtype, device=device)
        for row, length in enumerate(lengths):
            cache.view(param.batch, padded_length, 1, 576)[row, length:] = float("nan")
        q_nope, q_rope, q_scale, _ = quantize_dense_fp8(query, dequantize=False)
        k_nope, k_rope, k_scale, _ = quantize_dense_fp8(cache, dequantize=False)
        yield (
            q_nope,
            q_rope,
            k_nope,
            k_rope,
            q_scale,
            k_scale,
            table,
            cache_seqlens,
            512,
            {"causal": True},
        )


@pytest.mark.flash_mla_with_kvcache_w8a8
@pytest.mark.skipif(not SUPPORTED, reason="FP8 MLA requires NVIDIA Hopper and TLE")
def test_flash_mla_with_kvcache_w8a8():
    FlashMLAWithKVCacheW8A8Benchmark().run()
