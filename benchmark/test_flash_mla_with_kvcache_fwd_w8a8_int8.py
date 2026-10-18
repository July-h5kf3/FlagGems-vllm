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

import importlib.util
import math
import os
import random
from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import pytest
import torch

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close
from tests.test_flash_mla_with_kvcache_fwd_w8a8_int8 import (
    SUPPORTED,
    dense_int8_mla_reference,
    dense_mla_reference,
    make_dense_int8_inputs,
)

from . import base


@lru_cache(maxsize=1)
def vllm_mla_decode():
    reference_path = os.environ.get("FLAGGEMS_MLA_REFERENCE_PATH")
    if reference_path:
        spec = importlib.util.spec_from_file_location(
            "vllm_mla_reference", reference_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.decode_attention_fwd
    from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

    return decode_attention_fwd


# The unchanged PR851 upstream matrix includes both draft lengths and ragged caches.
UPSTREAM_SHAPES = [
    (batch, queries, length, heads, varlen)
    for batch in [128]
    for length in [4096, 8192, 16384]
    for heads in [16, 32, 64, 128]
    for queries in [1, 2]
    for varlen in [False, True]
]


class DenseInt8MLAInputs(NamedTuple):
    handle: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    query: torch.Tensor
    cache: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    logits: torch.Tensor
    scale: torch.Tensor
    table: torch.Tensor
    reference_lengths: tuple[torch.Tensor, ...]
    shape: tuple[int, int, int, int, bool]


def run_vllm_bf16_mla(inputs):
    for query_index, lengths in enumerate(inputs.reference_lengths):
        vllm_mla_decode()(
            inputs.query[:, query_index],
            inputs.cache,
            inputs.cache[..., :512],
            inputs.output[query_index],
            inputs.lse[query_index],
            inputs.table,
            lengths,
            inputs.logits,
            4,
            576**-0.5,
            64,
            k_scale=inputs.scale,
            v_scale=inputs.scale,
        )
    return inputs.output.permute(1, 0, 2, 3), inputs.lse.permute(1, 2, 0)


def run_dense_int8_mla(inputs):
    return inputs.handle()


class DenseInt8MLABenchmark(base.Benchmark):
    DEFAULT_SHAPES = UPSTREAM_SHAPES
    DEFAULT_SHAPE_DESC = "batch, queries, length, heads, varlen"

    def __init__(self):
        super().__init__(
            "flash_mla_with_kvcache_fwd_w8a8_int8", run_vllm_bf16_mla, [torch.bfloat16]
        )
        self.set_gems(run_dense_int8_mla)

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            batch, queries, mean_length, heads, varlen = shape
            lengths = [mean_length] * batch
            if varlen:
                generator = random.Random(0)
                for upstream_shape in UPSTREAM_SHAPES:
                    source_batch, source_queries, source_mean, _, source_varlen = (
                        upstream_shape
                    )
                    if not source_varlen:
                        continue
                    lengths = [
                        max(
                            int(generator.normalvariate(source_mean, source_mean / 2)),
                            source_queries,
                        )
                        for _ in range(source_batch)
                    ]
                    if tuple(upstream_shape) == tuple(shape):
                        break
                else:
                    raise ValueError("shape must belong to the PR851 matrix")
            padded_length = math.ceil(max(lengths) / 256) * 256
            tensors = make_dense_int8_inputs(
                batch,
                heads,
                padded_length,
                dtype,
                queries=queries,
                magnitude=1.0,
                seed=0,
            )
            q, qr, kv, kr, qs, ks, table, gpu_lengths = tensors
            table = table.flip(1)
            gpu_lengths.copy_(torch.tensor(lengths, dtype=torch.int32, device=q.device))
            # Poison unused cache positions as the upstream reference does.
            for row, length in enumerate(lengths):
                kr.view(batch, padded_length, 64)[row, length:] = float("nan")
                ks.view(batch, padded_length, 1)[row, length:] = float("nan")
            tensors = (q, qr, kv, kr, qs, ks, table, gpu_lengths)
            query = (torch.cat((q.float(), qr.float()), -1) * qs).to(dtype)
            cache = (
                (torch.cat((kv.float(), kr.float()), -1) * ks).to(dtype).unsqueeze(2)
            )
            handle, _ = flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
                *tensors, causal=True
            )
            reference_lengths = tuple(
                torch.tensor(
                    [length - queries + query_index + 1 for length in lengths],
                    dtype=torch.int32,
                    device=q.device,
                )
                for query_index in range(queries)
            )
            inputs = DenseInt8MLAInputs(
                handle,
                query,
                cache,
                torch.empty((queries, batch, heads, 512), dtype=dtype, device=q.device),
                torch.empty(
                    (queries, batch, heads), dtype=torch.float32, device=q.device
                ),
                torch.empty(
                    (batch, heads, 4, 513), dtype=torch.float32, device=q.device
                ),
                torch.ones((), dtype=torch.float32, device=q.device),
                table.to(torch.int64),
                reference_lengths,
                tuple(shape),
            )
            reference, reference_lse = run_vllm_bf16_mla(inputs)
            output, lse = run_dense_int8_mla(inputs)
            expected, expected_lse = dense_int8_mla_reference(tensors, causal=True)
            gems_assert_close(output, expected, dtype, atol=0.002)
            gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)
            baseline_expected, baseline_lse = dense_mla_reference(
                query, cache, table, gpu_lengths, causal=True
            )
            gems_assert_close(reference, baseline_expected, dtype, atol=0.002)
            gems_assert_close(reference_lse, baseline_lse, torch.float32, atol=2e-4)
            yield (inputs,)

    def record_shapes(self, inputs):
        return inputs.shape


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(
    not SUPPORTED, reason="backend has no dense INT8 MLA implementation"
)
def test_flash_mla_with_kvcache_fwd_w8a8_int8():
    DenseInt8MLABenchmark().run()
