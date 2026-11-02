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
from collections.abc import Callable
from functools import partial
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
    reference: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    shape: tuple[int, int, int, int, bool]


def run_native_bf16_mla(inputs):
    return inputs.reference()


def run_dense_int8_mla(inputs):
    return inputs.handle()


class DenseInt8MLABenchmark(base.Benchmark):
    DEFAULT_SHAPES = UPSTREAM_SHAPES
    DEFAULT_SHAPE_DESC = "batch, queries, length, heads, varlen"

    def __init__(self):
        super().__init__(
            "flash_mla_with_kvcache_fwd_w8a8_int8",
            run_native_bf16_mla,
            [torch.bfloat16],
        )
        self.set_gems(run_dense_int8_mla)

    def get_input_iter(self, dtype):
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

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
            metadata, _ = get_mla_metadata()
            inputs = DenseInt8MLAInputs(
                handle,
                partial(
                    flash_mla_with_kvcache,
                    query,
                    cache,
                    table,
                    gpu_lengths,
                    512,
                    metadata,
                    softmax_scale=576**-0.5,
                    causal=True,
                    out=torch.empty(
                        (batch, queries, heads, 512), dtype=dtype, device=q.device
                    ),
                ),
                tuple(shape),
            )
            reference, reference_lse = run_native_bf16_mla(inputs)
            output, lse = run_dense_int8_mla(inputs)
            expected, expected_lse = dense_int8_mla_reference(tensors, causal=True)
            gems_assert_close(output, expected, dtype, atol=0.002)
            gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)
            baseline_expected, baseline_lse = dense_mla_reference(
                query, cache, table, gpu_lengths, causal=True
            )
            # Native BF16 PV rounds P before the dot, affecting short-row cancellation.
            gems_assert_close(reference, baseline_expected, dtype, atol=0.004)
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
