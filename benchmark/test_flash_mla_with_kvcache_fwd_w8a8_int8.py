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
import os
from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import pytest
import torch

import flaggems_vllm
from tests.accuracy_utils import gems_assert_close
from tests.test_flash_mla_with_kvcache_fwd_w8a8_int8 import (
    SUPPORTED,
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


class DenseInt8MLAInputs(NamedTuple):
    handle: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    query: torch.Tensor
    cache: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    logits: torch.Tensor
    scale: torch.Tensor
    table: torch.Tensor
    lengths: torch.Tensor
    sequence_length: int


def run_vllm_bf16_mla(inputs):
    vllm_mla_decode()(
        inputs.query,
        inputs.cache,
        inputs.cache[..., :512],
        inputs.output,
        inputs.lse,
        inputs.table,
        inputs.lengths,
        inputs.logits,
        4,
        576**-0.5,
        64,
        k_scale=inputs.scale,
        v_scale=inputs.scale,
    )
    return inputs.output, inputs.lse


def run_dense_int8_mla(inputs):
    return inputs.handle()


class DenseInt8MLABenchmark(base.Benchmark):
    DEFAULT_SHAPES = [(1, 64, 640), (8, 128, 8192)]
    DEFAULT_SHAPE_DESC = "batch, heads, cache_length"

    def __init__(self):
        super().__init__(
            "flash_mla_with_kvcache_fwd_w8a8_int8", run_vllm_bf16_mla, [torch.bfloat16]
        )
        self.set_gems(run_dense_int8_mla)

    def get_input_iter(self, dtype):
        for batch, heads, length in self.shapes:
            tensors = make_dense_int8_inputs(batch, heads, length, dtype)
            q, qr, kv, kr, qs, ks, table, lengths = tensors
            query = (torch.cat((q.float(), qr.float()), -1) * qs).to(dtype).squeeze(1)
            cache = (
                (torch.cat((kv.float(), kr.float()), -1) * ks).to(dtype).unsqueeze(2)
            )
            handle, _ = flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
                *tensors
            )
            inputs = DenseInt8MLAInputs(
                handle,
                query,
                cache,
                torch.empty((batch, heads, 512), dtype=dtype, device=q.device),
                torch.empty((batch, heads), dtype=torch.float32, device=q.device),
                torch.empty(
                    (batch, heads, 4, 513), dtype=torch.float32, device=q.device
                ),
                torch.ones((), dtype=torch.float32, device=q.device),
                # INT64 page IDs avoid vLLM 0.19's INT32 address overflow in large pools.
                table.to(torch.int64),
                lengths,
                length,
            )
            reference, reference_lse = run_vllm_bf16_mla(inputs)
            output, lse = run_dense_int8_mla(inputs)
            gems_assert_close(
                output[:, 0].float(), reference.float(), torch.float32, atol=0.002
            )
            gems_assert_close(lse[..., 0], reference_lse, torch.float32, atol=0.002)
            yield (inputs,)

    def record_shapes(self, inputs):
        return (inputs.query.shape[0], inputs.query.shape[1], inputs.sequence_length)


@pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8
@pytest.mark.skipif(
    not SUPPORTED, reason="backend has no dense INT8 MLA implementation"
)
def test_flash_mla_with_kvcache_fwd_w8a8_int8():
    DenseInt8MLABenchmark().run()
