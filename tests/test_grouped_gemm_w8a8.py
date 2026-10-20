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

import pytest
import torch
import triton

import flaggems_vllm
from flaggems_vllm.testing.grouped_gemm import (
    deepgemm_grouped_gemm,
    grouped_gemm_cpu_reference,
    make_grouped_gemm_case,
)

from .accuracy_utils import gems_assert_close, gems_assert_equal

pytestmark = [
    pytest.mark.grouped_gemm_w8a8,
    pytest.mark.skipif(
        "grouped_gemm_w8a8" not in vars(flaggems_vllm),
        reason="requires a registered PPU grouped INT8 GEMM implementation",
    ),
]


@pytest.mark.parametrize(
    "lengths,n,k",
    [
        ([0, 1, 3, 0, 7], 65, 129),
        ([1, 32, 0, 65], 128, 256),
        ([0, 257, 3], 64, 128),
        ([1] + [0] * 255 + [13], 64, 6144),
        ([0, 1, 0], 17, 1),
        ([0, 0, 0], 64, 128),
    ],
)
def test_grouped_gemm_w8a8(lengths, n, k):
    args = make_grouped_gemm_case(lengths, n, k, flaggems_vllm.device)
    expected = grouped_gemm_cpu_reference(*args[:4], lengths)
    flaggems_vllm.grouped_gemm_w8a8(*args)
    gems_assert_equal(args[4].cpu(), expected)


@pytest.mark.parametrize("n,k", [(512, 6144), (6144, 256)])
def test_grouped_gemm_w8a8_deepgemm(n, k):
    pytest.importorskip("deep_gemm")
    lengths = [1] * 32 + [0] * 224 + [13]
    args = make_grouped_gemm_case(lengths, n, k, flaggems_vllm.device)
    expected = torch.empty_like(args[4])
    deepgemm_grouped_gemm(*args[:4], expected, *args[5:])
    flaggems_vllm.grouped_gemm_w8a8(*args)
    gems_assert_close(args[4], expected, torch.bfloat16, atol=1e-3)


def test_grouped_gemm_w8a8_int8_extremes():
    args = make_grouped_gemm_case([1, 0, 1], 65, 6144, flaggems_vllm.device)
    args[0].fill_(-128)
    args[2][0].fill_(-128)
    args[2][2].fill_(127)
    expected = grouped_gemm_cpu_reference(*args[:4], [1, 0, 1])
    flaggems_vllm.grouped_gemm_w8a8(*args)
    gems_assert_equal(args[4].cpu(), expected)


def test_grouped_gemm_w8a8_changing_expert_rows():
    for lengths in ([0, 257, 0, 511], [256, 0, 512, 0]):
        args = make_grouped_gemm_case(lengths, 64, 128, flaggems_vllm.device)
        expected = grouped_gemm_cpu_reference(*args[:4], lengths)
        flaggems_vllm.grouped_gemm_w8a8(*args)
        gems_assert_equal(args[4].cpu(), expected)


@pytest.mark.parametrize(
    "async_a,async_b,block_m,block_k,swap,stages",
    [(True, False, 64, 128, False, 3), (False, True, 128, 64, True, 4)],
)
@pytest.mark.parametrize("values", ["random", "extreme", "cancellation"])
def test_grouped_gemm_w8a8_async(
    async_a, async_b, block_m, block_k, swap, stages, values
):
    module = importlib.import_module(flaggems_vllm.grouped_gemm_w8a8.__module__)
    lengths = [0, 17, 65, 0, 131, 1]
    n, k = 128, 256 if values == "random" else 6144
    args = make_grouped_gemm_case(lengths, n, k, flaggems_vllm.device)
    if values != "random":
        args[0].fill_(-128)
        args[2].fill_(-128)
        if values == "cancellation":
            # Preserve an odd residual after partial sums exceed FP32's exact range.
            args[0].fill_(127)
            args[2].fill_(127)
            args[0][:, k // 2 :] = -127
            args[0][:, k // 3] = 126
    expected = grouped_gemm_cpu_reference(*args[:4], lengths)
    m, e = sum(lengths), len(lengths)
    workspace = (
        torch.empty(m + e * (block_m - 1), dtype=torch.int32, device=args[0].device),
        torch.empty(
            triton.cdiv(m, block_m) + e, dtype=torch.int32, device=args[0].device
        ),
        torch.empty(1, dtype=torch.int32, device=args[0].device),
    )
    module.grouped_gemm_metadata[(e,)](
        *args[5:],
        *workspace,
        m,
        e,
        BLOCK_M=block_m,
        BLOCK_E=triton.next_power_of_2(e),
        num_warps=4,
    )
    tiles_m = triton.cdiv(triton.cdiv(m, block_m) + e - 1, 8) * 8
    module.grouped_gemm_tiled_kernel[(tiles_m,)](
        *args,
        *workspace,
        m,
        n,
        k,
        e,
        triton.next_power_of_2(e),
        BLOCK_M=block_m,
        BLOCK_N=128,
        BLOCK_K=block_k,
        WARPS=4,
        STAGES=stages,
        ASYNC_A=async_a,
        ASYNC_B=async_b,
        SCALAR_PROBE=True,
        SWAP=swap,
        TLE_PIPELINE=True,
        NAIVE=False,
        GROUP_M=8,
        ADAPTIVE_M=True,
        num_warps=4,
        num_stages=stages,
    )
    gems_assert_equal(args[4].cpu(), expected)


@pytest.mark.parametrize("cache,evict", [(".cg", ""), (".ca", "evict_last")])
def test_grouped_gemm_w8a8_gemv_cache_tail(cache, evict):
    module = importlib.import_module(flaggems_vllm.grouped_gemm_w8a8.__module__)
    lengths = [1, 0, 8, 3]
    n, k = 65, 260
    args = make_grouped_gemm_case(lengths, n, k, flaggems_vllm.device)
    expected = grouped_gemm_cpu_reference(*args[:4], lengths)
    module.grouped_gemv_int8_kernel[(triton.cdiv(n, 8), sum(lengths))](
        *args[:6],
        sum(lengths),
        n,
        k,
        len(lengths),
        BLOCK_N=8,
        BLOCK_K4=32,
        WARPS=1,
        STAGES=1,
        TLE_PIPELINE=False,
        CACHE_B=cache,
        EVICT_B=evict,
        num_warps=1,
        num_stages=1,
    )
    gems_assert_equal(args[4].cpu(), expected)
