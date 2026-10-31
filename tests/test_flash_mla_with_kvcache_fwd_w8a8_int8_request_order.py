# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest
import torch

import flaggems_vllm

from .accuracy_utils import gems_assert_close
from .test_flash_mla_with_kvcache_fwd_w8a8_int8 import (
    dense_int8_mla_reference,
    make_dense_int8_inputs,
)

OP_NAME = "flash_mla_with_kvcache_fwd_w8a8_int8"
VENDOR_MODULE = "flaggems_vllm.runtime.backend._thead.ops." + OP_NAME
CUSTOMIZED_OPS = dict(flaggems_vllm.runtime.backend.get_customized_ops())
SUPPORTED = (
    OP_NAME in CUSTOMIZED_OPS and CUSTOMIZED_OPS[OP_NAME].__module__ == VENDOR_MODULE
)
pytestmark = [
    pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8,
    pytest.mark.skipif(not SUPPORTED, reason="backend has no PPU request-order path"),
]


@pytest.mark.parametrize(
    "batch,heads,queries,splits,varlen",
    [(128, 16, 1, 1, True), (65, 17, 2, 4, True), (128, 16, 1, 1, False)],
)
def test_flash_mla_int8_request_order(
    monkeypatch, batch, heads, queries, splits, varlen
):
    vendor = importlib.import_module(VENDOR_MODULE)
    kernel = vendor.dense_int8_mla_kernel.jit_function

    class SortedKernel:
        def __getitem__(self, grid):
            def launch(*args):
                return kernel[grid({"BH": 16})](
                    *args,
                    BH=16,
                    BK=128,
                    FUSED_PV=True,
                    PAIR_PV=True,
                    QK_SPLITS=4,
                    COLUMN_SOFTMAX=False,
                    SORT_REQUESTS=True,
                    num_warps=8,
                    num_stages=1,
                )

            return launch

    monkeypatch.setattr(vendor, "dense_int8_mla_kernel", SortedKernel())
    tensors = make_dense_int8_inputs(
        batch, heads, 257, queries=queries, strided=True, seed=7531
    )
    if varlen:
        lengths = [(row * 73) % 258 for row in range(batch)]
    else:
        lengths = [257] * batch
    tensors[7].copy_(
        torch.tensor(
            lengths,
            dtype=torch.int32,
            device=tensors[7].device,
        )
    )
    output, lse = flaggems_vllm.flash_mla_with_kvcache_fwd_w8a8_int8(
        *tensors, num_splits=splits, causal=True
    )
    expected, expected_lse = dense_int8_mla_reference(tensors, causal=True)
    gems_assert_close(output.float(), expected, torch.float32, atol=0.002)
    gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)
