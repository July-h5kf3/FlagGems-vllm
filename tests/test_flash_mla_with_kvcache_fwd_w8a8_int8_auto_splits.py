# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm

from .accuracy_utils import gems_assert_close
from .test_flash_mla_with_kvcache_fwd_w8a8_int8 import (
    dense_int8_mla_reference,
    make_dense_int8_inputs,
)
from .test_flash_mla_with_kvcache_fwd_w8a8_int8_request_order import SUPPORTED

pytestmark = [
    pytest.mark.flash_mla_with_kvcache_fwd_w8a8_int8,
    pytest.mark.skipif(
        not SUPPORTED, reason="backend has no dense INT8 MLA implementation"
    ),
]


@pytest.mark.parametrize("batch,varlen", [(65, False), (128, True)])
def test_flash_mla_int8_auto_splits(batch, varlen):
    tensors = make_dense_int8_inputs(
        batch, 16, 4097, ragged=varlen, strided=True, seed=9327
    )
    expected, expected_lse = dense_int8_mla_reference(tensors, causal=True)
    for splits in [None, 1, 4]:
        handle, (output, lse) = (
            flaggems_vllm.prepare_flash_mla_with_kvcache_fwd_w8a8_int8(
                *tensors, causal=True, num_splits=splits
            )
        )
        gems_assert_close(output, expected, torch.bfloat16, atol=0.002)
        gems_assert_close(lse, expected_lse, torch.float32, atol=2e-4)
        assert splits is None or handle.splits == splits
        replay, replay_lse = handle()
        gems_assert_close(replay, expected, torch.bfloat16, atol=0.002)
        gems_assert_close(replay_lse, expected_lse, torch.float32, atol=2e-4)
