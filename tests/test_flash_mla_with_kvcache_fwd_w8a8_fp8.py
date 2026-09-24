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

from flaggems_vllm.ops.flash_mla import HAS_TLE_FLASH_MLA as HAS_TLE
from flaggems_vllm.ops.flash_mla_fp8.flash_mla_with_kvcache_fwd_w8a8_fp8 import (
    flash_mla_with_kvcache_fwd_w8a8_fp8,
    prepare_flash_mla_with_kvcache_fwd_w8a8_fp8,
)
from tests.flash_mla_fp8_utils import assert_dense_accuracy as _assert_close
from tests.flash_mla_fp8_utils import dense_reference as _reference
from tests.flash_mla_fp8_utils import make_dense_inputs as _make_inputs

from . import conftest as cfg

pytestmark = [
    pytest.mark.flash_mla_with_kvcache_fwd_w8a8_fp8,
    pytest.mark.skipif(
        not HAS_TLE
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 9,
        reason="requires Hopper and FlagTree GPU extensions",
    ),
]


CASES = [(1, 64, 128)] if cfg.QUICK_MODE else [(1, 64, 128), (2, 64, 640)]


@pytest.mark.parametrize("batch,h_q,seqlen", CASES)
def test_flash_mla_with_kvcache_fwd_w8a8_fp8_accuracy(batch, h_q, seqlen):
    inputs = _make_inputs(batch, h_q, seqlen)
    out, lse = flash_mla_with_kvcache_fwd_w8a8_fp8(
        inputs["q_nope"],
        inputs["q_rope"],
        inputs["k_lora"],
        inputs["k_rope"],
        inputs["q_scale"],
        inputs["k_scale"],
        inputs["block_table"],
        inputs["cache_seqlens"],
        512,
    )
    ref_out, ref_lse = _reference(inputs)
    _assert_close(out, lse, ref_out, ref_lse)


def test_flash_mla_with_kvcache_fwd_w8a8_fp8_prepared_outputs_are_deterministic():
    inputs = _make_inputs(1, 64, 128)
    handle, (fresh_out, fresh_lse) = prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
        inputs["q_nope"],
        inputs["q_rope"],
        inputs["k_lora"],
        inputs["k_rope"],
        inputs["q_scale"],
        inputs["k_scale"],
        inputs["block_table"],
        inputs["cache_seqlens"],
        512,
        initial_cache_seqlens=inputs["lengths"],
        max_cache_seqlens=inputs["lengths"],
    )
    out = torch.empty_like(fresh_out)
    lse = torch.empty_like(fresh_lse)
    caller_out, caller_lse = handle(out=out, lse=lse)
    expected_out = caller_out.clone()
    expected_lse = caller_lse.clone()
    replay_out, replay_lse = handle(out=out, lse=lse)

    assert torch.equal(fresh_out, expected_out)
    assert torch.equal(fresh_lse, expected_lse)
    assert torch.equal(replay_out, expected_out)
    assert torch.equal(replay_lse, expected_lse)


def test_dense_fp8_requires_compiler_support(monkeypatch):
    module = importlib.import_module(
        "flaggems_vllm.ops.flash_mla_fp8.flash_mla_with_kvcache_fwd_w8a8_fp8"
    )
    inputs = _make_inputs(1, 64, 128)
    monkeypatch.setattr(module, "HAS_TLE", False)
    with pytest.raises(NotImplementedError, match="FlagTree GPU extensions"):
        flash_mla_with_kvcache_fwd_w8a8_fp8(
            inputs["q_nope"],
            inputs["q_rope"],
            inputs["k_lora"],
            inputs["k_rope"],
            inputs["q_scale"],
            inputs["k_scale"],
            inputs["block_table"],
            inputs["cache_seqlens"],
            512,
        )


def test_fp8_reuses_bf16_tle_support():
    bf16 = importlib.import_module("flaggems_vllm.ops.flash_mla")
    dense = importlib.import_module(
        "flaggems_vllm.ops.flash_mla_fp8.flash_mla_with_kvcache_fwd_w8a8_fp8"
    )
    sparse = importlib.import_module(
        "flaggems_vllm.ops.flash_mla_fp8.flash_mla_sparse_fwd_w8a8_fp8"
    )
    assert (
        dense._ensure_triton_descriptor_allocator
        is bf16._ensure_triton_descriptor_allocator
    )
    assert dense._get_tensor_descriptor_cls is bf16._get_tensor_descriptor_cls
    assert dense._get_num_sms is bf16._get_num_sms
    assert sparse._get_num_sms is bf16._get_num_sms
    assert dense.HAS_TLE == sparse.HAS_TLE == bf16.HAS_TLE_FLASH_MLA


@pytest.mark.parametrize("batch", [1, 2])
def test_bf16_and_fp8_prepared_execution_interleave(batch):
    bf16 = importlib.import_module("flaggems_vllm.ops.flash_mla")
    inputs = _make_inputs(batch, 64, 128)
    expected, expected_lse = _reference(inputs)
    plan = bf16.get_flash_mla_tle_decode_plan(
        b=batch,
        s_q=1,
        h_q=64,
        h_kv=1,
        d=576,
        dv=512,
        block_size=64,
        dtype=torch.bfloat16,
        device=inputs["q"].device,
        causal=False,
    )
    bf16_before = plan.run(
        inputs["q"],
        inputs["blocked_k"].unsqueeze(2),
        inputs["block_table"],
        inputs["cache_seqlens"],
    )
    torch.testing.assert_close(bf16_before.float(), expected, atol=1e-3, rtol=1e-2)
    handle, (output, lse) = prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
        inputs["q_nope"],
        inputs["q_rope"],
        inputs["k_lora"],
        inputs["k_rope"],
        inputs["q_scale"],
        inputs["k_scale"],
        inputs["block_table"],
        inputs["cache_seqlens"],
        512,
        initial_cache_seqlens=inputs["lengths"],
        max_cache_seqlens=inputs["lengths"],
    )
    _assert_close(output, lse, expected, expected_lse)
    saved_output, saved_lse = output.clone(), lse.clone()
    bf16_after = plan.run(
        inputs["q"],
        inputs["blocked_k"].unsqueeze(2),
        inputs["block_table"],
        update_metadata=False,
    )
    torch.testing.assert_close(bf16_after, bf16_before, atol=0, rtol=0)
    output, lse = handle()
    torch.testing.assert_close(output, saved_output, atol=0, rtol=0)
    torch.testing.assert_close(lse, saved_lse, atol=0, rtol=0)


@pytest.mark.parametrize(
    "batch,heads,use_pdl,pretranspose",
    [
        (4, 64, False, False),
        (4, 64, True, False),
        (16, 128, False, True),
        (4, 64, True, True),
    ],
)
def test_dense_fp8_compile_time_schedules(
    batch, heads, use_pdl, pretranspose, monkeypatch
):
    module = importlib.import_module(
        "flaggems_vllm.ops.flash_mla_fp8.flash_mla_with_kvcache_fwd_w8a8_fp8"
    )
    handle_type = module._FlashMLAFp8PreparedHandle
    monkeypatch.setattr(
        handle_type, "_use_programmatic_dependent_launch", lambda self: use_pdl
    )
    monkeypatch.setattr(handle_type, "_use_pretranspose_v1", lambda self: pretranspose)
    inputs = _make_inputs(batch, heads, 640)
    expected, expected_lse = _reference(inputs)
    handle, (output, lse) = prepare_flash_mla_with_kvcache_fwd_w8a8_fp8(
        inputs["q_nope"],
        inputs["q_rope"],
        inputs["k_lora"],
        inputs["k_rope"],
        inputs["q_scale"],
        inputs["k_scale"],
        inputs["block_table"],
        inputs["cache_seqlens"],
        512,
        initial_cache_seqlens=inputs["lengths"],
        max_cache_seqlens=inputs["lengths"],
    )
    _assert_close(output, lse, expected, expected_lse)
    saved_output, saved_lse = output.clone(), lse.clone()
    output, lse = handle()
    torch.testing.assert_close(output, saved_output, atol=0, rtol=0)
    torch.testing.assert_close(lse, saved_lse, atol=0, rtol=0)
