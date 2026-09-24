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

from flaggems_vllm.ops.flash_mla_fp8.common import HAS_TLE
from flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8 import (
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
        "flaggems_vllm.ops.flash_mla_with_kvcache_fwd_w8a8_fp8"
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
