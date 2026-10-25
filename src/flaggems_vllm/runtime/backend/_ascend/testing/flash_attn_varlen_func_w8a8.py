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

from functools import lru_cache

import torch

import flaggems_vllm

INT8_API_NAMES = ("flash_attn_varlen_func_w8a8",)
SUPPORTS_BF16_REFERENCE = False
BASELINE_DESCRIPTION = "Baseline: vLLM-Ascend BF16 FIA; input quantization excluded."


def validate_exports() -> None:
    assert INT8_API_NAMES[0] in flaggems_vllm.ops.__all__
    assert "flash_attn_varlen_func_w8a8_int8" not in flaggems_vllm.ops.__all__


@lru_cache(maxsize=1)
def causal_mask(device):
    return torch.triu(torch.ones(2048, 2048, device=device), diagonal=1).to(torch.int8)


def fia(query, key, value, table, q_cum, k_cum, block_size, scale):
    import torch_npu

    (output, _) = torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,
        value=value,
        atten_mask=causal_mask(query.device),
        block_table=table,
        input_layout="TND",
        block_size=block_size,
        actual_seq_lengths=q_cum,
        actual_seq_lengths_kv=k_cum,
        num_key_value_heads=key.shape[1],
        num_heads=query.shape[1],
        scale=scale,
        sparse_mode=3,
    )
    return output


def benchmark_baseline(bf16_args, _int8_args):
    (query, key, value) = bf16_args[:3]
    meta = bf16_args[-1]
    return fia(
        query,
        key,
        value,
        bf16_args[17],
        meta["q_cum"],
        meta["k_cum"],
        meta["block_size"],
        meta["fia_scale"],
    )


def prepare_benchmark_reference(bf16_args, dequantized):
    (q, k, v) = bf16_args[:3]
    page = k.shape[1]
    metadata = dict(
        bf16_args[-1],
        q_cum=bf16_args[4][1:].cpu().tolist(),
        k_cum=bf16_args[7].cpu().tolist(),
        block_size=page,
        fia_scale=bf16_args[10],
    )
    baseline = fia(
        dequantized[0],
        dequantized[1].permute(0, 2, 1, 3).contiguous(),
        dequantized[2].permute(0, 2, 1, 3).contiguous(),
        bf16_args[17],
        metadata["q_cum"],
        metadata["k_cum"],
        page,
        metadata["fia_scale"],
    )
    bf16_args = (
        q,
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        *bf16_args[3:-1],
        metadata,
    )
    return (bf16_args, baseline)
