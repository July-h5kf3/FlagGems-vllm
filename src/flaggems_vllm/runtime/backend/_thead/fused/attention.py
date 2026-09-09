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

import torch

from flaggems_vllm.runtime.backend._thead.fused.flash_attn_varlen_func_w8a8_int8 import (
    flash_attn_varlen_func_w8a8_int8,
)


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """Dispatch PPU INT8 attention; retain the shared FP16/BF16 implementation."""
    if q.dtype == torch.int8:
        impl = flash_attn_varlen_func_w8a8_int8
    else:
        # The backend is loaded while shared ops are being imported.
        from flaggems_vllm.ops.attention import flash_attn_varlen_func as impl

    return impl(
        q,
        k,
        v,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        cu_seqlens_k,
        seqused_k,
        q_v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        block_table,
        return_softmax_lse,
        out,
        scheduler_metadata,
        q_descale,
        k_descale,
        v_descale,
        s_aux,
        num_splits,
        cp_world_size,
        cp_rank,
        cp_tot_seqused_k,
        fa_version,
    )
