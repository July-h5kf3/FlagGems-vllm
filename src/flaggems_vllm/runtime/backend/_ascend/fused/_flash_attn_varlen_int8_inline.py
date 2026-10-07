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

"""Mixed Cube/Vector attention with caller-managed tiling and synchronization."""

import hashlib
from functools import lru_cache
from pathlib import Path

import torch
import triton

from flaggems_vllm.runtime import torch_device_fn

_HEAD_CAUSAL_MIN_CONTEXT = 3072
_HEAD_UNMASKED_MIN_CONTEXT = 2048
_PACKED_DECODE_MIN_GROUPS = 16
_PACKED_DECODE_MIN_KV = 1024


TILING_FIELDS = (
    "heads",
    "kvHeads",
    "tasks",
    "vectorBlocks",
    "tableStride",
    "pageStride",
    "rowStride",
    "ksBlocks",
    "vsBlocks",
    "cubeBlocks",
    "totalGroups",
    "groupSize",
    "headGroupSize",
    "groupSplits",
    "queryLen",
    "originalQueryLen",
    "queryGroups",
    "hasCuq",
    "causal",
    "qsBlocks",
    "qsStrideB",
    "qsStrideH",
    "qsStrideBlock",
    "ksStrideB",
    "ksStrideH",
    "ksStrideBlock",
    "vsStrideB",
    "vsStrideH",
    "vsStrideBlock",
    "nTiles",
    "scoreBytes",
    "probBytes",
    "pvBytes",
    "alphaBytes",
    "vNzBytes",
    "separateCorrection",
    "corrProbBytes",
    "corrPvBytes",
)


@lru_cache(maxsize=None)
def _bitcode_key(path):
    # File contents, not only the Python registration, affect generated code.
    # The bundle is immutable for the lifetime of the process.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def _supported_device(device):
    # The mixed core mapping have been validated for this physical SKU.
    with torch_device_fn.device(device):
        return "910B4" in torch_device_fn.get_device_name(device)


def _scale_supported(scale, batch, heads, length, device):
    if not isinstance(scale, torch.Tensor):
        return False
    if (
        scale.dtype != torch.float32
        or scale.ndim != 3
        or scale.device != device
        or scale.shape[:2] != (batch, heads)
    ):
        return False
    if length > 0 and (
        scale.shape[2] == 0
        or (scale.stride(2) != 0 and scale.shape[2] < triton.cdiv(length, 128))
    ):
        return False
    # The device tiling uses uint32 strides and offsets. Non-contiguous and
    # broadcast scale views are supported, but their reachable offset must fit.
    if any(stride < 0 or stride >= 2**32 for stride in scale.stride()):
        return False
    return (
        sum(
            max(size - 1, 0) * stride
            for size, stride in zip(scale.shape, scale.stride())
        )
        < 2**32
    )


def _eligible(q, k, v, cuq, table, used, qs, ks, vs, out, maxq, maxk):
    if not all(isinstance(x, torch.Tensor) for x in (q, k, v, cuq, table, used)):
        return False
    if (
        q.device.type != "npu"
        or q.ndim != 3
        or q.shape[2] != 128
        or k.ndim != 4
        or k.shape[1] != 16
        or k.shape[3] != 128
        or v.shape != k.shape
        or table.ndim != 2
        or q.shape[0] == 0
    ):
        return False
    if (
        not isinstance(maxq, int)
        or not isinstance(maxk, int)
        or not 0 < maxq < 2**32
        or not 0 <= maxk < 2**32
    ):
        return False
    batch = table.shape[0]
    if (
        batch < 1
        or cuq.shape != (batch + 1,)
        or used.shape != (batch,)
        or table.shape[1] < triton.cdiv(maxk, 16)
        or q.shape[0] > batch * maxq
    ):
        return False
    heads, kv_heads = q.shape[1], k.shape[2]
    if heads < 1 or kv_heads < 1 or heads % kv_heads:
        return False
    single_head = q.shape[0] == batch * maxq and maxq >= 128
    # Grouped kernels physically pack four Q heads per KV head. The long
    # uniform single-head kernels instead map queryHead // group on device.
    if heads // kv_heads != 4 and not single_head:
        return False
    # C220 Nd2NzParams.srcDValue is uint16. Grouped Q copies individual rows
    # with source stride D; the single-head path copies across token rows.
    if k.stride(1) >= 2**16 or (single_head and q.stride(0) >= 2**16):
        return False
    for tensor in (q, k, v):
        if (
            tensor.dtype != torch.int8
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            return False
    for tensor in (cuq, table, used):
        if (
            tensor.dtype != torch.int32
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            return False
    if max(x.numel() for x in (q, k, v, cuq, table, used)) >= 2**32:
        return False
    if out is not None and (
        not isinstance(out, torch.Tensor)
        or out.dtype != torch.bfloat16
        or out.shape != q.shape
        or out.device != q.device
        or not out.is_contiguous()
    ):
        return False
    for scale, heads, length in (
        (qs, heads, maxq),
        (ks, kv_heads, maxk),
        (vs, kv_heads, maxk),
    ):
        if not _scale_supported(scale, batch, heads, length, q.device):
            return False
    return True


def _metadata(
    fields,
    q,
    k,
    table,
    qs,
    ks,
    vs,
    maxq,
    causal,
    width,
    tile,
    head=False,
    small=False,
    packed=False,
):
    batch = table.shape[0]
    rows = tile if head else tile * 4
    pv_rows = 2 * rows + 1
    pv_m = triton.cdiv(pv_rows, 16) * 16
    query_groups = 2 if small else triton.cdiv(maxq, tile)
    heads, kv_heads = q.shape[1], k.shape[2]
    groups = batch * query_groups * (heads if head else kv_heads)
    blocks = min(20, groups)
    values = dict.fromkeys(fields, 0)
    values.update(
        heads=heads,
        kvHeads=kv_heads,
        tasks=batch * query_groups * tile * heads,
        hasCuq=1,
        vectorBlocks=2 * blocks,
        tableStride=table.shape[1],
        pageStride=k.stride(0),
        rowStride=k.stride(1),
        cubeBlocks=blocks,
        totalGroups=groups,
        groupSize=4,
        headGroupSize=4,
        groupSplits=1,
        queryLen=tile,
        originalQueryLen=maxq,
        queryGroups=query_groups,
        causal=int(causal),
        qsBlocks=qs.shape[2],
        ksBlocks=ks.shape[2],
        vsBlocks=vs.shape[2],
        qsStrideB=qs.stride(0),
        qsStrideH=qs.stride(1),
        qsStrideBlock=qs.stride(2),
        ksStrideB=ks.stride(0),
        ksStrideH=ks.stride(1),
        ksStrideBlock=ks.stride(2),
        vsStrideB=vs.stride(0),
        vsStrideH=vs.stride(1),
        vsStrideBlock=vs.stride(2),
        nTiles=triton.cdiv(table.shape[1], 8),
        scoreBytes=blocks * 2 * rows * width * 4,
        probBytes=blocks * 2 * pv_m * width,
        pvBytes=blocks * 2 * pv_rows * 128 * 4,
    )
    if head:
        # Q128 uses separate high/low PV operations; smaller grouped tiles
        # concatenate both planes and their correction row in one PV matrix.
        high_m = triton.cdiv(rows + 1, 16) * 16
        values["probBytes"] = blocks * 2 * (high_m + rows) * width
        values["pvBytes"] = blocks * 2 * (2 * rows + 1) * 128 * 4
    if width >= 512:
        extra_rows = triton.cdiv(rows // 2, 16) * 16
        values["corrProbBytes"] = blocks * 2 * 2 * extra_rows * width
        values["corrPvBytes"] = blocks * 2 * rows * 128 * 4
    if packed:
        groups = batch * (kv_heads // 4)
        blocks = min(20, groups)
        values.update(
            totalGroups=groups,
            cubeBlocks=blocks,
            vectorBlocks=2 * blocks,
            headGroupSize=16,
            scoreBytes=blocks * 2 * 16 * width * 4,
            probBytes=blocks * 2 * 64 * width,
            pvBytes=blocks * 2 * 36 * 128 * 4,
            corrProbBytes=blocks * 2 * 4 * 16 * width,
            corrPvBytes=blocks * 2 * 4 * 4 * 128 * 4,
        )
    if any(not 0 <= value < 2**32 for value in values.values()):
        return None
    workspace_bytes = (
        values["scoreBytes"]
        + values["probBytes"]
        + values["pvBytes"]
        + values["corrProbBytes"]
        + values["corrPvBytes"]
    )
    workspace_bytes += blocks * 2 * 64 if width >= 512 else 0
    return tuple(values[name] for name in fields), blocks, workspace_bytes


def try_run(q, k, v, maxq, cuq, maxk, table, used, qs, ks, vs, out, causal):
    """Return None for unsupported scope, otherwise return the public output.

    CUQ/used/table *contents* follow the same validity contract as the public
    operator; this path never copies sequence metadata to the CPU to inspect it.
    """
    if not _eligible(q, k, v, cuq, table, used, qs, ks, vs, out, maxq, maxk):
        return None
    if not _supported_device(q.device):
        return None
    width = 256 if vs.stride(2) == 0 or vs.shape[2] == 1 else 128
    uniform = q.shape[0] == table.shape[0] * maxq
    minimum_wide_k = 256 if maxq <= 4 else 512
    width = (
        (1024 if maxk >= 2048 else 512)
        if uniform and maxq <= 8 and maxk > minimum_wide_k and width == 256
        else width
    )
    estimated_visible_keys = maxk - (maxq - 1) / 2 if causal else maxk
    if (
        uniform
        and maxq >= 128
        and width == 256
        and estimated_visible_keys
        >= (_HEAD_CAUSAL_MIN_CONTEXT if causal else _HEAD_UNMASKED_MIN_CONTEXT)
    ):
        width = 512
    # Q128 tails can regress small mixed batches (for example Q=[129, 5]).
    # Restrict the larger tile to long-prefill mixtures validated in benchmarks.
    mixed_head = not uniform and maxq >= 1024 and width == 256
    # Unlike grouped copies, single-head Nd2Nz uses the inter-token Q stride.
    if mixed_head and q.stride(0) >= 2**16:
        return None
    head = (uniform and maxq >= 128) or mixed_head
    partitioned_head = head and uniform and width == 512 and table.shape[0] > 1
    packed_decode = (
        uniform
        and maxq == 1
        and maxk >= _PACKED_DECODE_MIN_KV
        and q.shape[1] == 4 * k.shape[2]
        and k.shape[2] % 4 == 0
        and table.shape[0] * (k.shape[2] // 4) >= _PACKED_DECODE_MIN_GROUPS
        and (vs.stride(2) == 0 or vs.shape[2] == 1)
    )
    if packed_decode:
        width = 512
    if head:
        tile = 128
    elif width >= 512 and not packed_decode:
        tile = 4
    else:
        tile = min(16, triton.next_power_of_2(maxq))
    prepared = _metadata(
        TILING_FIELDS,
        q,
        k,
        table,
        qs,
        ks,
        vs,
        maxq,
        causal,
        width,
        tile,
        head=head,
        packed=packed_decode,
    )
    if prepared is None:
        return None
    metadata, blocks, workspace_bytes = prepared
    mixed = mixed_head or (not uniform and tile == 16 and width == 256)
    small = None
    if mixed or partitioned_head:
        small = _metadata(
            TILING_FIELDS,
            q,
            k,
            table,
            qs,
            ks,
            vs,
            maxq,
            causal,
            256 if partitioned_head else width,
            128 if partitioned_head else 2,
            head=partitioned_head,
            small=not partitioned_head,
        )
        if small is None:
            return None
        workspace_bytes = max(workspace_bytes, small[2])
    if head and width == 512:
        flag_offset = triton.cdiv(workspace_bytes, 64) * 16
        metadata = (*metadata, flag_offset)
        workspace_bytes = flag_offset * 4 + metadata[10] * 2 * 4
    from triton.experimental.tle.language.dsa.ascend.custom_ops.registry import (
        CUSTOM_OPS_BITCODE,
    )

    from ._flash_attn_varlen_int8_head_cube import (
        launch_cube_tle,
        launch_grouped_tle,
        launch_hybrid_large_tle,
        launch_hybrid_small_tle,
        launch_packed_tle,
    )

    ordinary_head = head and not (partitioned_head or mixed)
    cube_bundle_key = (_bitcode_key(CUSTOM_OPS_BITCODE),)
    small_bundle_key = cube_bundle_key
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=q.device)
    args = (q, k, v, table, used, qs, ks, vs, out, workspace.view(torch.int32), cuq)

    def launch_cube(launch_blocks, launch_width, mode, launch_metadata):
        if launch_width == 128:
            # Uniform queries are required by the N128 Vector path.
            launch_args = (*args[:9], workspace.view(torch.int32), None)
        else:
            launch_args = args
        workspace_i32 = workspace.view(torch.int32)
        if mode == 0:
            launch_kernel = launch_grouped_tle
            launch_args = (
                *args[:9],
                workspace.view(torch.int32),
                cuq,
                cube_bundle_key,
                launch_width,
                mode,
                launch_metadata,
            )
        elif mode == 6:
            launch_kernel = launch_packed_tle
            launch_args = (
                *args[:9],
                workspace.view(torch.int32),
                cuq,
                cube_bundle_key,
                launch_width,
                mode,
                launch_metadata,
            )
        elif mode == 2:
            launch_kernel = launch_hybrid_large_tle
            launch_args = (*args, cube_bundle_key, launch_width, mode, launch_metadata)
        elif mode == 1:
            launch_kernel = launch_hybrid_small_tle
            launch_args = (
                *args[:9],
                workspace.view(torch.int32),
                cuq,
                small_bundle_key,
                launch_width,
                mode,
                launch_metadata,
            )
        else:
            launch_kernel = launch_cube_tle
            launch_args = (
                *launch_args,
                workspace_i32,
                cube_bundle_key,
                launch_width,
                mode,
                launch_metadata,
            )
        launch_kernel[(launch_blocks,)](
            *launch_args,
            disable_auto_inject_block_sync=True,
            multibuffer=False,
            enable_legacy_insert_load_store_for_mix_cv=mode == 1,
            enable_auto_bind_sub_block=True,
            disable_fma=launch_width == 128 or mode == 6,
            enable_ubuf_saving=(
                (
                    mode in (0, 2, 3, 4, 5, 6)
                    and launch_width != 128
                    and not (mode == 0 and launch_width == 1024)
                )
                or (
                    launch_width == 128
                    and not (
                        launch_metadata[15] % 128 == 0
                        and launch_metadata[10] <= launch_blocks
                    )
                )
            ),
        )

        if mode in (3, 4, 5) and launch_width == 512:
            from ._flash_attn_varlen_int8_head_replay_cube import (
                launch_cube_tle as replay_kernel,
            )

            replay_metadata = list(launch_metadata[:38])
            replay_metadata[14] = 32
            replay_metadata[16] = triton.cdiv(replay_metadata[15], 32)
            replay_metadata[10] = (
                replay_metadata[10] // launch_metadata[16] * replay_metadata[16]
            )
            replay_metadata[9] = min(20, replay_metadata[10])
            replay_metadata[3] = 2 * replay_metadata[9]
            replay_metadata[2] = replay_metadata[10] * 32
            rings = 2 * replay_metadata[9]
            replay_metadata[30] = rings * 32 * 512 * 4
            replay_metadata[31] = rings * 80 * 512
            replay_metadata[32] = rings * 65 * 128 * 4
            replay_metadata[36] = rings * 32 * 512
            replay_metadata[37] = rings * 32 * 128 * 4
            replay_metadata = (
                *replay_metadata,
                launch_metadata[38],
                launch_metadata[16],
            )
            replay_kernel[(replay_metadata[9],)](
                *launch_args[:-1],
                replay_metadata,
                disable_auto_inject_block_sync=True,
                multibuffer=False,
                enable_legacy_insert_load_store_for_mix_cv=False,
                enable_auto_bind_sub_block=True,
                disable_fma=False,
                enable_ubuf_saving=True,
            )

    with torch_device_fn.device(q.device):
        if partitioned_head:
            small_metadata, small_blocks, _ = small
            launch_cube(small_blocks, 256, 5, small_metadata)
            launch_cube(blocks, 512, 5, metadata)
        elif mixed:
            small_metadata, small_blocks, _ = small
            launch_cube(small_blocks, width, 1, small_metadata)
            if mixed_head:
                launch_cube(blocks, width, 4, metadata)
            else:
                launch_cube(blocks, width, 2, metadata)
        elif ordinary_head:
            launch_cube(blocks, width, 3, metadata)
        else:
            if packed_decode:
                launch_cube(blocks, width, 6, metadata)
            else:
                launch_cube(blocks, width, 0, metadata)
    return out
