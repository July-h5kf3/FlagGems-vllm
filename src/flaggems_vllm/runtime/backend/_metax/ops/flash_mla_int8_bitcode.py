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

"""C550 MLA: MACA device cores linked through Triton extern_libs."""

import functools
import hashlib
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.core import extern
from triton.runtime.cache import get_cache_manager

from flaggems_vllm.runtime import torch_device_fn
from flaggems_vllm.runtime.backend._metax.ops import flash_mla_int8_metadata as common
from flaggems_vllm.runtime.backend._metax.ops.flash_mla_with_kvcache_fwd_w8a8_int8 import (
    FlashMLAInt8PreparedHandle,
)


@functools.lru_cache(maxsize=1)
def _compiler():
    path = Path(os.environ.get("MACA_PATH", "/opt/maca")) / "mxgpu_llvm/bin/mxcc"
    return path.resolve(), subprocess.check_output(
        [str(path), "--version"], stderr=subprocess.STDOUT
    )


@functools.lru_cache(maxsize=128)
def _library(stage, tile, spec):
    """Build one CTA-wide device function before CUDA Graph capture."""
    compiler, version = _compiler()
    source = Path(__file__).with_name("flash_mla_int8.cu")
    flags = [
        "-x",
        "maca",
        "--offload-arch=xcore1000",
        "-maca-device-only",
        "-fgpu-rdc",
        "-emit-llvm",
        "-c",
        "-O3",
        f"-DFMLA_STAGE_{stage.upper()}",
    ]
    if stage == "small":
        flags.append(f"-DHEADS={tile}")
    elif stage == "values":
        flags.append(f"-DVDIM={tile}")
    flags.extend(f"-DFMLA_{key}={value}" for key, value in spec)
    key = hashlib.sha256(
        source.read_bytes() + version + str(compiler).encode() + repr(flags).encode()
    ).hexdigest()
    cache = get_cache_manager(key)
    filename = f"mla_int8_{stage}.bc"
    cached = cache.get_file(filename)
    if cached is not None:
        return cached
    # A unique temporary directory also permits concurrent handle preparation.
    with tempfile.TemporaryDirectory(prefix="flaggems-mla-int8-") as temporary:
        output = Path(temporary) / filename
        subprocess.run(
            [str(compiler), *flags, str(source), "-o", str(output)], check=True
        )
        return cache.put(output.read_bytes(), filename)


def _scale_spec(handle):
    value = struct.unpack("f", struct.pack("f", handle.scale * 1.4426950408889634))[0]
    return (
        ("HQ", handle.heads),
        ("SQ", handle.queries),
        ("CAP", handle.tensors[6].shape[1] * 64),
        ("CAUSAL", int(handle.causal)),
        ("BF16", int(handle.tensors[1].dtype == torch.bfloat16)),
        ("SCALE", value.hex() + "f"),
    )


SMALL_SIGNATURE = (tl.int64,) * 15 + (tl.int32,) * 6 + (tl.float32,) + (tl.int32,) * 3


@extern
def _small_device(*args, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        list(args),
        {SMALL_SIGNATURE: ("mla_int8_small", tl.float32)},
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def small_native(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Partial,
    Stats,
    Output,
    LSE,
    Q_STRIDES: tl.constexpr,
    QR_STRIDES: tl.constexpr,
    KV_STRIDES: tl.constexpr,
    KR_STRIDES: tl.constexpr,
    QS_STRIDES: tl.constexpr,
    KS_STRIDES: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    LENGTH_STRIDE: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    SQ: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr,
    BatchOrder=None,
):
    _small_device(
        tl.cast(Q, tl.int64),
        tl.cast(QR, tl.int64),
        tl.cast(KV, tl.int64),
        tl.cast(KR, tl.int64),
        tl.cast(QS, tl.int64),
        tl.cast(KS, tl.int64),
        tl.cast(Table, tl.int64),
        tl.cast(Lengths, tl.int64),
        tl.cast(Partial, tl.int64),
        tl.cast(Stats, tl.int64),
        tl.cast(Output, tl.int64),
        tl.cast(LSE, tl.int64),
        tl.cast(TABLE_STRIDE, tl.int64),
        tl.cast(LENGTH_STRIDE, tl.int64),
        (
            tl.full((), 0, tl.int64)
            if BatchOrder is None
            else tl.cast(BatchOrder, tl.int64)
        ),
        tl.cast(B, tl.int32),
        tl.cast(H, tl.int32),
        tl.cast(SQ, tl.int32),
        tl.cast(SPLITS, tl.int32),
        tl.cast(CAUSAL, tl.int32),
        tl.cast(QR.dtype.element_ty == tl.bfloat16, tl.int32),
        tl.cast(SCALE * 1.4426950408889634, tl.float32),
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


SCORES_SIGNATURE = (tl.int64,) * 14 + (tl.int32,) * 6 + (tl.float32,) + (tl.int32,) * 3


@extern
def _scores_device(*args, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        list(args),
        {SCORES_SIGNATURE: ("mla_int8_scores", tl.float32)},
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def scores_native(
    Q,
    QR,
    KV,
    KR,
    QS,
    KS,
    Table,
    Lengths,
    Quantized,
    PageStats,
    SplitStats,
    Q_STRIDES: tl.constexpr,
    QR_STRIDES: tl.constexpr,
    KV_STRIDES: tl.constexpr,
    KR_STRIDES: tl.constexpr,
    QS_STRIDES: tl.constexpr,
    KS_STRIDES: tl.constexpr,
    TS: tl.constexpr,
    LS: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    CAP: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BH: tl.constexpr = 64,
    SPLITS: tl.constexpr = 4,
    BatchOrder=None,
    COL_STATS: tl.constexpr = True,
    NATIVE_SYNC: tl.constexpr = True,
):
    _scores_device(
        tl.cast(Q, tl.int64),
        tl.cast(QR, tl.int64),
        tl.cast(KV, tl.int64),
        tl.cast(KR, tl.int64),
        tl.cast(QS, tl.int64),
        tl.cast(KS, tl.int64),
        tl.cast(Table, tl.int64),
        tl.cast(Lengths, tl.int64),
        tl.cast(Quantized, tl.int64),
        tl.cast(PageStats, tl.int64),
        tl.cast(SplitStats, tl.int64),
        (
            tl.full((), 0, tl.int64)
            if BatchOrder is None
            else tl.cast(BatchOrder, tl.int64)
        ),
        tl.cast(TS, tl.int64),
        tl.cast(LS, tl.int64),
        tl.cast(H, tl.int32),
        tl.cast(SQ, tl.int32),
        tl.cast(CAP, tl.int32),
        tl.cast(SPLITS, tl.int32),
        tl.cast(CAUSAL, tl.int32),
        tl.cast(QR.dtype.element_ty == tl.bfloat16, tl.int32),
        tl.cast(SCALE * 1.4426950408889634, tl.float32),
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


VALUES_SIGNATURE = (tl.int64,) * 14 + (tl.int32,) * 11


@extern
def _values_device(*args, _semantic=None):
    return tl.core.extern_elementwise(
        "",
        "",
        list(args),
        {VALUES_SIGNATURE: ("mla_int8_pv", tl.float32)},
        is_pure=False,
        _semantic=_semantic,
    )


@triton.jit
def values_native_fragments(
    KV,
    Table,
    Lengths,
    Scores,
    CacheMetadata,
    Metadata,
    PS,
    Output,
    LSE,
    Partial,
    PartialD,
    SplitFlag,
    KV_STRIDES: tl.constexpr,
    TS: tl.constexpr,
    LS: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    SQ: tl.constexpr,
    CAP: tl.constexpr,
    BH: tl.constexpr = 64,
    SPLITS: tl.constexpr = 1,
    ADAPTIVE: tl.constexpr = False,
    QUANTIZED_CACHE: tl.constexpr = True,
    SORT_BATCHES: tl.constexpr = False,
    UNIFORM_SPLITS: tl.constexpr = 1,
    STAGED: tl.constexpr = False,
):
    tl.static_assert(BH == 64 and QUANTIZED_CACHE)
    _values_device(
        tl.cast(KV, tl.int64),
        tl.cast(Table, tl.int64),
        tl.cast(Lengths, tl.int64),
        tl.cast(Scores, tl.int64),
        tl.cast(CacheMetadata, tl.int64),
        tl.cast(Metadata, tl.int64),
        tl.cast(PS, tl.int64),
        tl.cast(Output, tl.int64),
        tl.cast(Partial, tl.int64),
        tl.cast(SplitFlag, tl.int64),
        tl.cast(KV_STRIDES[0], tl.int64),
        tl.cast(KV_STRIDES[1], tl.int64),
        tl.cast(TS, tl.int64),
        tl.cast(LS, tl.int64),
        tl.cast(B, tl.int32),
        tl.cast(H * SQ, tl.int32),
        tl.cast(CAP, tl.int32),
        tl.cast(SPLITS, tl.int32),
        tl.cast(ADAPTIVE, tl.int32),
        tl.cast(SORT_BATCHES, tl.int32),
        tl.cast(UNIFORM_SPLITS, tl.int32),
        tl.cast(Output.dtype.element_ty == tl.bfloat16, tl.int32),
        tl.program_id(0),
        tl.program_id(1),
        tl.program_id(2),
    )


class SmallKernel:
    def __init__(self, heads, handle):
        spec = _scale_spec(handle) + (("B", handle.batch), ("SPLITS", handle.splits))
        self.bitcode = _library("small", heads, spec)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.update(
                num_warps=4,
                num_stages=2,
                scenario="mla",
                extern_libs={"mla_int8_small": self.bitcode},
            )
            return small_native[grid](*args, **kwargs)

        return launch


class ScoresKernel:
    def __init__(self, handle):
        spec = _scale_spec(handle) + (("SPLITS", handle.score_splits),)
        self.bitcode = _library("scores", 64, spec)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.update(
                num_warps=4,
                num_stages=1,
                scenario="mla",
                extern_libs={"mla_int8_scores": self.bitcode},
            )
            return scores_native[grid](*args, **kwargs)

        return launch


class ValuesKernel:
    def __init__(self, dims, handle):
        self.dims = dims
        spec = (
            ("B", handle.batch),
            ("HEADS", handle.heads * handle.queries),
            ("CAP", handle.capacity),
            ("SPLITS", handle.value_splits),
            ("ADAPTIVE", int(handle.adaptive)),
            ("SORT", int(handle.sort_batches)),
            ("UNIFORM", handle.uniform_value_splits),
            ("BF16", int(handle.output.dtype == torch.bfloat16)),
        )
        self.bitcode = _library("values", dims, spec)

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            kwargs.update(
                num_warps=self.dims // 64,
                num_stages=1,
                scenario="mla",
                extern_libs={"mla_int8_pv": self.bitcode},
            )
            return values_native_fragments[(512 // self.dims, *grid[1:])](
                *args, **kwargs
            )

        return launch


class NativeSmallHandle(FlashMLAInt8PreparedHandle):
    def __init__(self, tensors, scale, causal):
        capacity = tensors[6].shape[1] * 64
        splits = min(
            (
                4
                if tensors[0].shape[1] * tensors[0].shape[2] == 32 and capacity <= 8192
                else 8
            ),
            triton.next_power_of_2(tensors[6].shape[1]),
        )
        super().__init__(tensors, splits, scale, causal)
        self.head_tile = self.queries * self.heads
        self.normalized_partials = True
        if self.head_tile == 32 and 1 < self.batch <= 256:
            self.batch_order = torch.empty(
                (self.batch,), dtype=torch.int32, device=tensors[0].device
            )
            self.batch_order_kernel = common.dense_int8_batch_order
        self.kernel_arguments = {"BatchOrder": self.batch_order}
        self.grid = (1, self.batch, self.splits)
        self.kernel = SmallKernel(self.head_tile, self)


class NativeCachedHandle(FlashMLAInt8PreparedHandle):
    def __init__(self, tensors, scale, causal):
        super().__init__(tensors, 1, scale, causal)
        merged = self.queries * self.heads
        rows = self.batch * merged
        device = tensors[0].device
        self.capacity = tensors[6].shape[1] * 64
        self.head_tile = 64
        self.pv_head_tile = 64
        self.prequantized = False
        self.grouped = False
        self.quantized_cache = True
        self.qk_native_statistics = True
        self.native_barriers = True
        self.value_staging = True
        self.adaptive = merged <= 64
        self.sort_batches = self.batch <= 256
        self.value_splits = (
            2 if self.adaptive and self.capacity <= 32768 else 4 if self.adaptive else 1
        )
        self.uniform_value_splits = (
            2
            if self.adaptive
            and (self.capacity == 8192 or 16384 <= self.capacity <= 32768)
            else 1
        )
        self.value_merge_heads = 4 if self.adaptive else 0
        self.score_splits = (
            8 if merged == 64 or (merged <= 128 and self.capacity >= 16384) else 4
        )
        if 1 < self.batch <= 256:
            self.batch_order = torch.empty(
                (self.batch,), dtype=torch.int32, device=device
            )
            self.batch_order_kernel = common.dense_int8_batch_order
        self.scores = torch.empty(
            (self.batch, self.capacity // 64, merged, 64),
            dtype=torch.uint16,
            device=device,
        )
        self.cache_metadata = torch.empty(
            (self.batch, self.capacity // 64, 2, merged, 4),
            dtype=torch.float32,
            device=device,
        )
        self.stats = torch.empty(
            (self.batch, self.score_splits, 3, 4, merged),
            dtype=torch.float32,
            device=device,
        )
        self.probability_metadata = torch.empty(
            (rows, 2), dtype=torch.float32, device=device
        )
        self.probability_scale = torch.empty(
            (rows,), dtype=torch.float32, device=device
        )
        self.split_flag = torch.empty(
            (self.batch + 1,) if self.sort_batches else (),
            dtype=torch.int32,
            device=device,
        )
        workspace_splits = self.value_splits if self.adaptive else 0
        self.partial = torch.empty(
            (workspace_splits, rows, 512), dtype=torch.float32, device=device
        )
        self.partial_denominator = torch.empty(
            (workspace_splits, rows), dtype=torch.float32, device=device
        )
        self.score_kernel = ScoresKernel(self)
        self.value_kernel = ValuesKernel(256 if merged <= 64 else 512, self)

    def __call__(self):
        q, qr, kv, kr, qs, ks, table, lengths = self.tensors
        merged = self.queries * self.heads
        rows = self.batch * merged
        with torch_device_fn.device(q.device):
            self._prepare_batch_order()
            self.score_kernel[(self.score_splits, triton.cdiv(merged, 64), self.batch)](
                *self.tensors,
                self.scores,
                self.cache_metadata,
                self.stats,
                q.stride(),
                qr.stride(),
                kv.stride(),
                kr.stride(),
                qs.stride(),
                ks.stride(),
                table.stride(0),
                lengths.stride(0),
                self.heads,
                self.queries,
                self.capacity,
                self.scale,
                self.causal,
                BH=64,
                SPLITS=self.score_splits,
                COL_STATS=True,
                NATIVE_SYNC=True,
                BatchOrder=self.batch_order,
            )
            common.dense_int8_probability_merge[(triton.cdiv(rows, 32),)](
                self.stats,
                self.probability_metadata,
                self.probability_scale,
                self.lse,
                lengths,
                self.split_flag,
                lengths.stride(0),
                self.batch,
                triton.next_power_of_2(self.batch),
                self.heads,
                self.queries,
                rows,
                self.score_splits,
                self.adaptive,
                QUANTIZED_CACHE=True,
                SORT_BATCHES=self.sort_batches,
                num_warps=4,
                num_stages=1,
            )
            self.value_kernel[
                (2, triton.cdiv(merged, 64), self.batch * self.value_splits)
            ](
                kv,
                table,
                lengths,
                self.scores,
                self.cache_metadata,
                self.probability_metadata,
                self.probability_scale,
                self.output,
                self.lse,
                self.partial,
                self.partial_denominator,
                self.split_flag,
                kv.stride(),
                table.stride(0),
                lengths.stride(0),
                self.batch,
                self.heads,
                self.queries,
                self.capacity,
                BH=64,
                SPLITS=self.value_splits,
                ADAPTIVE=self.adaptive,
                UNIFORM_SPLITS=self.uniform_value_splits,
                QUANTIZED_CACHE=True,
                SORT_BATCHES=self.sort_batches,
                STAGED=True,
            )
            if self.adaptive:
                common.dense_int8_value_merge[(self.batch, triton.cdiv(merged, 4))](
                    self.partial,
                    self.partial_denominator,
                    self.probability_metadata,
                    self.probability_scale,
                    self.output,
                    self.lse,
                    self.split_flag,
                    self.batch,
                    self.heads,
                    self.queries,
                    self.value_splits,
                    QUANTIZED_CACHE=True,
                    UNIFORM_SPLITS=self.uniform_value_splits,
                    num_warps=4,
                    num_stages=1,
                )
        return self.output, self.lse
