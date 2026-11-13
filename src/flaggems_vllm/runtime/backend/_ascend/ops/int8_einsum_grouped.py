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

import hashlib
import platform
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al

from flaggems_vllm.runtime.backend._ascend.device import get_device_core_counts


@lru_cache(maxsize=1)
def copy_source_hash():
    # Triton's Python dependency hash does not include the native fragment.
    return hashlib.sha256(
        Path(__file__).with_name("int8_einsum_copy.cpp").read_bytes()
    ).hexdigest()


@lru_cache(maxsize=1)
def _copy_bitcode():
    """Build the two DMA fragments with the active CANN toolchain once per process."""
    compiler = shutil.which("ccec")
    if compiler is None:
        raise RuntimeError("Ascend INT8 grouped pipeline requires CANN ccec")
    compiler = Path(compiler)
    linker = compiler.resolve().with_name("llvm-link")
    directory = Path(tempfile.mkdtemp(prefix="flaggems-int8-copy-"))
    source = Path(__file__).with_name("int8_einsum_copy.cpp")
    common = [
        str(compiler),
        "-O2",
        "-x",
        "cce",
        "--cce-auto-sync=off",
        "--cce-aicore-only",
        "--cce-generic-addrspace=off",
        "-mllvm",
        "-disable-llvm-optzns",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-enable-print",
        "--cce-enable-sanitizer",
        "-std=c++17",
    ]
    # CANN's compiler may not discover the host GCC standard library headers.
    for include in sorted(Path("/usr/include/c++").glob("*"), reverse=True):
        if (include / "cstdint").is_file():
            common.extend(["-I", str(include)])
            target = (
                Path("/usr/include")
                / f"{platform.machine()}-linux-gnu/c++"
                / include.name
            )
            if target.is_dir():
                common.extend(["-I", str(target)])
            break
    objects = []
    for mixed in (False, True):
        output = directory / ("mixed.bc" if mixed else "cube.bc")
        options = ["-cce-enable-mix", "-mllvm", "-enable-mix=true"] if mixed else []
        subprocess.run(
            common + options + [str(source), "-emit-llvm", "-c", "-o", str(output)],
            check=True,
        )
        objects.append(str(output))
    bitcode = directory / "copy.bc"
    subprocess.run([str(linker), *objects, "-o", str(bitcode)], check=True)
    return str(bitcode)


@al.register_custom_op
class fg_blockdiag_gmajor:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE2
    mode = al.MODE.SIMD

    def __init__(
        self,
        src,
        actual_rows: tl.uint16,
        stride: tl.uint16,
        block_rows: tl.uint16,
        groups: tl.uint16,
        out=None,
    ):
        assert out is not None and out.dtype == tl.int8 and len(out.shape) == 2
        self.symbol = "custom_fg_blockdiag_gmajor"
        self.bitcode = _copy_bitcode()


@al.register_custom_op
class fg_fullk_nd2nz:
    core = al.CORE.CUBE
    pipe = al.PIPE.PIPE_MTE2
    mode = al.MODE.SIMD

    def __init__(
        self, src, rows: tl.uint16, columns: tl.uint16, stride: tl.uint16, out=None
    ):
        assert out is not None and out.dtype == tl.int8 and len(out.shape) == 2
        self.symbol = "custom_fg_fullk_nd2nz"
        self.bitcode = _copy_bitcode()


@triton.jit
def _diagonal_x_load(
    P, actual_rows, BM: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr
):
    src = tl.make_block_ptr(
        P, (BM, GROUPS * 128), (STRIDE, 1), (0, 0), (BM, GROUPS * 128), (1, 0)
    )
    return tle.dsa.ascend.raw(
        "fg_blockdiag_gmajor",
        src,
        actual_rows,
        STRIDE,
        BM,
        GROUPS,
        out=tl.full((BM * GROUPS, GROUPS * 128), 0, tl.int8),
    )


@triton.jit
def _diagonal_y_load(P, BN: tl.constexpr, K: tl.constexpr, STRIDE: tl.constexpr):
    src = tl.make_block_ptr(P, (BN, K), (STRIDE, 1), (0, 0), (BN, K), (1, 0))
    return tle.dsa.ascend.raw(
        "fg_fullk_nd2nz", src, BN, K, STRIDE, out=tl.full((BN, K), 0, tl.int8)
    ).T


@triton.jit
def _explicit_group_pipeline(
    X,
    XS,
    Y,
    YS,
    P,
    Output,
    B,
    SX: tl.constexpr,
    SXS: tl.constexpr,
    SY: tl.constexpr,
    SYS: tl.constexpr,
    SN: tl.constexpr,
    BM: tl.constexpr,
    GROUPS: tl.constexpr,
    CORES: tl.constexpr,
    COPY_SOURCE_HASH: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles = tl.cdiv(B, BM) * 8
    with al.scope(core_mode="vector"):
        for bank in tl.static_range(2):
            al.sync_block_set(
                "vector", "cube", 2 + bank, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX
            )
        vm = tl.arange(0, BM)
        vn = tl.arange(0, 128)
        vk = tl.arange(0, 32)
        for tile in range(pid, tiles, CORES):
            row = tile // 8 * BM
            col = tile % 8 * 128
            m = row.to(tl.int64) + vm.to(tl.int64)
            sx = tl.load(
                XS + m[:, None] * SXS[0] + vk[None, :].to(tl.int64) * SXS[2],
                m[:, None] < B,
                other=0,
            ).to(tl.float32)
            sy = tl.load(
                YS + (col // SN).to(tl.int64) * SYS[1] + vk.to(tl.int64) * SYS[2]
            ).to(tl.float32)
            scales = (sx * sy[None, :]).T
            acc = tl.zeros((BM, 128), tl.float32)
            for pair in tl.range(0, 32 // (2 * GROUPS), loop_unroll_factor=2):
                for bank in tl.static_range(2):
                    al.sync_block_wait(
                        "cube", "vector", bank, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2
                    )
                    block = pair * 2 + bank
                    for phase in tl.static_range(GROUPS):
                        ptr = (
                            P + (pid * 2 + bank) * GROUPS * BM * 128 + phase * BM * 128
                        )
                        partial = tl.load(ptr + vm[:, None] * 128 + vn[None, :]).to(
                            tl.float32
                        )
                        scale = al.extract_slice(
                            scales, [block * GROUPS + phase, 0], [1, BM], [1, 1]
                        ).reshape((BM,))
                        acc = tl.fma(partial, scale[:, None], acc)
                    al.sync_block_set(
                        "vector", "cube", 2 + bank, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX
                    )
            tl.store(
                Output + m[:, None] * 1024 + col + vn[None, :], acc, m[:, None] < B
            )
    with al.scope(core_mode="cube"):
        cm = tl.arange(0, BM * GROUPS)
        cn = tl.arange(0, 128)
        for tile in range(pid, tiles, CORES):
            row = tile // 8 * BM
            col = tile % 8 * 128
            actual = tl.minimum(B - row, BM)
            for pair in range(32 // (2 * GROUPS)):
                for bank in tl.static_range(2):
                    block = pair * 2 + bank
                    r = block.to(tl.int64) * (128 * GROUPS)
                    a = _diagonal_x_load(
                        X + row.to(tl.int64) * SX[0] + r, actual, BM, GROUPS, SX[0]
                    )
                    b = _diagonal_y_load(
                        Y + col.to(tl.int64) * SY[1] + r, 128, GROUPS * 128, SY[1]
                    )
                    partial = tl.dot(a, b, out_dtype=tl.int32)
                    al.sync_block_wait(
                        "vector", "cube", 2 + bank, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX
                    )
                    ptr = P + (pid * 2 + bank) * GROUPS * BM * 128
                    tl.store(ptr + cm[:, None] * 128 + cn[None, :], partial)
                    al.sync_block_set(
                        "cube", "vector", bank, al.PIPE.PIPE_FIX, al.PIPE.PIPE_MTE2
                    )
        for bank in tl.static_range(2):
            al.sync_block_wait(
                "vector", "cube", 2 + bank, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_FIX
            )


def grouped_int8_einsum(x, xs, y, ys, output):
    """K128 partials share a paired Cube tile; Vector retains FP32 scale order."""
    b = x.shape[0]
    # Smaller row tiles keep short batches from computing mostly padding.
    bm = 16 if b <= 32 else (64 if b <= 128 else 128)
    cores = min(get_device_core_counts(x.device)[0], triton.cdiv(b, bm) * 8)
    partial = torch.empty((cores, 2, 2, bm, 128), dtype=torch.int32, device=x.device)
    _explicit_group_pipeline[(cores,)](
        x,
        xs,
        y,
        ys,
        partial,
        output,
        b,
        x.stride(),
        xs.stride(),
        y.stride(),
        ys.stride(),
        128,
        bm,
        2,
        cores,
        copy_source_hash(),
        num_warps=4,
        num_stages=1,
        multibuffer=False,
        enable_ubuf_saving=True,
        disable_auto_inject_block_sync=True,
        enable_legacy_insert_load_store_for_mix_cv=True,
        disable_auto_cv_work_space_manage=False,
    )
