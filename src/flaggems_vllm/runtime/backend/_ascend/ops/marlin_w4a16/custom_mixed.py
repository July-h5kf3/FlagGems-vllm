# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from .compiler import compat
from .prepare_packed import prepare

SOURCE_ROOT = Path(__file__).parent
ROOT = SOURCE_ROOT / "_build"
ROOT.mkdir(exist_ok=True)


@lru_cache(maxsize=128)
def register(n, k, bm, bn, tasks, grid):
    source = (SOURCE_ROOT / "mixed_custom.cpp").read_text()
    rev = hashlib.sha256(
        (source + str((n, k, bm, bn, tasks, grid))).encode()
    ).hexdigest()[:16]
    cop = "marlin_mix_cube_" + rev
    vop = "marlin_mix_vec_" + rev
    cpp = ROOT / ("marlin_mixed_" + rev + ".cpp")
    defs = dict(
        N=n,
        K=k,
        BM=bm,
        BN=bn,
        VBN=bn // 2,
        TASKS=tasks,
        GRID=grid,
        STAGES=2,
        CUBE_ENTRY="_mlir_ciface_" + cop,
        VEC_ENTRY="_mlir_ciface_" + vop,
    )
    cpp.write_text(
        "".join((f"#define MRL_{key} {val}\n" for (key, val) in defs.items())) + source
    )
    for kind, name in [("cube", cop), ("vec", vop)]:
        bc = ROOT / (name + ".bc")
        if not bc.exists():
            subprocess.check_call(
                [
                    s.replace("dav-c220-cube", "dav-c220-" + kind)
                    for s in compat.compile_cmd(cpp, bc)
                ]
                + ["-O3"]
            )
        if kind == "cube":

            def init(self, a, w, e, o, pid, out=None):
                self.arg_type["pid"] = tl.int32

        else:

            def init(self, w, s, f, e, o, pid, sub, out=None):
                self.arg_type["pid"] = tl.int32
                self.arg_type["sub"] = tl.int32

        cls = type(
            name,
            (),
            dict(
                name=name,
                core=al.CORE.CUBE if kind == "cube" else al.CORE.VECTOR,
                pipe=al.PIPE.PIPE_ALL if kind == "cube" else al.PIPE.PIPE_V,
                mode=al.MODE.SIMD,
                symbol=name,
                bitcode=str(bc),
                source=str(cpp),
                extra_attr="" if kind == "cube" else "flaggems_pass_outputs=true",
                compile=compat.makefile_compile().replace(
                    "dav-c220-cube", "dav-c220-" + kind
                ),
                __init__=init,
            ),
        )
        al.register_custom_op(cls)
    return (cop, vop)


@triton.jit
def kernel(
    A,
    Q,
    S,
    Safe,
    Experts,
    Work,
    Output,
    COP: tl.constexpr,
    VOP: tl.constexpr,
    BN: tl.constexpr,
):
    with al.scope(core_mode="cube"):
        dummy = tl.full((16,), 0, tl.int32)
        al.custom(COP, A, Work, Experts, Output, tl.program_id(0), out=dummy)
    with al.scope(core_mode="vector"):
        scratch = tl.full((7 * (BN // 2) * 128 // 4,), 0, tl.int32)
        al.custom(
            VOP,
            Q,
            S,
            Safe,
            Experts,
            Work,
            tl.program_id(0),
            al.sub_vec_id(),
            out=scratch,
        )


def gemm(a, w, s, experts, out, bm, bn=128):
    n = out.shape[1]
    bn = min(n, 256, 32768 // bm)
    k = a.shape[1]
    tasks = a.shape[0] // bm * (n // bn)
    cores = triton.runtime.driver.active.utils.get_device_properties(a.device.index)[
        "num_aicore"
    ]
    grid = min(tasks, cores)
    (q, scale, safe) = prepare(w, s)
    (cop, vop) = register(n, k, bm, bn, tasks, grid)
    work = torch.empty((grid * 2 * bn * 128,), device=a.device, dtype=a.dtype)
    kernel[grid,](
        a,
        q,
        scale,
        safe,
        experts,
        work,
        out,
        cop,
        vop,
        bn,
        disable_auto_inject_block_sync=True,
        num_warps=1,
    )
