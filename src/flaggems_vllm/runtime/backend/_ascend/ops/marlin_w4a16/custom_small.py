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
from .custom_aux import combine, silu
from .custom_mixed import gemm

SOURCE_ROOT = Path(__file__).parent
ROOT = SOURCE_ROOT / "_build"
ROOT.mkdir(exist_ok=True)


@lru_cache(maxsize=64)
def register(k, t, r, bm, grid):
    source = (SOURCE_ROOT / "small_pack.cpp").read_text()
    digest = hashlib.sha256((source + str((k, t, r, bm, grid))).encode()).hexdigest()[
        :16
    ]
    name = "marlin_small_pack_" + digest
    cpp = ROOT / (name + ".cpp")
    bc = ROOT / (name + ".bc")
    defs = dict(K=k, TOPK=t, R=r, BM=bm, GRID=grid, ENTRY="_mlir_ciface_" + name)
    cpp.write_text(
        "".join(f"#define MRL_{key} {value}\n" for key, value in defs.items()) + source
    )
    if not bc.exists():
        subprocess.check_call(
            [
                s.replace("dav-c220-cube", "dav-c220-vec")
                for s in compat.compile_cmd(cpp, bc)
            ]
            + ["-O3"]
        )

    def init(self, x, i, e, v, c, o, pid, out=None):
        self.arg_type["pid"] = tl.int32

    cls = type(
        name,
        (),
        dict(
            name=name,
            core=al.CORE.VECTOR,
            pipe=al.PIPE.PIPE_V,
            mode=al.MODE.SIMD,
            symbol=name,
            bitcode=str(bc),
            source=str(cpp),
            extra_attr="flaggems_pass_outputs=true",
            compile=compat.makefile_compile().replace("dav-c220-cube", "dav-c220-vec"),
            __init__=init,
        ),
    )
    al.register_custom_op(cls)
    return name


@triton.jit
def kernel(
    X, IDs, Experts, Inv, Active, A, OP: tl.constexpr, K: tl.constexpr, BM: tl.constexpr
):
    scratch = tl.full((BM * K // 2 + 16,), 0, tl.int32)
    al.custom(OP, X, IDs, Experts, Inv, Active, A, tl.program_id(0), out=scratch)


def run(x, w1, w2, s1, s2, p, ids):
    m, k = x.shape
    t = ids.shape[1]
    r = m * t
    n = w1.shape[1] // 2
    bm = 16
    experts = torch.empty(r, device=x.device, dtype=torch.int32)
    inv = torch.empty_like(experts)
    active = torch.empty(1, device=x.device, dtype=torch.int32)
    a0 = torch.empty((r * bm, k), device=x.device, dtype=x.dtype)
    h = torch.empty((r * bm, 2 * n), device=x.device, dtype=x.dtype)
    a = torch.empty((r * bm, n), device=x.device, dtype=x.dtype)
    z = torch.empty((r * bm, k), device=x.device, dtype=x.dtype)
    out = torch.empty_like(x)
    cores = triton.runtime.driver.active.utils.get_device_properties(x.device.index)[
        "num_vectorcore"
    ]
    grid = min(r, cores)
    name = register(k, t, r, bm, grid)
    kernel[(grid,)](
        x,
        ids,
        experts,
        inv,
        active,
        a0,
        name,
        k,
        bm,
        disable_auto_inject_block_sync=True,
        num_warps=1,
    )
    gemm(a0, w1, s1, experts, h, bm, 256)
    silu(h, a, active)
    gemm(a, w2, s2, experts, z, bm, 256)
    combine(z, p, out, inv)
    return out
