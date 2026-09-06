# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path

import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from .compiler import compat

SOURCE_ROOT = Path(__file__).parent
ROOT = SOURCE_ROOT / "_build"
ROOT.mkdir(exist_ok=True)


@lru_cache(maxsize=256)
def register(k, total, topk, kind, inv, cores):
    source = (SOURCE_ROOT / "aux_custom.cpp").read_text()
    b = min(k & -k, 4096 if kind else 256)
    grid = min(total // b, cores)
    rev = hashlib.sha256(
        (source + str((k, total, topk, kind, inv, b, grid))).encode()
    ).hexdigest()[:16]
    name = "marlin_aux_" + rev
    cpp = ROOT / (name + ".cpp")
    bc = ROOT / (name + ".bc")
    defs = dict(
        K=k,
        TOTAL=total,
        TOPK=topk,
        KIND=kind,
        INV=int(inv),
        B=b,
        GRID=grid,
        ENTRY="_mlir_ciface_" + name,
    )
    cpp.write_text(
        "".join(f"#define MRL_{key} {val}\n" for key, val in defs.items()) + source
    )
    if not bc.exists():
        subprocess.check_call(
            [
                s.replace("dav-c220-cube", "dav-c220-vec")
                for s in compat.compile_cmd(cpp, bc)
            ]
            + ["-O3"]
        )

    def init(self, a, p, inv, o, pid, out=None):
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
    return name, grid, b


@triton.jit
def kernel(A, P, Inv, Output, OP: tl.constexpr, B: tl.constexpr):
    scratch = tl.full((B * 5,), 0, tl.int32)
    al.custom(OP, A, P, Inv, Output, tl.program_id(0), out=scratch)


def silu(a, out):
    cores = triton.runtime.driver.active.utils.get_device_properties(a.device.index)[
        "num_vectorcore"
    ]
    name, grid, b = register(out.shape[1], out.numel(), 1, 0, False, cores)
    kernel[(grid,)](
        a, a, a, out, name, b, disable_auto_inject_block_sync=True, num_warps=1
    )


def combine(a, p, out, inv=None):
    cores = triton.runtime.driver.active.utils.get_device_properties(a.device.index)[
        "num_vectorcore"
    ]
    name, grid, b = register(
        out.shape[1], out.numel(), p.shape[1], 1, inv is not None, cores
    )
    kernel[(grid,)](
        a,
        p,
        p if inv is None else inv,
        out,
        name,
        b,
        disable_auto_inject_block_sync=True,
        num_warps=1,
    )
