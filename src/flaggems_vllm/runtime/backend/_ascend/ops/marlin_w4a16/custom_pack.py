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


@lru_cache(maxsize=128)
def register(k, r, bm, topk, tiles, grid):
    source = (SOURCE_ROOT / "pack_custom.cpp").read_text()
    rev = hashlib.sha256(
        (source + str((k, r, bm, topk, tiles, grid))).encode()
    ).hexdigest()[:16]
    name = "marlin_pack_" + rev
    cpp = ROOT / (name + ".cpp")
    bc = ROOT / (name + ".bc")
    defs = dict(
        K=k, R=r, BM=bm, TOPK=topk, TILES=tiles, GRID=grid, ENTRY="_mlir_ciface_" + name
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

    def init(self, x, r, c, off, e, o, tile, out=None):
        self.arg_type["tile"] = tl.int32

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
def kernel(X, R, C, Offsets, Experts, A, OP: tl.constexpr, K: tl.constexpr):
    scratch = tl.full((2 * K,), 0, tl.int32)
    al.custom(OP, X, R, C, Offsets, Experts, A, tl.program_id(0), out=scratch)


def pack(x, routes, counts, offsets, experts, out, bm, topk):
    k = x.shape[1]
    cores = triton.runtime.driver.active.utils.get_device_properties(x.device.index)[
        "num_vectorcore"
    ]
    grid = min(experts.numel(), cores)
    name = register(k, routes.shape[1], bm, topk, experts.numel(), grid)
    kernel[(grid,)](
        x,
        routes,
        counts,
        offsets,
        experts,
        out,
        name,
        k,
        disable_auto_inject_block_sync=True,
        num_warps=1,
    )
