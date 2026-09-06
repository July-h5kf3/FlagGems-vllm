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

SOURCE_ROOT = Path(__file__).parent
ROOT = SOURCE_ROOT / "_build"
ROOT.mkdir(exist_ok=True)


@lru_cache(maxsize=64)
def register(r, e, grid):
    source = (SOURCE_ROOT / "routes_custom.cpp").read_text()
    rev = hashlib.sha256((source + str((r, e, grid))).encode()).hexdigest()[:16]
    name = "marlin_routes_" + rev
    cpp = ROOT / (name + ".cpp")
    bc = ROOT / (name + ".bc")
    cpp.write_text(
        f"#define MRL_R {r}\n#define MRL_E {e}\n#define MRL_GRID {grid}\n#define MRL_ENTRY _mlir_ciface_{name}\n"
        + source
    )
    if not bc.exists():
        subprocess.check_call(
            [
                s.replace("dav-c220-cube", "dav-c220-vec")
                for s in compat.compile_cmd(cpp, bc)
            ]
            + ["-O3"]
        )

    def init(self, ids, pattern, routes, counts, expert, out=None):
        self.arg_type["expert"] = tl.int32

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
def pattern_kernel(P):
    i = tl.arange(0, 4096)
    tl.store(P + i, i.to(tl.float32))


@lru_cache(maxsize=16)
def pattern(device):
    out = torch.empty((4096,), device=device, dtype=torch.float32)
    pattern_kernel[(1,)](out)
    return out


@triton.jit
def kernel(IDs, P, Routes, Counts, OP: tl.constexpr):
    scratch = tl.full((16576,), 0, tl.int32)
    al.custom(OP, IDs, P, Routes, Counts, tl.program_id(0), out=scratch)


def routes(ids, routes, counts):
    cores = triton.runtime.driver.active.utils.get_device_properties(ids.device.index)[
        "num_vectorcore"
    ]
    grid = min(counts.numel(), cores)
    name = register(ids.numel(), counts.numel(), grid)
    kernel[(grid,)](
        ids,
        pattern(ids.device),
        routes,
        counts,
        name,
        disable_auto_inject_block_sync=True,
        num_warps=1,
    )
