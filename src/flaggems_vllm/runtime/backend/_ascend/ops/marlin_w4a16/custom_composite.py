# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from .compiler import compat
from .prepare_packed import prepare

SOURCE = Path(__file__).parent
ROOT = SOURCE / "_build"
ROOT.mkdir(exist_ok=True)


@lru_cache(maxsize=128)
def register(m, k, n, t, grid):
    r = m * t
    bm = 16
    bn1 = min(2 * n, 256)
    bn2 = min(k, 256)
    offsets = {}
    size = 0
    for key, count in [
        ("ep", r * 4),
        ("ip", r * 4),
        ("cp", 32),
        ("a0", r * bm * k * 2),
        ("h", r * bm * 2 * n * 2),
        ("a", r * bm * n * 2),
        ("z", r * bm * k * 2),
        ("work", grid * 2 * max(bn1, bn2) * 128 * 2),
    ]:
        offsets[key] = size
        size = (size + count + 511) // 512 * 512
    # Each mixed launch covers every physical AIC and its two AIVs.
    # All cores participate in every hardware barrier, including idle cores.
    # SyncAll<false> uses flags 11/12/13; GEMM rings use separate flags 2/3.
    text = '#include "kernel_operator.h"\nusing namespace AscendC;\n'

    def embed(ns, file, defs, vector=False):
        source = (SOURCE / file).read_text().replace('#include "kernel_operator.h"', "")
        keys = set(re.findall(r"MRL_[A-Z_0-9]+", source))
        return (
            ("#if defined(__DAV_C220_VEC__)\n" if vector else "")
            + "namespace "
            + ns
            + " {\n"
            + "".join(f"#define MRL_{key} {val}\n" for key, val in defs.items())
            + source
            + "\n}\n"
            + "".join(f"#undef {key}\n" for key in sorted(keys))
            + ("#endif\n" if vector else "")
        )

    text += embed(
        "Pack",
        "small_pack.cpp",
        dict(K=k, TOPK=t, R=r, BM=bm, GRID=2 * grid, ENTRY="pack_stage"),
        True,
    )
    for ns, nn, kk, bn in [("G1", 2 * n, k, bn1), ("G2", k, n, bn2)]:
        text += embed(
            ns,
            "mixed_custom.cpp",
            dict(
                N=nn,
                K=kk,
                BM=bm,
                BN=bn,
                VBN=bn // 2,
                TASKS=r * (nn // bn),
                GRID=grid,
                STAGES=2,
                CUBE_ENTRY=ns + "_cube",
                VEC_ENTRY=ns + "_vec",
            ),
        )
    text += embed(
        "Act",
        "aux_custom.cpp",
        dict(
            K=n,
            B=n,
            TOTAL=r * bm * n,
            TOPK=1,
            ACTIVE_E=0,
            RP=1,
            GRID=2 * grid,
            KIND=0,
            INV=0,
            ENTRY="act_stage",
        ),
        True,
    )
    text += embed(
        "Reduce",
        "aux_custom.cpp",
        dict(
            K=k,
            B=min(k, 1024),
            TOTAL=m * k,
            TOPK=t,
            ACTIVE_E=-1,
            RP=1,
            GRID=2 * grid,
            KIND=1,
            INV=1,
            ENTRY="reduce_stage",
        ),
        True,
    )
    addr = "".join(f"int64_t {key}=arena+{val};\n" for key, val in offsets.items())
    signature = (
        "int64_t x,int64_t q1,int64_t s1,int64_t f1,int64_t q2,int64_t s2,"
        "int64_t f2,int64_t ids,int64_t p,int64_t arena,int64_t out,int32_t pid"
    )
    text += (
        '#if defined(__DAV_C220_CUBE__)\nextern "C" [aicore] __attribute__((always_inline)) void COMPOSITE_CUBE('
        + signature
        + ") {\n"
        + addr
        + """
    SyncAll<false>();
    G1::G1_cube(a0,work,ep,h,pid);
    SyncAll<false>(); SyncAll<false>();
    G2::G2_cube(a,work,ep,z,pid);
    SyncAll<false>(); SyncAll<false>();
    }\n#endif\n"""
    )
    text += (
        '#if defined(__DAV_C220_VEC__)\nextern "C" [aicore] __attribute__((always_inline)) void COMPOSITE_VEC('
        + signature
        + ",int32_t sub,int64_t scratch) {\n"
        + addr
        + """
    int vpid=pid*2+sub;
    Pack::pack_stage(x,ids,ep,ip,cp,a0,vpid,scratch);
    SyncAll<false>();
    G1::G1_vec(q1,s1,f1,ep,work,pid,sub,scratch);
    SyncAll<false>();
    Act::act_stage(h,p,cp,a,vpid,scratch);
    SyncAll<false>();
    G2::G2_vec(q2,s2,f2,ep,work,pid,sub,scratch);
    SyncAll<false>();
    Reduce::reduce_stage(z,p,ip,out,vpid,scratch);
    SyncAll<false>();
    }\n#endif\n"""
    )
    digest = hashlib.sha256(text.encode()).hexdigest()[:16]
    cop = "marlin_comp_cube_" + digest
    vop = "marlin_comp_vec_" + digest
    text = text.replace("COMPOSITE_CUBE", "_mlir_ciface_" + cop).replace(
        "COMPOSITE_VEC", "_mlir_ciface_" + vop
    )
    cpp = ROOT / (digest + ".cpp")
    cpp.write_text(text)
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

            def init(
                self, x, q1, s1, f1, q2, s2, f2, ids, p, arena, outp, pid, out=None
            ):
                self.arg_type["pid"] = tl.int32

        else:

            def init(
                self, x, q1, s1, f1, q2, s2, f2, ids, p, arena, outp, pid, sub, out=None
            ):
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
    scratch = max(
        bm * k * 2 + 64, 7 * (max(bn1, bn2) // 2) * 128, 20 * max(n, min(k, 1024))
    )
    return cop, vop, size, (scratch + 3) // 4


@triton.jit
def kernel(
    X,
    Q1,
    S1,
    F1,
    Q2,
    S2,
    F2,
    IDs,
    P,
    Arena,
    O,
    COP: tl.constexpr,
    VOP: tl.constexpr,
    SCRATCH: tl.constexpr,
):
    with al.scope(core_mode="cube"):
        dummy = tl.full((16,), 0, tl.int32)
        al.custom(
            COP,
            X,
            Q1,
            S1,
            F1,
            Q2,
            S2,
            F2,
            IDs,
            P,
            Arena,
            O,
            tl.program_id(0),
            out=dummy,
        )
    with al.scope(core_mode="vector"):
        scratch = tl.full((SCRATCH,), 0, tl.int32)
        al.custom(
            VOP,
            X,
            Q1,
            S1,
            F1,
            Q2,
            S2,
            F2,
            IDs,
            P,
            Arena,
            O,
            tl.program_id(0),
            al.sub_vec_id(),
            out=scratch,
        )


def run(x, w1, w2, s1, s2, p, ids):
    m, k = x.shape
    t = ids.shape[1]
    n = w1.shape[1] // 2
    grid = triton.runtime.driver.active.utils.get_device_properties(x.device.index)[
        "num_aicore"
    ]
    cop, vop, size, scratch = register(m, k, n, t, grid)
    q1, s1, f1 = prepare(w1, s1)
    q2, s2, f2 = prepare(w2, s2)
    arena = torch.empty((size,), device=x.device, dtype=torch.uint8)
    out = torch.empty_like(x)
    kernel[(grid,)](
        x,
        q1,
        s1,
        f1,
        q2,
        s2,
        f2,
        ids,
        p,
        arena,
        out,
        cop,
        vop,
        scratch,
        disable_auto_inject_block_sync=True,
        num_warps=1,
    )
    return out
