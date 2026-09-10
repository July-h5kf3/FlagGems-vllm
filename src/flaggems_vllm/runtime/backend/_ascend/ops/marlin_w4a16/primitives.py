# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Opt-in CANN 9.0 ABI bridge for the three FlagTree Marlin primitives.

Keep the public raw API and FlagTree validation/C++ implementation. The existing
Marlin compiler shim uses raw i64 addresses; adapters construct the descriptors
expected by FlagTree and explicitly request output arguments. No global raw
replacement, package edits or toolkit changes are performed.
"""

import functools
import hashlib
import subprocess
from pathlib import Path

from flaggems_vllm.runtime.backend._ascend.ops.marlin_w4a16.compiler import compat
from triton.experimental.tle.language.dsa.ascend.custom_ops import registry

CUSTOM = Path(registry.__file__).resolve().parent
BUILD = Path(__file__).resolve().parent / "_build" / "flagtree_abi"


def template_include():
    import os

    override = os.environ.get("FLAGTREE_TEMPLATE_INCLUDE")
    if override and (Path(override) / "Utils.h").is_file():
        return Path(override)
    for root in CUSTOM.parents:
        candidate = (
            root / "third_party/ascend/AscendNPU-IR/bishengir/lib/Template/include"
        )
        if (candidate / "Utils.h").is_file():
            return candidate
    raise RuntimeError(
        "CANN 9.0 custom ABI requires FlagTree Template headers; "
        "set FLAGTREE_TEMPLATE_INCLUDE to the directory containing Utils.h"
    )


SOURCES = {
    "compare_scalar": ("mask_ops/compare_scalar.cpp", "custom_compare_scalar_float"),
    "gather_mask": ("mask_ops/gather_mask.cpp", "custom_gather_mask_float"),
    "cast_int4_to_fp16": ("cast_ops/cast_int4_to_fp16.cpp", "custom_cast_int4_to_fp16"),
}


@functools.lru_cache(None)
def build(kind, n):
    INCLUDE = template_include()
    relative, callee = SOURCES[kind]
    source = CUSTOM / relative
    if kind == "compare_scalar":
        signature = "int64_t src, float scalar, int64_t dst"
        body = (
            f"auto s = view<float>(src, {n}); auto d = view<uint16_t>(dst, {n // 16});\n    "
            f"_mlir_ciface_{callee}(&s, scalar, &d);"
        )
    elif kind == "gather_mask":
        signature = "int64_t src, int64_t mask, int64_t dst, int64_t count"
        body = (
            f"auto s = view<float>(src, {n}); auto m = view<uint16_t>(mask, {n // 16});\n    "
            f"auto d = view<float>(dst, {n}); auto c = view<int32_t>(count, 8);\n    "
            f"_mlir_ciface_{callee}(&s, &m, &d, &c);"
        )
    else:
        signature = "int64_t src, int64_t dst"
        body = (
            f"auto s = view<uint8_t>(src, {n}); auto d = view<half>(dst, {2 * n});\n    "
            f"_mlir_ciface_{callee}(&s, &d);"
        )
    code = f"""#include "{source}"
template <typename T>
[aicore] __attribute__((always_inline)) memref_t<__ubuf__ T, 1> view(int64_t address, int64_t size) {{
    auto pointer = reinterpret_cast<__ubuf__ T *>(address);
    return {{pointer, pointer, 0, {{size}}, {{1}}}};
}}
extern "C" [aicore] __attribute__((always_inline)) void
_mlir_ciface_ABI_ENTRY({signature}) {{
    {body}
}}
"""
    digest = hashlib.sha256(
        (
            code + source.read_text() + (CUSTOM / "mask_ops/mask_common.h").read_text()
        ).encode()
    ).hexdigest()[:16]
    symbol = f"marlin_ft_{kind}_{n}_{digest}"
    BUILD.mkdir(parents=True, exist_ok=True)
    cpp, bc = BUILD / (symbol + ".cpp"), BUILD / (symbol + ".bc")
    cpp.write_text(code.replace("ABI_ENTRY", symbol))
    if not bc.exists():
        command = [
            x.replace("dav-c220-cube", "dav-c220-vec")
            for x in compat.compile_cmd(cpp, bc)
        ]
        subprocess.run(command + ["-O3", "-I" + str(INCLUDE)], check=True)
    return symbol, cpp, bc


def configure(instance, kind, src):
    symbol, cpp, bc = build(kind, src.numel.value)
    instance.symbol = symbol
    instance.source, instance.bitcode = str(cpp), str(bc)
    instance.extra_attr = "flaggems_pass_outputs=true"
    instance.compile = (
        compat.makefile_compile().replace("dav-c220-cube", "dav-c220-vec")
        + " -O3 -I"
        + str(template_include())
    )


def install():
    if getattr(registry, "_marlin_cann90_bridge", False):
        return
    compare_init = registry.compare_scalar.__init__
    gather_init = registry.gather_mask.__init__
    cast_init = registry.cast_int4_to_fp16.__init__

    def compare(self, src, scalar, out=None):
        compare_init(self, src, scalar, out=out)
        configure(self, "compare_scalar", src)

    def gather(self, src, mask, out=None):
        gather_init(self, src, mask, out=out)
        configure(self, "gather_mask", src)

    def cast(self, src, out=None):
        cast_init(self, src, out=out)
        configure(self, "cast_int4_to_fp16", src)

    registry.compare_scalar.__init__ = compare
    registry.gather_mask.__init__ = gather
    registry.cast_int4_to_fp16.__init__ = cast
    registry._marlin_cann90_bridge = True


@functools.lru_cache(None)
def register():
    """Register public primitives; adapt their ABI for the existing CANN 9.0 path."""
    required = (
        "compare_scalar",
        "gather_mask",
        "cast_int4_to_fp16",
        "cube_begin",
        "cube_end",
    )
    missing = [name for name in required if not hasattr(registry, name)]
    if missing:
        raise RuntimeError(
            "Install the FlagTree Marlin custom-op branch; missing: "
            + ", ".join(missing)
        )
    install()
    # No output descriptors are needed for the two scalar-token barriers.
    source = CUSTOM / "sync_ops/cube_boundary.cpp"
    for name in ("cube_begin", "cube_end"):
        cls = getattr(registry, name)
        original = cls.__init__

        def init(self, token, _original=original, _name=name):
            _original(self, token)
            BUILD.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
            symbol = "marlin_ft_" + _name
            cpp = BUILD / (symbol + "_" + digest + ".cpp")
            bc = cpp.with_suffix(".bc")
            cpp.write_text(
                source.read_text()
                .replace("custom_cube_begin", "marlin_ft_cube_begin")
                .replace("custom_cube_end", "marlin_ft_cube_end")
            )
            if not bc.exists():
                subprocess.run(compat.compile_cmd(cpp, bc) + ["-O3"], check=True)
            self.symbol, self.source, self.bitcode = symbol, str(cpp), str(bc)
            self.compile = compat.makefile_compile() + " -O3"

        cls.__init__ = init
