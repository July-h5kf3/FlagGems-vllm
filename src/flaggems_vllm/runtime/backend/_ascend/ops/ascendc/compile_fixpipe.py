# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build the small AscendC Fixpipe fragment used by the einsum epilogue."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_SRC = _DIR / "fixpipe_w8a8_mm.cpp"
_ARTIFACT = _DIR / "fixpipe_w8a8_mm.bc"
_SYMBOL = "fixpipe_vdeqf16"


def _find_ccec() -> str:
    candidate = os.environ.get("CCEC") or os.environ.get("BISHENG")
    if candidate and Path(candidate).is_file():
        return candidate
    found = shutil.which("ccec") or shutil.which("bisheng")
    if found:
        return found
    toolkit = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME"
    )
    if toolkit:
        candidate = Path(toolkit) / "compiler" / "ccec_compiler" / "bin" / "ccec"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("ccec/bisheng not found; source the CANN environment")


def _tikcpp_include() -> Path:
    toolkit = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME"
    )
    candidates = []
    if toolkit:
        candidates.append(Path(toolkit) / "aarch64-linux" / "tikcpp" / "tikcfw")
    candidates.append(
        Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/tikcpp/tikcfw")
    )
    for candidate in candidates:
        if (candidate / "kernel_operator.h").is_file():
            return candidate
    raise FileNotFoundError("tikcpp/tikcfw/kernel_operator.h not found")


def _cxx_includes() -> list[str]:
    for version in ("12", "11", "13"):
        include = Path(f"/usr/include/c++/{version}")
        if (include / "cstdint").is_file():
            return [f"-I{include}", f"-I/usr/include/aarch64-linux-gnu/c++/{version}"]
    return []


def compile_cmd(src: Path | None = None, out: Path | None = None) -> list[str]:
    src = src or _SRC
    out = out or _ARTIFACT
    tikcpp = _tikcpp_include()
    return [
        _find_ccec(),
        "-x",
        "cce",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-aicore-only",
        "-std=c++17",
        f"-I{tikcpp}",
        f"-I{tikcpp / 'interface'}",
        f"-I{tikcpp / 'impl'}",
        *_cxx_includes(),
        "-DFLAGGEMS_COMMON_IR_IFACE",
        "-emit-llvm",
        "-c",
        str(src),
        "-o",
        str(out),
    ]


def makefile_compile() -> str:
    tikcpp = _tikcpp_include()
    include_flags = " ".join(_cxx_includes())
    return (
        f"{_find_ccec()} -x cce --cce-aicore-arch=dav-c220-cube --cce-aicore-only "
        f"-std=c++17 -I{tikcpp} -I{tikcpp / 'interface'} -I{tikcpp / 'impl'} "
        f"{include_flags} -DFLAGGEMS_COMMON_IR_IFACE -emit-llvm -c $< -o $@"
    )


def _is_llvm_bitcode(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"BC\xc0\xde"
    except OSError:
        return False


def ensure_bitcode() -> Path:
    if (
        _ARTIFACT.exists()
        and _ARTIFACT.stat().st_mtime >= _SRC.stat().st_mtime
        and _is_llvm_bitcode(_ARTIFACT)
    ):
        return _ARTIFACT
    subprocess.check_call(compile_cmd())
    if not _is_llvm_bitcode(_ARTIFACT):
        raise RuntimeError(f"ccec did not write LLVM bitcode to {_ARTIFACT}")
    return _ARTIFACT


def source_path() -> Path:
    return _SRC


def symbol() -> str:
    return _SYMBOL
