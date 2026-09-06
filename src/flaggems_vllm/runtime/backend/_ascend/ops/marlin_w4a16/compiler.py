# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Use a frozen CANN 9.0 CommonIR adapter in a writable build directory."""

import importlib.util
from pathlib import Path

SOURCE_ROOT = Path(__file__).parent
BUILD = SOURCE_ROOT / "_build"
BUILD.mkdir(exist_ok=True)
(BUILD / "compile-debug").mkdir(exist_ok=True)
source = (SOURCE_ROOT / "compile_fixpipe.py").read_text()
source = source.replace(
    "/data/ldc/ops_work/flaggems-vllm/work/compiler-debug", str(BUILD / "compile-debug")
)
target = BUILD / "compile_fixpipe.py"
if not target.exists() or target.read_text() != source:
    target.write_text(source)
spec = importlib.util.spec_from_file_location("flaggems_marlin_cann90_compat", target)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)
compat.install_cann90_custom_op_compat()
