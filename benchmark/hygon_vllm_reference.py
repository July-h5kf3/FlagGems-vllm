# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only 64-bit addressing fix for installed vLLM 0.6.2.

The original Triton expert offset is int32; on gfx936 its generated buffer
load also has a 2 GiB resource range. Large expert banks require promoting the
expert ID before multiplying by its stride. Do not modify installed packages.
"""

import hashlib
import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

_PATCH_DIR = None
_PATCH_METADATA = None


def ensure_large_weight_addressing(weights):
    global _PATCH_DIR, _PATCH_METADATA
    if _PATCH_METADATA is not None:
        return _PATCH_METADATA
    required = any(
        sum((size - 1) * stride for size, stride in zip(w.shape, w.stride()))
        * w.element_size()
        + w.element_size()
        >= 2**31 - 2
        for w in weights
    )
    if not required:
        return dict(address_patch=False)
    module = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    original = module.fused_moe_kernel.src
    before = "off_experts = tl.load(expert_ids_ptr + pid_m)"
    after = before + ".to(tl.int64)"
    if original.count(before) != 1 or after in original:
        raise RuntimeError(
            "Unexpected vLLM source; review the address fix before benchmarking"
        )
    patched = original.replace(before, after, 1)
    _PATCH_DIR = tempfile.TemporaryDirectory(prefix="hygon-vllm-address-")
    path = Path(_PATCH_DIR.name) / "reference.py"
    path.write_text(
        "import triton\nimport triton.language as tl\n\n@triton.jit\n" + patched
    )
    name = "_hygon_vllm_address_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    copied = importlib.util.module_from_spec(spec)
    sys.modules[name] = copied
    spec.loader.exec_module(copied)
    module.fused_moe_kernel = copied.fused_moe_kernel
    _PATCH_METADATA = dict(
        address_patch=True,
        before=before,
        after=after,
        original_kernel_sha256=hashlib.sha256(original.encode()).hexdigest(),
        patched_kernel_sha256=hashlib.sha256(patched.encode()).hexdigest(),
    )
    return _PATCH_METADATA
