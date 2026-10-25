# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Precision tests for fused_marlin_moe (FlagGems Phase 2).

Phase 2 routes through the wna16 Triton kernel (fused_moe_kernel_gptq_awq)
for true fused-dequant W4A16 GEMM. Inputs are therefore real GPTQ-quantized
weights produced by vLLM's quantize_weights, not unit-scale FP16 stand-ins.

Oracle: dequantized weights run through a naive PyTorch SwiGLU MoE reference.
The wrapper sees packed uint8 weights; the reference sees the matching
fp16/bf16 w_ref returned by quantize_weights so quantization round-off is
shared by both sides.
"""

from importlib import import_module
from pathlib import Path
from types import ModuleType

import flaggems_vllm


def backend_validation_module(operator: str) -> ModuleType | None:
    vendor = flaggems_vllm.runtime.backend.get_vendor_module()
    source = Path(vendor.__file__).parent / "testing" / f"{operator}.py"
    if not source.is_file():
        return None
    return import_module(f"{vendor.__name__}.testing.{operator}")
