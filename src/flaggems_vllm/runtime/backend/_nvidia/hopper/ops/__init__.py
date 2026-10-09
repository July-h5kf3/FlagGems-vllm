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

import triton

if triton.__version__ >= "3.4":
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.w8a8_block_fp8_matmul import (  # noqa: F401
        w8a8_block_fp8_matmul,
    )

__all__ = ["w8a8_block_fp8_matmul"]

from flaggems_vllm.utils import has_triton_tle_attrs

if has_triton_tle_attrs(("gpu.warp_specialize", "gpu.wgmma", "gpu.copy"), 3, 6, 0):
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.flash_mla_with_kvcache_fwd_w8a8_fp8 import (
        flash_mla_with_kvcache_fwd_w8a8_fp8,
        get_mla_fp8_metadata,
        prepare_flash_mla_with_kvcache_fwd_w8a8_fp8,
    )

    __all__ += [
        "flash_mla_with_kvcache_fwd_w8a8_fp8",
        "prepare_flash_mla_with_kvcache_fwd_w8a8_fp8",
        "get_mla_fp8_metadata",
    ]
