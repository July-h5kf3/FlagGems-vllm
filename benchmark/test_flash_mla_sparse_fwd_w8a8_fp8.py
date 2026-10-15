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

import pytest

from . import test_flash_mla_sparse_fwd as sparse_fixture


@pytest.mark.flash_mla_sparse_fwd_w8a8_fp8
def test_flash_mla_sparse_fwd_w8a8_fp8():
    if not sparse_fixture.HAS_FP8_MLA:
        pytest.skip("FP8 MLA requires NVIDIA Hopper and Triton TLE")
    if not sparse_fixture.HAS_VLLM_FLASHMLA_SPARSE:
        raise RuntimeError(sparse_fixture.VLLM_FLASHMLA_SPARSE_UNAVAILABLE_REASON)
    sparse_fixture.FlashmlaSparseBenchmark("flash_mla_sparse_fwd_w8a8_fp8").run()
