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

from benchmark import test_flash_mla_sparse_fwd as sparse_fixture


@pytest.mark.flash_mla_sparse_fwd_w8a8_fp8
@pytest.mark.skipif(
    not sparse_fixture.HAS_FP8_MLA, reason="FP8 MLA requires NVIDIA Hopper and TLE"
)
def test_flash_mla_sparse_fp8_long_reduction(monkeypatch):
    # Real record 195 exposes the error from persistent FP8 PV accumulation.
    monkeypatch.setattr(sparse_fixture, "_flashmla_sparse_counter", 195)
    param = sparse_fixture.TestParam(547, 98816, 8320, h_q=64, d_qk=512)
    inputs = next(sparse_fixture.FlashmlaSparseBenchmark.make_input_flashmla(param))
    reference, _, reference_lse = sparse_fixture.vllm_flash_mla_sparse_fwd(*inputs)
    packed = sparse_fixture.quantize_sparse_fp8_fixture(*inputs)
    output, lse = sparse_fixture.run_sparse_fp8_fixture(*packed)
    sparse_fixture.assert_sparse_fixture_close(
        output[:, 0], lse[:, :, 0], reference, reference_lse
    )
