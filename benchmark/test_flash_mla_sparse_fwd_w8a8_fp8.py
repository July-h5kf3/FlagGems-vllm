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

import json
import os

import pytest
import torch
import triton

import flaggems_vllm
from tests.test_flash_mla_sparse_fwd_w8a8_fp8 import assert_accuracy, make_inputs

# D576 sparse decode shapes from FlagGems #5010's benchmark, unchanged.
STANDARD_SHAPES = [(128, 128, k) for k in (128, 256, 512, 1024, 2048)] + [
    (64, 128, k) for k in (256, 512, 1024, 2048, 4096)
]


@pytest.mark.flash_mla_sparse_fwd_w8a8_fp8
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_mla_sparse_fwd_w8a8_fp8():
    from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

    records = []
    for batch, heads, topk in STANDARD_SHAPES:
        inputs, query, cache = make_inputs(batch, heads, topk)
        sink = torch.randn(heads, device="cuda", dtype=torch.float32)
        reference_query = query[:, 0]
        reference_cache = cache.reshape(-1, 1, 576)

        def baseline():
            return flash_mla_sparse_fwd(
                reference_query, reference_cache, inputs[-1], 576**-0.5, 512, sink
            )

        def candidate():
            return flaggems_vllm.flash_mla_sparse_fwd_w8a8_fp8(*inputs, attn_sink=sink)

        reference, _, reference_lse = baseline()
        output, lse = candidate()
        assert_accuracy(output, lse, reference[:, None], reference_lse[:, :, None])
        # Paired graph runs include all operator kernels and exclude input preparation.
        measurements = []
        for _ in range(3):
            cuda_ms = triton.testing.do_bench_cudagraph(baseline, rep=100)
            fp8_ms = triton.testing.do_bench_cudagraph(candidate, rep=100)
            measurements.append((cuda_ms, fp8_ms))
        cuda_ms, fp8_ms = sorted(measurements, key=lambda pair: pair[0] / pair[1])[1]
        record = dict(
            batch=batch,
            heads=heads,
            topk=topk,
            cuda_bf16_ms=cuda_ms,
            fp8_ms=fp8_ms,
            speedup=cuda_ms / fp8_ms,
            output_relative_l2=float(
                (output[:, 0].float() - reference.float()).norm()
                / reference.float().norm().clamp_min(1e-12)
            ),
            lse_max_abs=float((lse[:, :, 0] - reference_lse).abs().max()),
            measurements=measurements,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    summary = dict(
        baseline="vLLM BF16 sparse CUDA, one query per request",
        torch_version=torch.__version__,
        triton_version=triton.__version__,
        device=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        graph_rep_ms=100,
        records=records,
        mean_speedup=sum(record["speedup"] for record in records) / len(records),
    )
    print(json.dumps(summary), flush=True)
    output_path = os.environ.get("SPARSE_FP8_BENCH_OUTPUT")
    if output_path:
        with open(output_path, "w") as output_file:
            json.dump(summary, output_file, indent=2)
