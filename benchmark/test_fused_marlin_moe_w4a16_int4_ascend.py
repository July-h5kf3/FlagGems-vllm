# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""PR #741's 53 shapes against the same-precision AscendC W4A16 chain."""

import pytest
import torch

import flaggems_vllm

from . import base


class MarlinAscendBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        from .marlin_ascend_utils import MODEL_GEOMETRY, PR5140_TRACE

        self.shapes = [(m, *MODEL_GEOMETRY, calls) for m, calls in PR5140_TRACE]
        self.shape_desc = "M, E, K, N, top_k, calls"

    def get_input_iter(self, dtype):
        from .marlin_ascend_utils import baseline, gems_call, inputs, weights

        ww = weights(256, 4096, 256, dtype)
        for m, e, k, n, t, calls in self.shapes:
            x, p, ids = inputs(m, e, k, t)
            torch.testing.assert_close(
                gems_call(x, ww, p, ids), baseline(x, ww, p, ids), rtol=0.02, atol=0.02
            )
            yield (x, ww, p, ids)


@pytest.mark.fused_marlin_moe_w4a16_int4
@pytest.mark.skipif(
    flaggems_vllm.device != "npu", reason="Requires Ascend and torch_npu"
)
def test_fused_marlin_moe_w4a16_int4():
    from .marlin_ascend_utils import baseline, gems_call

    b = MarlinAscendBenchmark(
        op_name="fused_marlin_moe_w4a16_int4",
        torch_op=baseline,
        dtypes=[torch.bfloat16],
    )
    b.set_gems(gems_call)
    b.run()
