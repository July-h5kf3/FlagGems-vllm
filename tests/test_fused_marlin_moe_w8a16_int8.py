# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib
from types import SimpleNamespace

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.fused_marlin_moe_w8a16_int8,
    pytest.mark.skipif(
        flaggems_vllm.device != "npu", reason="Ascend-specific validation"
    ),
]


@pytest.fixture(scope="module")
def utils():
    module = importlib.import_module("benchmark.test_fused_marlin_moe_w8a16_int8")
    return SimpleNamespace(
        weights=module._ascend_weights,
        inputs=module._ascend_inputs,
        baseline=module._ascend_baseline,
        reference=module._ascend_reference,
        gems_call=module._ascend_gems_call,
    )


@pytest.fixture(scope="module")
def ww(utils):
    return utils.weights(4, 256, 128, torch.bfloat16)


def test_public_registration():
    assert flaggems_vllm.fused_marlin_moe_w8a16_int8.__module__.startswith(
        "flaggems_vllm.runtime.backend._ascend"
    )
    assert "fused_marlin_moe_w8a16_int8" in flaggems_vllm.ops.__all__


@pytest.mark.parametrize("m", [0, 1, 2, 3, 7, 8, 17, 32, 33, 40, 257])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_accuracy(utils, ww, m, id_dtype):
    x, p, ids = utils.inputs(m, 4, 256, 2)
    ids = ids.to(id_dtype)
    got = utils.gems_call(x, ww, p, ids)
    if m == 0:
        assert got.shape == x.shape and got.dtype == x.dtype
        return
    expected = utils.baseline(x, ww, p, ids)
    torch.testing.assert_close(got, expected, rtol=0.01, atol=0.001)
    if m <= 8:
        torch.testing.assert_close(
            got.cpu(), utils.reference(x, ww, p, ids), rtol=0.01, atol=0.001
        )


@pytest.mark.parametrize(
    "mode", ["single_expert", "duplicate_routes", "zero_probabilities"]
)
def test_routing_edges(utils, ww, mode):
    x, p, ids = utils.inputs(33, 4, 256, 2)
    if mode == "single_expert":
        ids.fill_(0)
    elif mode == "duplicate_routes":
        ids[:, 1] = ids[:, 0]
    else:
        p.zero_()
    torch.testing.assert_close(
        utils.gems_call(x, ww, p, ids),
        utils.baseline(x, ww, p, ids),
        rtol=0.02,
        atol=0.02,
    )


@pytest.mark.parametrize("m", [1, 33])
@pytest.mark.parametrize("k,n", [(384, 128), (256, 384)])
def test_non_power_of_two_geometry(utils, m, k, n):
    weights = utils.weights(4, k, n, torch.bfloat16)
    x, p, ids = utils.inputs(m, 4, k, 2)
    got = utils.gems_call(x, weights, p, ids)
    expected = utils.baseline(x, weights, p, ids)
    torch.testing.assert_close(got, expected, rtol=0.01, atol=0.001)


@pytest.mark.parametrize("e,k,n", [(4, 7168, 128), (4, 128, 14336), (512, 128, 128)])
@pytest.mark.parametrize("m", [1, 40])
def test_extended_geometry(utils, e, k, n, m):
    weights = utils.weights(e, k, n, torch.bfloat16)
    x, p, ids = utils.inputs(m, e, k, 2)
    # Include the highest expert in both small and grouped dispatch paths.
    ids[:, 0] = e - 1
    torch.testing.assert_close(
        utils.gems_call(x, weights, p, ids).cpu(),
        utils.reference(x, weights, p, ids),
        rtol=0.02,
        atol=0.02,
    )
