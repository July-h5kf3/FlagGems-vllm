# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import pytest
import torch

import flaggems_vllm

pytestmark = [
    pytest.mark.fused_marlin_moe_w4a16_int4,
    pytest.mark.skipif(
        flaggems_vllm.device != "npu", reason="Ascend-specific validation"
    ),
]


@pytest.fixture(scope="module")
def utils():
    from pathlib import Path

    path = Path(__file__).parents[1] / "benchmark/marlin_ascend_utils.py"
    spec = importlib.util.spec_from_file_location("marlin_ascend_test_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ww(utils):
    return utils.weights(4, 256, 128, torch.bfloat16)


def test_public_registration():
    assert flaggems_vllm.fused_marlin_moe_w4a16_int4.__module__.startswith(
        "flaggems_vllm.runtime.backend._ascend"
    )
    assert "fused_marlin_moe_w4a16_int4" in flaggems_vllm.FULL_CONFIG_BY_FUNC


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


def test_mutation_invalidates_packed_cache(utils):
    w = utils.weights(4, 256, 128, torch.bfloat16)
    x, p, ids = utils.inputs(2, 4, 256, 2)
    ids.fill_(0)
    utils.gems_call(x, w, p, ids)
    w[0][0][0].bitwise_xor_(0x10)
    w[0][1][0].mul_(1.25)
    got = utils.gems_call(x, w, p, ids)
    torch.testing.assert_close(
        got.cpu(), utils.reference(x, w, p, ids), rtol=0.01, atol=0.001
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_size": 32},
        {"activation": "relu"},
        {"inplace": True},
        {"apply_router_weight_on_input": True},
        {"global_num_experts": 8},
    ],
)
def test_unsupported_options(utils, ww, kwargs):
    from flaggems_vllm.ops.fused_marlin_moe import QUANT_TYPE_UINT4B8

    x, p, ids = utils.inputs(1, 4, 256, 2)
    with pytest.raises(NotImplementedError):
        flaggems_vllm.fused_marlin_moe_w4a16_int4(
            x,
            ww[0][0],
            ww[1][0],
            None,
            None,
            ww[0][1],
            ww[1][1],
            p,
            ids,
            QUANT_TYPE_UINT4B8,
            **kwargs,
        )


def test_fp16_explicitly_unsupported(utils, ww):
    x, p, ids = utils.inputs(1, 4, 256, 2)
    with pytest.raises(NotImplementedError):
        utils.gems_call(x.half(), ww, p, ids)


def test_noncontiguous_ids_rejected(utils, ww):
    x, p, ids = utils.inputs(2, 4, 256, 2)
    ids = ids.long().T
    with pytest.raises(NotImplementedError):
        utils.gems_call(x, ww, p, ids)
