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
import torch

import flaggems_vllm
from benchmark.test_int8_einsum import _make_block_einsum_inputs

pytestmark = [
    pytest.mark.w8a8_block_int8_bmm,
    pytest.mark.skipif(flaggems_vllm.vendor_name != "hygon", reason="Hygon DCU INT8"),
]


@pytest.mark.parametrize("shape", [(3, 2, 129, 33), (128, 2, 256, 128)])
@pytest.mark.parametrize(
    "dtype", [torch.int8, torch.bfloat16, torch.float16, torch.float32]
)
@pytest.mark.parametrize("provide_output", [False, True])
def test_w8a8_block_int8_bmm_interface(shape, dtype, provide_output):
    torch.manual_seed(0)
    b, h, k, n = shape
    x, xs, y, ys, xf, yf = _make_block_einsum_inputs(
        b,
        h,
        k,
        n,
        (128, 128),
        flaggems_vllm.device,
        torch.int8 if dtype == torch.int8 else torch.bfloat16,
    )
    if dtype != torch.int8:
        x, y = x.to(dtype), y.to(dtype)
    # Match PR #3297: y/ys retain [B,N,K]/[B,N-block,K-block].
    x = x.permute(1, 0, 2)
    xs = xs.permute(1, 0, 2) if xs is not None else None
    z = (
        torch.empty((b, h, n), dtype=torch.float32, device=x.device).permute(1, 0, 2)
        if provide_output
        else None
    )
    result = flaggems_vllm.w8a8_block_int8_bmm(
        x, y, xs, ys, block_size=[128, 128], z=z, output_dtype=torch.float32
    )
    if provide_output:
        assert result is z
    assert result.shape == (h, b, n) and result.dtype == torch.float32
    if dtype == torch.int8:
        kk = torch.arange(k, device=x.device) // 128
        nn = torch.arange(n, device=x.device) // 128
        ref = torch.bmm(
            x.float() * xs[:, :, kk],
            (y.float() * ys[:, nn, :][:, :, kk]).transpose(1, 2),
        )
    else:
        ref = torch.bmm(x.float(), y.float().transpose(1, 2))
    nrms = ((result - ref).square().mean() / ref.square().mean()).sqrt()
    assert torch.isfinite(result).all() and nrms.item() < (
        0.10 if dtype == torch.int8 else 0.01
    )


def test_w8a8_block_int8_bmm_output_validation():
    x = torch.ones((2, 3, 128), dtype=torch.int8, device=flaggems_vllm.device)
    y = torch.ones((2, 128, 128), dtype=torch.int8, device=x.device)
    xs = torch.ones((2, 3, 1), device=x.device)
    ys = torch.ones((2, 1, 1), device=x.device)
    with pytest.raises(ValueError, match="shape"):
        flaggems_vllm.w8a8_block_int8_bmm(x, y, xs.transpose(1, 2), ys)
    z = torch.empty((2, 3, 128), dtype=torch.float32, device=x.device)
    with pytest.raises(ValueError, match="dtype"):
        flaggems_vllm.w8a8_block_int8_bmm(x, y, xs, ys, z=z)
    with pytest.raises(ValueError, match="scale"):
        flaggems_vllm.w8a8_block_int8_bmm(x.bfloat16(), y.bfloat16(), xs, ys)


@pytest.mark.parametrize(
    "shape,block_n",
    [((2, 3, 17, 33), 32), ((2, 17, 145, 257), 128), ((2, 32, 128, 256), 64)],
)
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_w8a8_block_int8_bmm_block_scales(shape, block_n, output_dtype, strided):
    torch.manual_seed(0)
    batch, rows, columns, reduction = shape
    step = 2 if strided else 1
    x = torch.randint(
        -128,
        128,
        (batch, rows, reduction * step),
        dtype=torch.int8,
        device=flaggems_vllm.device,
    )[:, :, ::step]
    y = torch.randint(
        -128,
        128,
        (batch, columns, reduction * step),
        dtype=torch.int8,
        device=x.device,
    )[:, :, ::step]
    reduction_blocks = (reduction + 127) // 128
    column_blocks = (columns + block_n - 1) // block_n
    xs = (
        torch.rand((batch, rows, reduction_blocks * step), device=x.device) * 0.01
        + 0.001
    ).to(output_dtype)[:, :, ::step]
    ys = (
        torch.rand((batch, column_blocks, reduction_blocks * step), device=x.device)
        * 0.01
        + 0.001
    ).to(output_dtype)[:, :, ::step]
    # Padding exposes stores beyond the caller's output view.
    storage = torch.full(
        (batch, rows, columns + 3), float("nan"), dtype=output_dtype, device=x.device
    )
    output = storage[:, :, :columns]
    actual = flaggems_vllm.w8a8_block_int8_bmm(
        x, y, xs, ys, block_size=(block_n, 128), z=output, output_dtype=output_dtype
    )
    reduction_indices = torch.arange(reduction, device=x.device) // 128
    column_indices = torch.arange(columns, device=x.device) // block_n
    reference = torch.bmm(
        x.float() * xs.float()[:, :, reduction_indices],
        (
            y.float() * ys.float()[:, column_indices, :][:, :, reduction_indices]
        ).transpose(1, 2),
    )
    assert actual is output
    assert actual.dtype == output_dtype and actual.shape == (batch, rows, columns)
    assert torch.isfinite(actual).all()
    nrms = (
        (actual.float() - reference).square().mean() / reference.square().mean()
    ).sqrt()
    assert nrms.item() < 0.015
    assert torch.isnan(storage[:, :, columns:]).all()


@pytest.mark.parametrize(
    "dtype", [torch.int8, torch.bfloat16, torch.float16, torch.float32]
)
@pytest.mark.parametrize(
    "shape", [(0, 3, 17, 128), (2, 0, 17, 128), (2, 3, 0, 128), (2, 3, 17, 0)]
)
@pytest.mark.parametrize("provide_output", [False, True])
def test_w8a8_block_int8_bmm_empty(shape, dtype, provide_output):
    batch, rows, columns, reduction = shape
    x = torch.empty((batch, rows, reduction), dtype=dtype, device=flaggems_vllm.device)
    y = torch.empty((batch, columns, reduction), dtype=dtype, device=x.device)
    if dtype == torch.int8:
        xs = torch.ones((batch, rows, (reduction + 127) // 128), device=x.device)
        ys = torch.ones(
            (batch, (columns + 127) // 128, (reduction + 127) // 128), device=x.device
        )
    else:
        xs, ys = None, None
    output = (
        torch.full(
            (rows, batch, columns), float("nan"), dtype=torch.float32, device=x.device
        ).transpose(0, 1)
        if provide_output
        else None
    )
    actual = flaggems_vllm.w8a8_block_int8_bmm(
        x, y, xs, ys, z=output, output_dtype=torch.float32
    )
    if provide_output:
        assert actual is output
    expected = torch.zeros((batch, rows, columns), dtype=torch.float32, device=x.device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
