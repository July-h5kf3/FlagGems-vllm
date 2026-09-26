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
from benchmark.test_int8_einsum import (
    EINSUM_LOW_PRECISION_DTYPE,
    _einsum_low_precision_available,
    _gems_einsum_bf16_wrapper,
    _make_block_einsum_inputs,
)

from .conftest import QUICK_MODE

_EINSUM_BATCHES = (1, 4, 8, 16, 32, 64, 128)
if not QUICK_MODE:
    _EINSUM_BATCHES += (4096, 8192, 16384, 32768)
_EINSUM_BLOCK_SHAPES = [
    (b, h, r, 1024) for h, r in [(8, 4096), (16, 7168)] for b in _EINSUM_BATCHES
]


@pytest.mark.int8_einsum
@pytest.mark.skipif(
    not _einsum_low_precision_available(), reason="requires Hygon or MetaX INT8"
)
@pytest.mark.parametrize("shape", _EINSUM_BLOCK_SHAPES)
def test_accuracy_int8_einsum(shape):
    x, xs, y, ys, xf, yf = _make_block_einsum_inputs(
        *shape, (128, 128), flaggems_vllm.device, EINSUM_LOW_PRECISION_DTYPE
    )
    out = flaggems_vllm.int8_einsum("bhr,hdr->bhd", x, xs, y, ys)
    b, h, r, d = shape
    assert out.shape == (b, h, d) and out.is_contiguous()
    assert torch.isfinite(out).all()
    rows = torch.linspace(0, b - 1, min(b, 32), device=x.device).long()
    cols = torch.linspace(0, d - 1, min(d, 32), device=x.device).long()
    kk = torch.arange(r, device=x.device) // 128
    xd = x[rows].float() * xs[rows][:, :, kk]
    yd = y[:, cols].float() * ys[:, cols // 128, :][:, :, kk]
    ref = torch.einsum("bhr,hdr->bhd", xd, yd)
    original = torch.einsum("bhr,hdr->bhd", xf[rows].float(), yf[:, cols].float())
    sampled = out[rows][:, :, cols].float()
    nrms = ((sampled - ref).square().mean() / ref.square().mean()).sqrt().item()
    total = (
        ((sampled - original).square().mean() / original.square().mean()).sqrt().item()
    )
    print(f"shape={shape} dequant_nrms={nrms:.6f} total_nrms={total:.6f}")
    kernel_limit = 0.01 if flaggems_vllm.vendor_name == "metax" else 0.10
    assert nrms < kernel_limit and total < 0.10
    # Validate the floating precision route for the same layouts, including
    # the largest interleaved input whose element offsets exceed int32.
    floating = _gems_einsum_bf16_wrapper(xf, None, yf, None, xf, yf)
    floating_sample = floating[rows][:, :, cols].float()
    floating_nrms = (
        ((floating_sample - original).square().mean() / original.square().mean())
        .sqrt()
        .item()
    )
    assert floating_nrms < 0.01


@pytest.mark.einsum
@pytest.mark.skipif(
    flaggems_vllm.vendor_name not in ("hygon", "metax"),
    reason="Hygon or MetaX precision dispatch",
)
@pytest.mark.parametrize("shape", [(3, 2, 129, 33), (16, 4, 256, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_einsum_precision_route(shape, dtype):
    torch.manual_seed(0)
    b, h, r, d = shape
    x = torch.randn((b, h, r), dtype=dtype, device=flaggems_vllm.device)
    y = torch.randn((h, d, r), dtype=dtype, device=x.device)
    out = flaggems_vllm.int8_einsum(
        "bhr,hdr->bhd", x, None, y, None, output_dtype=dtype
    )
    ref = torch.einsum("bhr,hdr->bhd", x.float(), y.float())
    error = ((out.float() - ref).square().mean() / ref.square().mean()).sqrt()
    assert error.item() < 0.01


@pytest.mark.int8_einsum
@pytest.mark.skipif(
    flaggems_vllm.vendor_name not in ("hygon", "metax"), reason="Hygon or MetaX INT8"
)
@pytest.mark.parametrize("layout", ["contiguous", "offset", "padded", "broadcast"])
@pytest.mark.parametrize("shape", [(16, 2, 64, 32), (32, 2, 128, 128), (3, 2, 129, 33)])
def test_int8_einsum_layouts(shape, layout):
    torch.manual_seed(0)
    b, h, r, d = shape
    if layout == "offset":
        x = torch.randint(
            -128, 128, (b * h * r + 1,), dtype=torch.int8, device=flaggems_vllm.device
        )[1:].view(b, h, r)
    elif layout == "padded":
        x = torch.randint(
            -128, 128, (b, h, r + 1), dtype=torch.int8, device=flaggems_vllm.device
        )[:, :, :r]
    elif layout == "broadcast":
        x = torch.randint(
            -128, 128, (1, h, r), dtype=torch.int8, device=flaggems_vllm.device
        ).expand(b, -1, -1)
    else:
        x = torch.randint(
            -128, 128, (b, h, r), dtype=torch.int8, device=flaggems_vllm.device
        )
    y = torch.randint(-128, 128, (h, d, r), dtype=torch.int8, device=x.device)
    xs = torch.rand((b, h, (r + 127) // 128), device=x.device) * 0.01
    ys = torch.rand((h, (d + 127) // 128, (r + 127) // 128), device=x.device) * 0.01
    out = flaggems_vllm.int8_einsum("bhr,hdr->bhd", x, xs, y, ys)
    kk = torch.arange(r, device=x.device) // 128
    nn = torch.arange(d, device=x.device) // 128
    ref = torch.einsum(
        "bhr,hdr->bhd", x.float() * xs[:, :, kk], y.float() * ys[:, nn, :][:, :, kk]
    )
    nrms = ((out.float() - ref).square().mean() / ref.square().mean()).sqrt()
    assert torch.isfinite(out).all() and nrms.item() < 0.10


@pytest.mark.int8_einsum
@pytest.mark.skipif(
    flaggems_vllm.vendor_name not in ("hygon", "metax"), reason="Hygon or MetaX INT8"
)
@pytest.mark.parametrize(
    "shape", [(0, 2, 128, 32), (3, 0, 128, 32), (3, 2, 0, 32), (3, 2, 128, 0)]
)
@pytest.mark.parametrize(
    "dtype", [torch.int8, torch.bfloat16, torch.float16, torch.float32]
)
def test_int8_einsum_empty(shape, dtype):
    b, h, r, d = shape
    x = torch.empty((b, h, r), dtype=dtype, device=flaggems_vllm.device)
    y = torch.empty((h, d, r), dtype=dtype, device=x.device)
    if dtype == torch.int8:
        xs = torch.ones((b, h, (r + 127) // 128), device=x.device)
        ys = torch.ones((h, (d + 127) // 128, (r + 127) // 128), device=x.device)
        output_dtype = torch.bfloat16
    else:
        xs, ys = None, None
        output_dtype = dtype
    out = flaggems_vllm.int8_einsum(
        "bhr,hdr->bhd", x, xs, y, ys, output_dtype=output_dtype
    )
    assert out.shape == (b, h, d) and out.dtype == output_dtype
    assert torch.count_nonzero(out) == 0


@pytest.mark.int8_einsum
@pytest.mark.skipif(
    flaggems_vllm.vendor_name not in ("hygon", "metax"), reason="Hygon or MetaX INT8"
)
def test_int8_einsum_validation_and_extremes():
    x = torch.full((16, 2, 64), -128, dtype=torch.int8, device=flaggems_vllm.device)
    y = torch.full((2, 32, 64), 127, dtype=torch.int8, device=x.device)
    y[:, ::2, :] = -128
    xs = torch.full((16, 2, 1), 0.5, device=x.device)
    ys = torch.full((2, 1, 1), 0.25, device=x.device)
    out = flaggems_vllm.int8_einsum("bhr,hdr->bhd", x, xs, y, ys)
    ref = (torch.einsum("bhr,hdr->bhd", x.float(), y.float()) * 0.125).bfloat16()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    with pytest.raises(ValueError, match="equation|supports"):
        flaggems_vllm.int8_einsum("bij,bjk->bik", x, xs, y, ys)
    with pytest.raises(ValueError, match="scale"):
        flaggems_vllm.int8_einsum("bhr,hdr->bhd", x, None, y, ys)
    with pytest.raises(TypeError, match="matching"):
        flaggems_vllm.int8_einsum("bhr,hdr->bhd", x, xs, y.float(), ys)


@pytest.mark.int8_einsum
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX INT8")
@pytest.mark.parametrize("output_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("block_n", [16, 128, 256])
def test_int8_einsum_scale_and_output_dtype(output_dtype, scale_dtype, block_n):
    torch.manual_seed(0)
    batch, heads, reduction, columns = 5, 2, 257, 65
    x = torch.randint(
        -128,
        128,
        (batch, heads, reduction * 2),
        device=flaggems_vllm.device,
        dtype=torch.int8,
    )[..., ::2]
    y = torch.randint(
        -128, 128, (heads, columns, reduction * 2), device=x.device, dtype=torch.int8
    )[..., ::2]
    xs = (
        torch.rand(batch, heads, 6, dtype=scale_dtype, device=x.device)[..., ::2] * 0.01
    )
    ys = (
        torch.rand(
            heads,
            (columns + block_n - 1) // block_n,
            6,
            dtype=scale_dtype,
            device=x.device,
        )[..., ::2]
        * 0.01
    )
    out = flaggems_vllm.int8_einsum(
        "bhr,hdr->bhd",
        x,
        xs,
        y,
        ys,
        block_size=(block_n, 128),
        output_dtype=output_dtype,
    )
    groups = torch.arange(reduction, device=x.device) // 128
    column_groups = torch.arange(columns, device=x.device) // block_n
    reference = torch.einsum(
        "bhr,hdr->bhd",
        x.float() * xs.float()[..., groups],
        y.float() * ys.float()[:, column_groups][:, :, groups],
    )
    torch.testing.assert_close(out, reference.to(output_dtype), rtol=0.01, atol=0.002)


@pytest.mark.int8_einsum
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX split-K")
@pytest.mark.parametrize("batch", [1, 17, 128])
def test_int8_einsum_split_reduction(batch):
    torch.manual_seed(0)
    x = torch.randint(
        -128, 128, (batch, 1, 513), device=flaggems_vllm.device, dtype=torch.int8
    )
    y = torch.randint(-128, 128, (1, 65, 513), device=x.device, dtype=torch.int8)
    xs = torch.rand(batch, 1, 5, device=x.device) * 0.01
    ys = torch.rand(1, 1, 5, device=x.device) * 0.01
    groups = torch.arange(513, device=x.device) // 128
    reference = torch.einsum(
        "bhr,hdr->bhd", x.float() * xs[..., groups], y.float() * ys[..., groups]
    )
    out = flaggems_vllm.int8_einsum(
        "bhr,hdr->bhd", x, xs, y, ys, output_dtype=torch.float32
    )
    torch.testing.assert_close(out, reference, rtol=2e-4, atol=2e-4)


@pytest.mark.int8_einsum
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX GEMV")
@pytest.mark.parametrize("reduction", [2048, 4097, 7168, 8192])
@pytest.mark.parametrize("heads", [1, 2])
def test_int8_einsum_single_row(reduction, heads):
    torch.manual_seed(0)
    x = torch.randint(
        -128,
        128,
        (1, heads, reduction * 2),
        device=flaggems_vllm.device,
        dtype=torch.int8,
    )[..., ::2]
    y = torch.randint(
        -128, 128, (heads, 65, reduction), device=x.device, dtype=torch.int8
    )
    groups = (reduction + 127) // 128
    xs = torch.rand((1, heads, groups), device=x.device) * 0.01
    ys = torch.rand((heads, 1, groups), device=x.device) * 0.01
    indices = torch.arange(reduction, device=x.device) // 128
    reference = torch.einsum(
        "bhr,hdr->bhd", x.float() * xs[..., indices], y.float() * ys[..., indices]
    )
    output = flaggems_vllm.int8_einsum(
        "bhr,hdr->bhd", x, xs, y, ys, output_dtype=torch.float32
    )
    torch.testing.assert_close(output, reference, rtol=2e-4, atol=2e-4)


@pytest.mark.int8_einsum
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX split-K")
@pytest.mark.parametrize("batch", [193, 224, 225, 256, 384, 385, 400, 448, 449, 512])
def test_int8_einsum_medium_batch_split(batch):
    torch.manual_seed(0)
    reduction, columns = 513, 1024
    x = torch.randint(
        -128, 128, (batch, 1, reduction), device=flaggems_vllm.device, dtype=torch.int8
    )
    y = torch.randint(
        -128, 128, (1, columns, reduction), device=x.device, dtype=torch.int8
    )
    xs = torch.rand(batch, 1, 5, device=x.device) * 0.01
    ys = torch.rand(1, 8, 5, device=x.device) * 0.01
    groups = torch.arange(reduction, device=x.device) // 128
    col_groups = torch.arange(columns, device=x.device) // 128
    reference = torch.einsum(
        "bhr,hdr->bhd",
        x.float() * xs[..., groups],
        y.float() * ys[:, col_groups][:, :, groups],
    )
    output = flaggems_vllm.int8_einsum(
        "bhr,hdr->bhd", x, xs, y, ys, output_dtype=torch.float32
    )
    torch.testing.assert_close(output, reference, rtol=2e-4, atol=2e-4)


@pytest.mark.int8_einsum
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX scale layout")
@pytest.mark.parametrize("batch", [8191, 8192, 8193])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("block_n", [16, 128])
def test_int8_einsum_large_batch_scale_layout(batch, strided, block_n):
    torch.manual_seed(0)
    reduction, columns, groups = 129, 65, 2
    step = 2 if strided else 1
    x = torch.randint(
        -128,
        128,
        (batch, 1, reduction * step),
        device=flaggems_vllm.device,
        dtype=torch.int8,
    )[..., ::step]
    y = torch.randint(
        -128, 128, (1, columns, reduction * step), device=x.device, dtype=torch.int8
    )[..., ::step]
    xs_storage = torch.rand(batch, 1, groups * step, device=x.device) * 0.01
    ys_storage = (
        torch.rand(
            1, (columns + block_n - 1) // block_n, groups * step, device=x.device
        )
        * 0.01
    )
    xs, ys = xs_storage[..., ::step], ys_storage[..., ::step]
    indices = torch.arange(reduction, device=x.device) // 128
    column_groups = torch.arange(columns, device=x.device) // block_n
    reference = torch.einsum(
        "bhr,hdr->bhd",
        x.float() * xs[..., indices],
        y.float() * ys[:, column_groups][:, :, indices],
    )
    output = flaggems_vllm.int8_einsum(
        "bhr,hdr->bhd",
        x,
        xs,
        y,
        ys,
        block_size=(block_n, 128),
        output_dtype=torch.float32,
    )
    torch.testing.assert_close(output, reference, rtol=2e-4, atol=2e-4)


@pytest.mark.int8_einsum
@pytest.mark.skipif(flaggems_vllm.vendor_name != "metax", reason="MetaX scale layout")
def test_int8_einsum_large_batch_nonfinite_scales():
    batch, reduction, columns = 8192, 129, 17
    x = torch.ones((batch, 1, reduction), device=flaggems_vllm.device, dtype=torch.int8)
    y = torch.ones((1, columns, reduction), device=x.device, dtype=torch.int8)
    xs = torch.full((batch, 1, 2), 0.125, device=x.device)
    ys = torch.full((1, 1, 2), 0.125, device=x.device)
    xs[0, 0, 0] = float("nan")
    xs[1, 0, 0] = float("inf")
    xs[2, 0, 0] = -float("inf")
    xs[3, 0, 0] = -0.0
    xs[4, 0, 0], xs[4, 0, 1] = float("inf"), -float("inf")
    output = flaggems_vllm.int8_einsum(
        "bhr,hdr->bhd", x, xs, y, ys, output_dtype=torch.float32
    )
    # The block-scaled contract applies each scale after its integer dot.
    reference = torch.zeros_like(output)
    for group, length in enumerate((128, 1)):
        reference += length * (xs[:, :, group, None] * ys[:, :, group])
    torch.testing.assert_close(output, reference, rtol=0, atol=0, equal_nan=True)
