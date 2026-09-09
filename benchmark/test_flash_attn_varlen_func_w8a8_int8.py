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

from . import base
from .test_flash_attn_varlen_func import FlashAttnVarlenBenchmark

vendor_name = flaggems_vllm.vendor_name


class FlashAttnVarlenInt8Benchmark(FlashAttnVarlenBenchmark):
    """Reuse upstream paged and PR #749 packed/ragged workloads."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Packed/ragged shapes from FlagGems-vllm PR #749, plus the existing
        # four paged/GQA workloads inherited from the upstream benchmark.
        shapes = []
        for batch in (1, 2, 4, 8):
            for heads, dim in ((16, 128), (32, 64)):
                for causal in (False, True):
                    shapes.append((batch, 512, heads, dim, causal))
            for length in (1024, 2048, 4096, 8192):
                for heads, dim in ((16, 128), (32, 64)):
                    shapes.append((batch, length, heads, dim, False))
        shapes.extend([(8, 8192, 16, 128, True), (8, 8192, 32, 64, True)])
        qlens, klens = (32, 128, 512, 4096), (1, 17, 129, 8192)
        for heads, dim in ((32, 64), (16, 128)):
            for causal in (False, True):
                shapes.append((qlens, klens, heads, dim, causal))
        core = [
            (1, 512, 16, 128, False),
            (1, 512, 32, 64, False),
            (2, 512, 16, 128, True),
            (1, 2048, 32, 64, False),
            (4, 4096, 32, 64, False),
            (8, 8192, 16, 128, True),
            (qlens, klens, 32, 64, False),
        ]
        self.shapes += (
            shapes
            if base.Config.bench_level == base.consts.BenchLevel.COMPREHENSIVE
            else core
        )

    def flash_attn_varlen_input_fn(self, config, dtype, device):
        if len(config) != 5:
            return super().flash_attn_varlen_input_fn(config, dtype, device)
        batch, length, heads, dim, causal = config
        if isinstance(batch, (tuple, list)):
            qlens, klens = batch, length
        else:
            step = max(1, length // (2 * batch))
            qlens = tuple(max(1, length - i * step - i % 3) for i in range(batch))
            klens = qlens[1:] + qlens[:1]
        q = torch.empty((sum(qlens), heads, dim), device=device, dtype=dtype).uniform_(
            -0.05, 0.05
        )
        k = torch.empty((sum(klens), heads, dim), device=device, dtype=dtype).uniform_(
            -0.05, 0.05
        )
        v = torch.empty_like(k).uniform_(-0.05, 0.05)
        cuq = torch.tensor(
            (0,) + tuple(qlens), device=device, dtype=torch.int32
        ).cumsum(0, dtype=torch.int32)
        cuk = torch.tensor(
            (0,) + tuple(klens), device=device, dtype=torch.int32
        ).cumsum(0, dtype=torch.int32)
        return (
            q,
            k,
            v,
            max(qlens),
            cuq,
            max(klens),
            cuk,
            None,
            None,
            0.0,
            dim**-0.5,
            causal,
            (-1, -1),
            0.0,
            None,
            False,
            False,
            None,
            False,
            torch.empty_like(q),
            {},
        )

    def get_input_iter(self, dtype):
        for bf16_args in super().get_input_iter(dtype):
            q, k, v = bf16_args[:3]
            batch = bf16_args[4].numel() - 1
            quantized, descales, dequantized = [], [], []
            for x, max_len in ((q, bf16_args[3]), (k, bf16_args[5]), (v, bf16_args[5])):
                # A head-wise scale shared by logical blocks also handles shared
                # physical cache pages in the upstream random block tables.
                axes = tuple(i for i in range(x.ndim) if i != x.ndim - 2)
                scale = x.float().abs().amax(axes).clamp_min(1e-8) / 127
                quantized.append(
                    (x.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
                )
                dequantized.append((quantized[-1].float() * scale[:, None]).to(dtype))
                descales.append(
                    scale[None, :, None].expand(
                        batch, x.shape[-2], (max_len + 127) // 128
                    )
                )
            int8_args = list(bf16_args)
            int8_args[:3] = quantized
            int8_args[19] = torch.empty_like(q)
            int8_args[-1] = dict(
                bf16_args[-1],
                q_descale=descales[0],
                k_descale=descales[1],
                v_descale=descales[2],
            )
            int8_args = tuple(int8_args)
            # Check the same quantized values, separating kernel error from
            # input quantization error. Timing still uses the original BF16 inputs.
            reference_args = (*dequantized, *bf16_args[3:])
            torch.testing.assert_close(
                _varlen_int8(bf16_args, int8_args),
                _varlen_bf16_baseline(reference_args, int8_args),
                atol=0.03,
                rtol=0.03,
            )
            yield bf16_args, int8_args


def _varlen_bf16_baseline(bf16_args, int8_args):
    return flaggems_vllm.flash_attn_varlen_func(*bf16_args[:-1], **bf16_args[-1])


def _varlen_int8(bf16_args, int8_args):
    return flaggems_vllm.flash_attn_varlen_func(*int8_args[:-1], **int8_args[-1])


@pytest.mark.skipif(vendor_name != "thead", reason="PPU-only API")
@pytest.mark.flash_attn_varlen_func_w8a8_int8
def test_flash_attn_varlen_func_w8a8_int8():
    # PPU vLLM 0.19 FA2 rejects INT8; FA3 accepts FP16/BF16/FP8 only.
    # latency_base is explicitly FlagGems-vllm BF16, not a native INT8 result.
    print("Baseline: FlagGems-vllm BF16; input quantization is excluded from timing.")
    bench = FlashAttnVarlenInt8Benchmark(
        op_name="flash_attn_varlen_func_w8a8_int8",
        torch_op=_varlen_bf16_baseline,
        gems_op=_varlen_int8,
        dtypes=[torch.bfloat16],
    )
    bench.run()
