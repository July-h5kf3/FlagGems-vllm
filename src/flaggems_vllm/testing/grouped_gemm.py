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

"""Reference-only grouped GEMM inputs shared by accuracy and performance tests."""

import torch


def make_grouped_gemm_case(lengths, n, k, device, seed=20261019):
    torch.manual_seed(seed)
    m, e = sum(lengths), len(lengths)
    A = torch.randint(-128, 128, (m, k), dtype=torch.int8, device=device)
    B = torch.randint(-128, 128, (e, n, k), dtype=torch.int8, device=device)
    As = 0.005 + 0.01 * torch.rand((m, 1), device=device)
    Bs = 0.005 + 0.01 * torch.rand((e, n, 1), device=device)
    C = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    rows = torch.tensor(lengths, dtype=torch.int32, device=device)
    indices = torch.repeat_interleave(
        torch.arange(e, dtype=torch.int32, device=device), rows, output_size=m
    )
    return A, As, B, Bs, C, indices, rows


def deepgemm_grouped_gemm(A, As, B, Bs, C, m_indices, m_rows):
    import deep_gemm

    deep_gemm.m_grouped_gemm_int8_int8_bf16_nt_nopad(
        (A, As), (B, Bs), C, m_indices, m_rows
    )


def grouped_gemm_cpu_reference(A, As, B, Bs, lengths):
    A, As, B, Bs = (t.cpu() for t in (A, As, B, Bs))
    output = torch.empty((A.shape[0], B.shape[1]), dtype=torch.bfloat16)
    start = 0
    for expert, count in enumerate(lengths):
        end = start + count
        dot = A[start:end].double() @ B[expert].double().T
        output[start:end] = (dot.float() * As[start:end] * Bs[expert, :, 0]).bfloat16()
        start = end
    return output
