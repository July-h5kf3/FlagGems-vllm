# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import flaggems_vllm
from tests.test_flash_mla_with_kvcache_sparse_w8a8_int8 import (
    SUPPORTED,
    assert_accuracy,
    make_inputs,
    reference,
)

from . import base
from .test_blas_perf_parallel import ParallelBenchmarkMixin


def run_native(inputs, query, cache, handle):
    from importlib import import_module

    adapter = import_module(
        f"flaggems_vllm.runtime.backend._{flaggems_vllm.vendor_name}.testing.flash_mla_with_kvcache_sparse_w8a8_int8"
    )
    return adapter.run_baseline(inputs, query, cache)


def run_gems(inputs, query, cache, handle):
    return handle()


class SparseMLAInt8Benchmark(ParallelBenchmarkMixin, base.Benchmark):
    RECORD_PARALLEL_RESULTS = True

    def __init__(self):
        super().__init__(
            "flash_mla_with_kvcache_sparse_w8a8_int8", run_native, [torch.bfloat16]
        )
        self.set_gems(run_gems)

    def get_input_iter(self, dtype):
        for batch, queries, heads, topk, pages, rope_dim in self.shapes:
            inputs, query, cache = make_inputs(
                batch, queries, heads, topk, pages, dtype, rope_dim
            )
            handle, actual = (
                flaggems_vllm.prepare_flash_mla_with_kvcache_sparse_w8a8_int8(*inputs)
            )
            torch.backends.cuda.matmul.allow_tf32 = False
            expected = reference(inputs)
            assert_accuracy(actual, expected)
            assert_accuracy(run_native(inputs, query, cache, handle), expected)
            yield inputs, query, cache, {"handle": handle}

    def record_shapes(self, inputs, query, cache, handle):
        return (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            inputs[-1].shape[-1],
            cache.shape[0],
            inputs[1].shape[-1],
        )


@pytest.mark.flash_mla_with_kvcache_sparse_w8a8_int8
@pytest.mark.skipif(not SUPPORTED, reason="No registered sparse INT8 KVCache MLA")
def test_flash_mla_with_kvcache_sparse_w8a8_int8():
    SparseMLAInt8Benchmark().run()
