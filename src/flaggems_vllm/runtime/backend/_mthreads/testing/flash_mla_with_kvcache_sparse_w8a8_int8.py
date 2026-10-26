# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0


def run_baseline(inputs, query, cache):
    from mate.flashmla import flash_mla_sparse_fwd

    batch, queries, heads, dims = query.shape
    output, _, lse = flash_mla_sparse_fwd(
        query.view(batch * queries, heads, dims),
        cache.view(-1, 1, dims),
        inputs[-1].view(batch * queries, 1, -1),
        dims**-0.5,
        512,
    )
    return output.view(batch, queries, heads, 512), lse.view(
        batch, queries, heads
    ).transpose(1, 2)
