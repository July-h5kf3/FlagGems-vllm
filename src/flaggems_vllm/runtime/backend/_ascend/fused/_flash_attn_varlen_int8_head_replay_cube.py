# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al

from ._flash_attn_varlen_int8_head_cube import (
    FIX,
    MTE1,
    MTE2,
    M,
    compute_score,
    copy_paged_tile,
    load_matrix_a,
    local_sync,
    probability_product,
)
from ._flash_attn_varlen_int8_head_replay_vector import vector_wide


@triton.jit
def launch_cube_tle(
    Query,
    Key,
    Value,
    Table,
    Used,
    QueryScale,
    KeyScale,
    ValueScale,
    Output,
    Workspace,
    Cuq,
    WorkspaceI32,
    BUNDLE_KEY: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    META: tl.constexpr,
):
    tl.static_assert(MODE == 3 or MODE == 4 or MODE == 5)
    Q: tl.constexpr = META[14]
    tl.static_assert(Q == 32 or Q == 64)
    tl.static_assert(N == 256 or N == 512)
    C: tl.constexpr = 256
    L1_KEY: tl.constexpr = Q * 128
    L1_VALUE: tl.constexpr = L1_KEY + N * 128
    L1_PROB: tl.constexpr = L1_VALUE + N * 128
    L0_PROB: tl.constexpr = Q * 128
    P_ROWS: tl.constexpr = 2 * Q + 16
    EXTRA: tl.constexpr = Q if N == 512 else 0
    tl.static_assert(L1_PROB + (P_ROWS + EXTRA) * N <= 512 * 1024)
    tl.static_assert(L0_PROB + (P_ROWS + EXTRA) * C <= 64 * 1024)
    tl.static_assert((P_ROWS + EXTRA) * 128 * 4 <= 128 * 1024)
    core = tl.program_id(0).to(tl.uint32)
    with al.scope(core_mode="cube"):
        HQ: tl.constexpr = META[0]
        HK: tl.constexpr = META[1]
        TS: tl.constexpr = META[4]
        PS: tl.constexpr = META[5]
        RS: tl.constexpr = META[6]
        BLOCKS: tl.constexpr = META[9]
        GROUPS: tl.constexpr = META[10]
        QLEN: tl.constexpr = META[15]
        QGROUPS: tl.constexpr = META[16]
        CAUSAL: tl.constexpr = META[18]
        Flags = WorkspaceI32 + META[38]
        SCORE = Workspace.to(tl.uint64)
        PROB = SCORE + META[30]
        PRODUCT = PROB + META[31]
        EXTRA_P = PRODUCT + META[32]
        EXTRA_PRODUCT = EXTRA_P + META[36]
        al.custom("cube_set_l0c_copy_params", 1, 0, 0)
        tle.dsa.tile_set_flag(FIX, M, 0)
        for ordinal in range(tl.cdiv(GROUPS, BLOCKS)):
            group = core + ordinal * BLOCKS
            if group < GROUPS:
                batch = group // (QGROUPS * HQ)
                head = group % HQ
                kvhead = head // (HQ // HK)
                qstart = (group // HQ % QGROUPS * Q).to(tl.int32)
                if MODE == 4:
                    query_base = tl.load(Cuq + batch).to(tl.uint32)
                    query_length = (
                        tl.load(Cuq + batch + 1).to(tl.uint32) - query_base
                    ).to(tl.int32)
                    active_group = (query_length > 4) & (qstart < query_length)
                else:
                    query_base = batch * QLEN
                    query_length = QLEN
                    active_group = True
                real_queries = tl.minimum(Q, query_length - qstart)
                nk = tl.load(Used + batch)
                if MODE == 5:
                    twice_visible = 2 * nk.to(tl.int64) - (QLEN - 1 if CAUSAL else 0)
                    active_group = (twice_visible >= (6144 if CAUSAL else 4096)) == (
                        N == 512
                    )
                fast_group = (batch * META[39] + qstart // 128) * HQ + head
                replay = tl.load(Flags + fast_group * 2, volatile=True) | tl.load(
                    Flags + fast_group * 2 + 1, volatile=True
                )
                active_group = active_group & (replay != 0)
                if active_group:
                    if CAUSAL:
                        visible = tl.minimum(
                            nk, tl.maximum(0, nk - query_length + qstart + real_queries)
                        )
                    else:
                        visible = nk
                    tiles = tl.cdiv(visible, N)
                    if tiles > 0:
                        qp = Query.to(tl.uint64) + (
                            ((query_base + qstart.to(tl.uint32)) * HQ + head) * 128
                        ).to(tl.uint64)
                        al.custom(
                            "cube_nd2nz_i8",
                            0,
                            qp,
                            1,
                            real_queries,
                            128,
                            0,
                            HQ * 128,
                            Q,
                            1,
                            0,
                        )
                        local_sync(MTE2, MTE1)
                        load_matrix_a(
                            0,
                            0,
                            Q,
                            128,
                            128,
                            0,
                            1,
                            1,
                            load_rows=(real_queries + 15) & -16,
                        )
                        local_sync(MTE1, MTE2)
                        local_sync(MTE1, M)
                        compute_score(
                            Key,
                            Table,
                            SCORE,
                            core,
                            0,
                            batch,
                            kvhead,
                            nk,
                            real_queries,
                            L1_KEY,
                            TS,
                            PS,
                            RS,
                            N,
                            C,
                            Q,
                        )
                        for tile in range(tiles):
                            if tile + 1 < tiles:
                                compute_score(
                                    Key,
                                    Table,
                                    SCORE,
                                    core,
                                    tile + 1,
                                    batch,
                                    kvhead,
                                    nk,
                                    real_queries,
                                    L1_KEY,
                                    TS,
                                    PS,
                                    RS,
                                    N,
                                    C,
                                    Q,
                                )
                            copy_paged_tile(
                                Value,
                                Table,
                                batch,
                                kvhead,
                                tile,
                                nk,
                                L1_VALUE,
                                TS,
                                PS,
                                RS,
                                N,
                                C,
                            )
                            al.sync_block_wait(
                                "vector",
                                "cube",
                                3 + 6 * (tile % 2),
                                sender_pipe=al.PIPE.PIPE_MTE3,
                                receiver_pipe=al.PIPE.PIPE_S,
                            )
                            ring = core * 2 + tile % 2
                            probability_product(
                                PROB,
                                PRODUCT,
                                ring * P_ROWS * N,
                                ring * (2 * Q + 1) * 128,
                                L1_PROB,
                                L0_PROB,
                                P_ROWS,
                                2 * Q + 1,
                                N,
                                C,
                                L1_VALUE,
                                True,
                                active_keys=tl.minimum(N, nk - tile * N),
                                EXTRA_PLANES=1,
                                EXTRA_ROWS=EXTRA,
                                EXTRA_STORE_ROWS=EXTRA,
                                EXTRA_PROB=EXTRA_P if N == 512 else None,
                                EXTRA_PRODUCT=EXTRA_PRODUCT if N == 512 else None,
                                extra_prob_offset=ring * Q * N,
                                extra_product_offset=ring * Q * 128,
                            )
                            al.sync_block_set("cube", "vector", 4 + 6 * (tile % 2))
        tle.dsa.tile_wait_flag(FIX, M, 0)
        tl.debug_barrier()
    vector_wide(
        QueryScale,
        KeyScale,
        ValueScale,
        Output,
        WorkspaceI32,
        WorkspaceI32,
        Used,
        Cuq,
        N,
        MODE,
        META,
    )
