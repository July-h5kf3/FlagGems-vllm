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


import triton
import triton.experimental.tle as tle
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.experimental.tle.language.dsa.ascend import custom_ops  # noqa: F401

from ._flash_attn_varlen_int8_head_replay_vector import vector_wide as vector_replay
from ._flash_attn_varlen_int8_head_vector import vector_aligned, vector_generic
from ._flash_attn_varlen_int8_head_vector import vector_head_n256 as vector_n256
from ._flash_attn_varlen_int8_head_vector import vector_head_n512 as vector_n512
from ._flash_attn_varlen_int8_vector import (
    vector_grouped,
    vector_hybrid_grouped,
    vector_hybrid_small,
    vector_packed,
)

MTE2 = tl.constexpr(tle.dsa.TilePipe.MTE2.value)


MTE1 = tl.constexpr(tle.dsa.TilePipe.MTE1.value)


M = tl.constexpr(tle.dsa.TilePipe.M.value)


FIX = tl.constexpr(tle.dsa.TilePipe.FIX.value)


@triton.jit
def local_sync(PRODUCER: tl.constexpr, CONSUMER: tl.constexpr):
    tle.dsa.tile_set_flag(PRODUCER, CONSUMER, 0)
    tle.dsa.tile_wait_flag(PRODUCER, CONSUMER, 0)


@triton.jit
def load_matrix_a(
    src_address,
    dst_address,
    ROWS,
    CHANNELS: tl.constexpr,
    K,
    k_start,
    RESET_MATRIX: tl.constexpr = 1,
    RESET_PADDING: tl.constexpr = 0,
    load_rows=None,
):
    matrix_rows = ROWS if load_rows is None else load_rows
    al.custom(
        "cube_load3d_a_into",
        src_address,
        [0, 0, 0, 255],
        1,
        ROWS,
        CHANNELS,
        K,
        matrix_rows,
        k_start,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        RESET_MATRIX,
        RESET_PADDING,
        dst_address,
    )


@triton.jit
def copy_page_run(
    SRC,
    physical,
    count,
    address,
    head,
    PS: tl.constexpr,
    RS: tl.constexpr,
    C: tl.constexpr,
    nz_rows,
    FEATURES: tl.constexpr = 128,
):
    ptr = SRC.to(tl.uint64) + (physical.to(tl.uint32) * PS + head * 128).to(tl.uint64)
    al.custom(
        "cube_nd2nz_i8", address, ptr, 1, count * 16, FEATURES, 0, RS, nz_rows, 1, 0
    )


@triton.jit
def copy_paged_tile(
    SRC,
    TABLE,
    batch,
    head,
    tile,
    nk,
    base_address,
    TS: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    PADDED_ROWS: tl.constexpr = False,
    FEATURES: tl.constexpr = 128,
):
    pages = (nk + 15) >> 4
    for part in tl.static_range(N // C):
        start = tile * (N // 16) + part * (C // 16)
        count = tl.minimum(C // 16, tl.maximum(0, pages - start))
        if count > 0:
            table_base = TABLE + (batch * TS + start).to(tl.uint64)
            first = tl.load(table_base)
            run = 1
            written = 0
            for page in range(1, count):
                following = tl.load(table_base + page)
                if following == first + run:
                    run += 1
                else:
                    copy_page_run(
                        SRC,
                        first,
                        run,
                        base_address + part * C * FEATURES + written * 16 * 32,
                        head,
                        PAGE_STRIDE,
                        ROW_STRIDE,
                        C,
                        C if PADDED_ROWS else ((count + 1) >> 1) * 32,
                        FEATURES,
                    )
                    written += run
                    first = following
                    run = 1
            copy_page_run(
                SRC,
                first,
                run,
                base_address + part * C * FEATURES + written * 16 * 32,
                head,
                PAGE_STRIDE,
                ROW_STRIDE,
                C,
                C if PADDED_ROWS else ((count + 1) >> 1) * 32,
                FEATURES,
            )
        else:
            pass


@triton.jit
def compute_score(
    KEY,
    TABLE,
    SCORE,
    core,
    tile,
    batch,
    head,
    nk,
    real_queries,
    L1_KEY: tl.constexpr,
    TS: tl.constexpr,
    PS: tl.constexpr,
    RS: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    SCORE_ROWS: tl.constexpr = 128,
    FIXED_KEYS: tl.constexpr = False,
):
    copy_paged_tile(
        KEY, TABLE, batch, head, tile, nk, L1_KEY, TS, PS, RS, N, C, FIXED_KEYS
    )
    local_sync(MTE2, MTE1)
    for part in tl.static_range(N // C):
        valid_keys = C if FIXED_KEYS else tl.minimum(C, nk - tile * N - part * C)
        if valid_keys > 0:
            key_rows = (valid_keys + 31) & -32
            al.custom(
                "cube_load2d_b_into",
                L1_KEY + part * C * 128,
                0,
                key_rows * 128 // 512,
                1,
                0,
                0,
                0,
                0,
                0,
            )
            local_sync(MTE1, MTE2)
            local_sync(MTE1, M)
            tle.dsa.tile_wait_flag(FIX, M, 0)
            q_rows = (real_queries + 15) & -16
            al.custom("cube_mmad_into", 0, 0, q_rows, 128, key_rows, 3, 0, 0, 1, 0)
            local_sync(M, MTE1)
            local_sync(M, FIX)
            score_ptr = SCORE + ((core * 2 + tile % 2) * SCORE_ROWS * N + part * C) * 4
            al.custom(
                "cube_copy_l0c2gm_i32",
                score_ptr,
                0,
                key_rows,
                real_queries,
                N,
                q_rows,
                3,
                0,
                0,
                0,
                1,
            )
            tle.dsa.tile_set_flag(FIX, M, 0)
        else:
            pass
    al.sync_block_set("cube", "vector", 2 + 6 * (tile % 2))


@triton.jit
def probability_product(
    PROB,
    PRODUCT,
    prob_offset,
    product_offset,
    L1_PROB: tl.constexpr,
    L0_PROB: tl.constexpr,
    ROWS: tl.constexpr,
    STORE_ROWS: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    L1_VALUE: tl.constexpr = 0,
    LOAD_VALUE: tl.constexpr = False,
    real_rows=None,
    real_store=None,
    active_keys=None,
    STREAM_VALUE: tl.constexpr = False,
    VALUE_NZ_ROWS: tl.constexpr = None,
    EXTRA_PLANES: tl.constexpr = 2,
    EXTRA_ROWS: tl.constexpr = 0,
    EXTRA_STORE_ROWS: tl.constexpr = 0,
    EXTRA_PROB=None,
    EXTRA_PRODUCT=None,
    extra_prob_offset=0,
    extra_product_offset=0,
):
    key_count = N if active_keys is None else active_keys
    active_rows = ROWS if real_rows is None else real_rows
    stored_rows = STORE_ROWS if real_store is None else real_store
    matrix_rows = active_rows + EXTRA_PLANES * EXTRA_ROWS
    ptr = PROB + prob_offset
    al.custom("cube_nd2nz_i8", L1_PROB, ptr, 1, active_rows, N, 0, N, matrix_rows, 1, 0)
    if EXTRA_ROWS:
        for half in tl.static_range(EXTRA_PLANES):
            al.custom(
                "cube_nd2nz_i8",
                L1_PROB + (active_rows + half * EXTRA_ROWS) * 32,
                EXTRA_PROB + extra_prob_offset + half * EXTRA_ROWS * N,
                1,
                EXTRA_ROWS,
                N,
                0,
                N,
                matrix_rows,
                1,
                0,
            )
    else:
        pass
    local_sync(MTE2, MTE1)
    for part in tl.static_range(N // C):
        if part * C < key_count:
            key_rows = tl.minimum(C, ((key_count - part * C + 31) & -32))
            load_matrix_a(
                L1_PROB,
                L0_PROB,
                matrix_rows,
                N,
                key_rows,
                part * C,
                1 if part == 0 else 0,
                0,
            )
            if LOAD_VALUE and (part == 0 or STREAM_VALUE):
                for value_index in tl.static_range(1 if STREAM_VALUE else N // C):
                    value_part = part if STREAM_VALUE else value_index
                    value_rows = tl.minimum(
                        C, ((key_count - value_part * C + 31) & -32)
                    )
                    for section in tl.static_range(C // 32):
                        if section * 32 < value_rows:
                            al.custom(
                                "cube_load_transpose_b_into",
                                L1_VALUE + value_part * C * 128 + section * 32 * 32,
                                0,
                                4,
                                (
                                    value_rows // 32
                                    if VALUE_NZ_ROWS is None
                                    else VALUE_NZ_ROWS // 32
                                ),
                                1,
                                0,
                                0,
                                (0 if STREAM_VALUE else value_part * C * 128)
                                + section * 32 * 128,
                            )
                        else:
                            pass
            else:
                pass
            local_sync(MTE1, MTE2)
            local_sync(MTE1, M)
            if part == 0:
                tle.dsa.tile_wait_flag(FIX, M, 0)
            else:
                pass
            al.custom(
                "cube_mmad_into",
                L0_PROB,
                0 if STREAM_VALUE else part * C * 128,
                matrix_rows,
                key_rows,
                128,
                0,
                0,
                0,
                1 if part == 0 else 0,
                0,
            )
            local_sync(M, MTE1)
        else:
            pass
    local_sync(M, FIX)
    output = PRODUCT + product_offset * 4
    al.custom(
        "cube_copy_l0c2gm_i32",
        output,
        0,
        128,
        stored_rows,
        128,
        matrix_rows,
        0,
        0,
        0,
        0,
        1,
    )
    if EXTRA_ROWS:
        for half in tl.static_range(EXTRA_PLANES):
            extra_output = (
                EXTRA_PRODUCT
                + (extra_product_offset + half * EXTRA_STORE_ROWS * 128) * 4
            )
            al.custom(
                "cube_copy_l0c2gm_i32",
                extra_output,
                (active_rows + half * EXTRA_ROWS) * 16 * 4,
                128,
                EXTRA_STORE_ROWS,
                128,
                matrix_rows,
                0,
                0,
                0,
                0,
                1,
            )
    else:
        pass
    tle.dsa.tile_set_flag(FIX, M, 0)


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
    C: tl.constexpr = 128 if N == 128 else 256
    L1_KEY: tl.constexpr = 128 * 128
    L1_VALUE: tl.constexpr = L1_KEY + N * 128
    L1_PROB: tl.constexpr = L1_VALUE + N * 128
    L0_PROB: tl.constexpr = 128 * 128
    tl.static_assert(L1_PROB + 144 * N <= 512 * 1024)
    tl.static_assert(L0_PROB + 144 * C <= 64 * 1024)
    tl.static_assert(N * 128 <= 64 * 1024)
    tl.static_assert(128 * C * 4 <= 128 * 1024)
    tl.static_assert(144 * 128 * 4 <= 128 * 1024)
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
        SB: tl.constexpr = META[30]
        PB: tl.constexpr = META[31]
        VB: tl.constexpr = META[32]
        SCORE = Workspace.to(tl.uint64)
        PROB = SCORE + SB
        PRODUCT = SCORE + SB + PB
        EXTRA_P = SCORE + SB + PB + VB
        EXTRA_PRODUCT = SCORE + SB + PB + VB + META[36]
        if N == 512:
            CONTROL = WorkspaceI32 + (SB + PB + VB + META[36] + META[37]) // 4
        else:
            pass
        al.custom("cube_set_l0c_copy_params", 1, 0, 0)
        tle.dsa.tile_set_flag(FIX, M, 0)
        for ordinal in range(tl.cdiv(GROUPS, BLOCKS)):
            group = core + ordinal * BLOCKS
            if group < GROUPS:
                batch = group // (QGROUPS * HQ)
                head = group % HQ
                kvhead = head // (HQ // HK)
                qstart = (group // HQ % QGROUPS * 128).to(tl.int32)
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
                if MODE != 4 and QLEN % 128 == 0:
                    real_queries = 128
                else:
                    real_queries = tl.minimum(128, query_length - qstart)
                nk = tl.load(Used + batch)
                if MODE == 5:
                    twice_visible = 2 * nk.to(tl.int64) - (QLEN - 1 if CAUSAL else 0)
                    use_wide = twice_visible >= (6144 if CAUSAL else 4096)
                    active_group = use_wide == (N == 512)
                else:
                    pass
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
                            128,
                            1,
                            0,
                        )
                        local_sync(MTE2, MTE1)
                        load_matrix_a(
                            0,
                            0,
                            128,
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
                                )
                            else:
                                pass
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
                            al.sync_block_wait("vector", "cube", 3 + 6 * (tile % 2))
                            tile_keys = tl.minimum(N, nk - tile * N)
                            ring = core * 2 + tile % 2
                            if real_queries < 128:
                                aligned_rows = (real_queries + 15) & -16
                                probability_product(
                                    PROB,
                                    PRODUCT,
                                    ring * 272 * N,
                                    ring * 257 * 128,
                                    L1_PROB,
                                    L0_PROB,
                                    128,
                                    128,
                                    N,
                                    C,
                                    L1_VALUE,
                                    True,
                                    aligned_rows,
                                    real_queries,
                                    active_keys=tile_keys,
                                )
                                probability_product(
                                    PROB,
                                    PRODUCT,
                                    ring * 272 * N + 128 * N,
                                    ring * 257 * 128 + 128 * 128,
                                    L1_PROB,
                                    L0_PROB,
                                    16,
                                    1,
                                    N,
                                    C,
                                    active_keys=tile_keys,
                                )
                                probability_product(
                                    PROB,
                                    PRODUCT,
                                    ring * 272 * N + 144 * N,
                                    ring * 257 * 128 + 129 * 128,
                                    L1_PROB,
                                    L0_PROB,
                                    128,
                                    128,
                                    N,
                                    C,
                                    0,
                                    False,
                                    aligned_rows,
                                    real_queries,
                                    active_keys=tile_keys,
                                )
                            else:
                                probability_product(
                                    PROB,
                                    PRODUCT,
                                    ring * 272 * N,
                                    ring * 257 * 128,
                                    L1_PROB,
                                    L0_PROB,
                                    144,
                                    129,
                                    N,
                                    C,
                                    L1_VALUE,
                                    True,
                                    active_keys=tile_keys,
                                )
                                probability_product(
                                    PROB,
                                    PRODUCT,
                                    ring * 272 * N + 144 * N,
                                    ring * 257 * 128 + 129 * 128,
                                    L1_PROB,
                                    L0_PROB,
                                    128,
                                    128,
                                    N,
                                    C,
                                    active_keys=tile_keys,
                                )
                            if N == 512:
                                mask = tl.load(CONTROL + ring * 16, volatile=True) | (
                                    tl.load(CONTROL + ring * 16 + 8, volatile=True) << 2
                                )
                                for chunk in range(4):
                                    if mask & (1 << chunk):
                                        probability_product(
                                            EXTRA_P,
                                            EXTRA_PRODUCT,
                                            ring * 128 * N + chunk * 32 * N,
                                            ring * 128 * 128 + chunk * 32 * 128,
                                            L1_PROB,
                                            L0_PROB,
                                            32,
                                            32,
                                            N,
                                            C,
                                            active_keys=tile_keys,
                                        )
                                    else:
                                        pass
                            else:
                                pass
                            al.sync_block_set("cube", "vector", 4 + 6 * (tile % 2))
                    else:
                        pass
                else:
                    pass
            else:
                pass
        tle.dsa.tile_wait_flag(FIX, M, 0)
        tl.debug_barrier()
    if N == 128 and MODE == 3:
        if META[15] % 128 == 0 and META[10] <= META[9]:
            vector_aligned(
                QueryScale,
                KeyScale,
                ValueScale,
                Output,
                Workspace,
                WorkspaceI32,
                Used,
                Cuq,
                META,
            )
        else:
            vector_generic(
                QueryScale,
                KeyScale,
                ValueScale,
                Output,
                Workspace,
                WorkspaceI32,
                Used,
                Cuq,
                META,
            )
    elif N == 256:
        vector_n256(
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
    elif N == 512:
        vector_n512(
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
    else:
        tl.static_assert(False, "Unsupported head geometry")


@triton.jit
def grouped_cube(
    Query,
    Key,
    Value,
    Table,
    Used,
    Workspace,
    Cuq,
    N: tl.constexpr,
    META: tl.constexpr,
    SELECTOR: tl.constexpr,
):
    tl.static_assert(N == 128 or N == 256 or N == 512 or N == 1024)
    C: tl.constexpr = 512 if N >= 512 else N
    EXTRA_ROWS: tl.constexpr = tl.cdiv(META[14] * 2, 16) * 16
    QUERY_TILE: tl.constexpr = META[14]
    ROWS: tl.constexpr = QUERY_TILE * 4
    MATRIX_ROWS: tl.constexpr = tl.cdiv(ROWS, 16) * 16
    PRODUCT_ROWS: tl.constexpr = 2 * ROWS + 1
    PRODUCT_MATRIX_ROWS: tl.constexpr = tl.cdiv(PRODUCT_ROWS, 16) * 16
    LIVE_PROB_ROWS: tl.constexpr = PRODUCT_MATRIX_ROWS + (
        2 * EXTRA_ROWS if N >= 512 else 0
    )
    L1_KEY: tl.constexpr = MATRIX_ROWS * 128
    L1_VALUE: tl.constexpr = L1_KEY + N * 128
    L1_PROB: tl.constexpr = L1_VALUE + N * 128
    L0_PROB: tl.constexpr = MATRIX_ROWS * 128
    tl.static_assert(L1_PROB + LIVE_PROB_ROWS * N <= 512 * 1024)
    tl.static_assert(L0_PROB + LIVE_PROB_ROWS * C <= 64 * 1024)
    tl.static_assert(C * 128 <= 64 * 1024)
    tl.static_assert(MATRIX_ROWS * C * 4 <= 128 * 1024)
    tl.static_assert(LIVE_PROB_ROWS * 128 * 4 <= 128 * 1024)
    core = tl.program_id(0).to(tl.uint32)
    with al.scope("cube"):
        HQ: tl.constexpr = META[0]
        HK: tl.constexpr = META[1]
        TS: tl.constexpr = META[4]
        PS: tl.constexpr = META[5]
        RS: tl.constexpr = META[6]
        BLOCKS: tl.constexpr = META[9]
        GROUPS: tl.constexpr = META[10]
        QGROUPS: tl.constexpr = META[16]
        CAUSAL: tl.constexpr = META[18]
        SCORE = Workspace.to(tl.uint64)
        PROB = SCORE + META[30]
        PRODUCT = PROB + META[31]
        if N >= 512:
            EXTRA_PROB = PRODUCT + META[32]
            EXTRA_PRODUCT = EXTRA_PROB + META[36]
        else:
            pass
        al.custom("cube_set_l0c_copy_params", 1, 0, 0)
        tle.dsa.tile_set_flag(FIX, M, 0)
        for ordinal in range(tl.cdiv(GROUPS, BLOCKS)):
            group = core + ordinal * BLOCKS
            if group < GROUPS:
                batch = group // (QGROUPS * HK)
                kvhead = group % HK
                qstart = (group // HK % QGROUPS * QUERY_TILE).to(tl.int32)
                query_base = tl.load(Cuq + batch).to(tl.uint32)
                query_length = (tl.load(Cuq + batch + 1).to(tl.uint32) - query_base).to(
                    tl.int32
                )
                if SELECTOR == 1:
                    selected = query_length <= 4
                elif SELECTOR == 2:
                    selected = query_length > 4
                else:
                    selected = True
                if selected and qstart < query_length:
                    real_queries = tl.minimum(QUERY_TILE, query_length - qstart)
                    nk = tl.load(Used + batch)
                    if CAUSAL:
                        visible = tl.minimum(
                            nk, tl.maximum(0, nk - query_length + qstart + real_queries)
                        )
                    else:
                        visible = nk
                    tiles = tl.cdiv(visible, N)
                    if tiles > 0:
                        for row in tl.static_range(QUERY_TILE):
                            safe_row = tl.where(row < real_queries, row, 0)
                            query_ptr = Query.to(tl.uint64) + (
                                (
                                    (query_base + qstart.to(tl.uint32) + safe_row) * HQ
                                    + kvhead * 4
                                )
                                * 128
                            ).to(tl.uint64)
                            al.custom(
                                "cube_nd2nz_i8",
                                row * 4 * 32,
                                query_ptr,
                                1,
                                4,
                                128,
                                0,
                                128,
                                MATRIX_ROWS,
                                1,
                                0,
                            )
                        local_sync(MTE2, MTE1)
                        load_matrix_a(0, 0, MATRIX_ROWS, 128, 128, 0)
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
                            ROWS,
                            L1_KEY,
                            TS,
                            PS,
                            RS,
                            N,
                            C,
                            ROWS,
                            True,
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
                                    ROWS,
                                    L1_KEY,
                                    TS,
                                    PS,
                                    RS,
                                    N,
                                    C,
                                    ROWS,
                                    True,
                                )
                            else:
                                pass
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
                                True,
                            )
                            if N >= 512:
                                al.sync_block_wait(
                                    "vector",
                                    "cube",
                                    3 + 6 * (tile % 2),
                                    sender_pipe=al.PIPE.PIPE_MTE3,
                                    receiver_pipe=al.PIPE.PIPE_S,
                                )
                            else:
                                al.sync_block_wait("vector", "cube", 3 + 6 * (tile % 2))
                            probability_product(
                                PROB,
                                PRODUCT,
                                (core * 2 + tile % 2) * PRODUCT_MATRIX_ROWS * N,
                                (core * 2 + tile % 2) * PRODUCT_ROWS * 128,
                                L1_PROB,
                                L0_PROB,
                                PRODUCT_MATRIX_ROWS,
                                PRODUCT_ROWS,
                                N,
                                C,
                                L1_VALUE,
                                True,
                                active_keys=tl.minimum(N, nk - tile * N),
                                STREAM_VALUE=N == 1024,
                                VALUE_NZ_ROWS=C,
                                EXTRA_ROWS=EXTRA_ROWS if N >= 512 else 0,
                                EXTRA_STORE_ROWS=ROWS // 2 if N >= 512 else 0,
                                EXTRA_PROB=EXTRA_PROB if N >= 512 else None,
                                EXTRA_PRODUCT=EXTRA_PRODUCT if N >= 512 else None,
                                extra_prob_offset=(core * 2 + tile % 2)
                                * 2
                                * EXTRA_ROWS
                                * N,
                                extra_product_offset=(core * 2 + tile % 2) * ROWS * 128,
                            )
                            al.sync_block_set("cube", "vector", 4 + 6 * (tile % 2))
                    else:
                        pass
                else:
                    pass
            else:
                pass
        tle.dsa.tile_wait_flag(FIX, M, 0)
        tl.debug_barrier()


@triton.jit
def launch_hybrid_small_tle(
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
    BUNDLE_KEY: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    META: tl.constexpr,
):
    tl.static_assert(MODE == 1 and N == 256 and META[14] == 2)
    grouped_cube(Query, Key, Value, Table, Used, Workspace, Cuq, N, META, 1)
    vector_hybrid_small(
        QueryScale, KeyScale, ValueScale, Output, Workspace, Used, Cuq, META
    )


@triton.jit
def launch_hybrid_large_tle(
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
    BUNDLE_KEY: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    META: tl.constexpr,
):
    tl.static_assert(N == 256 and MODE == 2 and META[14] == 16)
    grouped_cube(Query, Key, Value, Table, Used, Workspace, Cuq, N, META, 2)
    vector_hybrid_grouped(
        QueryScale, KeyScale, ValueScale, Output, Workspace, Used, Cuq, META, 2
    )


@triton.jit
def launch_grouped_tle(
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
    BUNDLE_KEY: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    META: tl.constexpr,
):
    tl.static_assert(MODE == 0)
    if META[4] == 0 or META[7] == 0 or META[8] == 0:
        batch: tl.constexpr = META[10] // (META[16] * META[1])
        total = tl.load(Cuq + batch).to(tl.int64) * META[0] * 128
        offsets = tl.arange(0, 256)
        for start in range(tl.program_id(0) * 256, total, tl.num_programs(0) * 256):
            tl.store(Output + start + offsets, 0, start + offsets < total)
    else:
        grouped_cube(Query, Key, Value, Table, Used, Workspace, Cuq, N, META, 0)
        vector_grouped(
            QueryScale, KeyScale, ValueScale, Output, Workspace, Used, Cuq, N, META
        )


@triton.jit
def packed_score(
    Key,
    Table,
    Score,
    core,
    tile,
    batch,
    kvhead,
    nk,
    TS: tl.constexpr,
    PS: tl.constexpr,
    RS: tl.constexpr,
):
    copy_paged_tile(
        Key, Table, batch, kvhead, tile, nk, 8192, TS, PS, RS, 512, 512, True, 512
    )
    local_sync(MTE2, MTE1)
    for head in tl.static_range(4):
        load_matrix_a(head * 2048, 0, 16, 128, 128, 0)
        al.custom(
            "cube_load2d_b_into",
            8192 + head * 512 * 128,
            0,
            512 * 128 // 512,
            1,
            0,
            0,
            0,
            0,
            0,
        )
        local_sync(MTE1, MTE2)
        local_sync(MTE1, M)
        tle.dsa.tile_wait_flag(FIX, M, 0)
        al.custom("cube_mmad_into", 0, 0, 16, 128, 512, 0, 0, 0, 1, 0)
        local_sync(M, MTE1)
        local_sync(M, FIX)
        destination = Score + ((core * 2 + tile % 2) * 16 * 512 + head * 4 * 512) * 4
        al.custom(
            "cube_copy_l0c2gm_i32", destination, 0, 512, 4, 512, 16, 0, 0, 0, 0, 1
        )
        tle.dsa.tile_set_flag(FIX, M, 0)
    al.sync_block_set("cube", "vector", 2 + 6 * (tile % 2))


@triton.jit
def packed_cube(Query, Key, Value, Table, Used, Workspace, Cuq, META: tl.constexpr):
    tl.static_assert(270336 + 32 * 512 <= 512 * 1024)
    tl.static_assert(2048 + 32 * 512 <= 64 * 1024)
    tl.static_assert(512 * 128 <= 64 * 1024)
    tl.static_assert(16 * 512 * 4 <= 128 * 1024)
    core = tl.program_id(0).to(tl.uint32)
    with al.scope("cube"):
        HQ: tl.constexpr = META[0]
        HK: tl.constexpr = META[1]
        TS: tl.constexpr = META[4]
        PS: tl.constexpr = META[5]
        RS: tl.constexpr = META[6]
        BLOCKS: tl.constexpr = META[9]
        GROUPS: tl.constexpr = META[10]
        SCORE = Workspace.to(tl.uint64)
        PROB = SCORE + META[30]
        PRODUCT = PROB + META[31]
        EXTRA_PROB = PRODUCT + META[32]
        EXTRA_PRODUCT = EXTRA_PROB + META[36]
        al.custom("cube_set_l0c_copy_params", 1, 0, 0)
        tle.dsa.tile_set_flag(FIX, M, 0)
        for ordinal in range(tl.cdiv(GROUPS, BLOCKS)):
            group = core + ordinal * BLOCKS
            if group < GROUPS:
                batch = group // (HK // 4)
                kvhead = (group % (HK // 4)) * 4
                query_base = tl.load(Cuq + batch).to(tl.uint32)
                nk = tl.load(Used + batch)
                tiles = tl.cdiv(nk, 512)
                if tiles > 0:
                    for head in tl.static_range(4):
                        query_ptr = Query.to(tl.uint64) + (
                            (query_base * HQ + kvhead * 4 + head * 4) * 128
                        ).to(tl.uint64)
                        al.custom(
                            "cube_nd2nz_i8",
                            head * 2048,
                            query_ptr,
                            1,
                            4,
                            128,
                            0,
                            128,
                            16,
                            1,
                            0,
                        )
                    local_sync(MTE2, MTE1)
                    packed_score(
                        Key, Table, SCORE, core, 0, batch, kvhead, nk, TS, PS, RS
                    )
                    for tile in range(tiles):
                        if tile + 1 < tiles:
                            packed_score(
                                Key,
                                Table,
                                SCORE,
                                core,
                                tile + 1,
                                batch,
                                kvhead,
                                nk,
                                TS,
                                PS,
                                RS,
                            )
                        else:
                            pass
                        copy_paged_tile(
                            Value,
                            Table,
                            batch,
                            kvhead,
                            tile,
                            nk,
                            8192,
                            TS,
                            PS,
                            RS,
                            512,
                            512,
                            True,
                            512,
                        )
                        al.sync_block_wait("vector", "cube", 3 + 6 * (tile % 2))
                        ring = core * 2 + tile % 2
                        for head in tl.static_range(4):
                            probability_product(
                                PROB,
                                PRODUCT,
                                (ring * 64 + head * 16) * 512,
                                (ring * 36 + head * 9) * 128,
                                270336,
                                2048,
                                16,
                                9,
                                512,
                                512,
                                8192 + head * 512 * 128,
                                True,
                                active_keys=tl.minimum(512, nk - tile * 512),
                                VALUE_NZ_ROWS=512,
                                EXTRA_PLANES=1,
                                EXTRA_ROWS=16,
                                EXTRA_STORE_ROWS=4,
                                EXTRA_PROB=EXTRA_PROB,
                                EXTRA_PRODUCT=EXTRA_PRODUCT,
                                extra_prob_offset=(ring * 4 + head) * 16 * 512,
                                extra_product_offset=(ring * 4 + head) * 4 * 128,
                            )
                        al.sync_block_set("cube", "vector", 4 + 6 * (tile % 2))
                else:
                    pass
            else:
                pass
        tle.dsa.tile_wait_flag(FIX, M, 0)
        tl.debug_barrier()


@triton.jit
def launch_packed_tle(
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
    BUNDLE_KEY: tl.constexpr,
    N: tl.constexpr,
    MODE: tl.constexpr,
    META: tl.constexpr,
):
    tl.static_assert(MODE == 6 and N == 512)
    packed_cube(Query, Key, Value, Table, Used, Workspace, Cuq, META)
    vector_packed(
        QueryScale, KeyScale, ValueScale, Output, Workspace, Used, Cuq, N, META
    )


@triton.jit
def launch_replay_tle(
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
    vector_replay(
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
