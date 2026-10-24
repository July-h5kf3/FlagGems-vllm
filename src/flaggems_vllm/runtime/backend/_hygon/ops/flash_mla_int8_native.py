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


import triton.experimental.gluon as g
import triton.experimental.gluon.language as gl
import triton.language as tl
from triton.experimental.gluon.language import (
    BlockedLayout,
    SliceLayout,
    SwizzledSharedLayout,
)
from triton.experimental.gluon.language.amd.cdna3 import buffer_load


@g.jit
def interleave_mla_pv_bytes(first, second, selector):
    return gl.inline_asm_elementwise(
        "v_perm_b32 $0, $2, $1, $3",
        constraints="=v,v,v,v",
        args=[first, second, selector],
        dtype=gl.uint32,
        is_pure=True,
        pack=1,
    )


@g.jit
def accumulate_mla_pv_limbs(High, Low, Base, HE, HO, LE, LO):
    # Keep the limb accumulators in v32..159 and the LDS read buffers in v0..31.
    return gl.inline_asm_elementwise(
        """s_waitcnt lgkmcnt(0)
v_xor_b32 v160, 64, $192
ds_read_m32x16_b16_alt v[0:3], $192 offset:0
ds_read_m32x16_b16_alt v[4:7], v160 offset:0
ds_read_m32x16_b16_alt v[8:11], $192 offset:128
ds_read_m32x16_b16_alt v[12:15], v160 offset:128
ds_read_m32x16_b16_alt v[16:19], $192 offset:256
ds_read_m32x16_b16_alt v[20:23], v160 offset:256
ds_read_m32x16_b16_alt v[24:27], $192 offset:384
ds_read_m32x16_b16_alt v[28:31], v160 offset:384
s_waitcnt lgkmcnt(7)
s_nop 0
v_mmac_i32_16x16x32_i8 v[32:35], $128, v[0:1], v[32:35]
v_mmac_i32_16x16x32_i8 v[64:67], $128, v[2:3], v[64:67]
v_mmac_i32_16x16x32_i8 v[96:99], $160, v[0:1], v[96:99]
v_mmac_i32_16x16x32_i8 v[128:131], $160, v[2:3], v[128:131]
s_waitcnt lgkmcnt(6)
s_nop 0
v_mmac_i32_16x16x32_i8 v[36:39], $128, v[4:5], v[36:39]
v_mmac_i32_16x16x32_i8 v[68:71], $128, v[6:7], v[68:71]
v_mmac_i32_16x16x32_i8 v[100:103], $160, v[4:5], v[100:103]
v_mmac_i32_16x16x32_i8 v[132:135], $160, v[6:7], v[132:135]
s_waitcnt lgkmcnt(5)
s_nop 0
v_mmac_i32_16x16x32_i8 v[40:43], $128, v[8:9], v[40:43]
v_mmac_i32_16x16x32_i8 v[72:75], $128, v[10:11], v[72:75]
v_mmac_i32_16x16x32_i8 v[104:107], $160, v[8:9], v[104:107]
v_mmac_i32_16x16x32_i8 v[136:139], $160, v[10:11], v[136:139]
s_waitcnt lgkmcnt(4)
s_nop 0
v_mmac_i32_16x16x32_i8 v[44:47], $128, v[12:13], v[44:47]
v_mmac_i32_16x16x32_i8 v[76:79], $128, v[14:15], v[76:79]
v_mmac_i32_16x16x32_i8 v[108:111], $160, v[12:13], v[108:111]
v_mmac_i32_16x16x32_i8 v[140:143], $160, v[14:15], v[140:143]
s_waitcnt lgkmcnt(3)
s_nop 0
v_mmac_i32_16x16x32_i8 v[48:51], $128, v[16:17], v[48:51]
v_mmac_i32_16x16x32_i8 v[80:83], $128, v[18:19], v[80:83]
v_mmac_i32_16x16x32_i8 v[112:115], $160, v[16:17], v[112:115]
v_mmac_i32_16x16x32_i8 v[144:147], $160, v[18:19], v[144:147]
s_waitcnt lgkmcnt(2)
s_nop 0
v_mmac_i32_16x16x32_i8 v[52:55], $128, v[20:21], v[52:55]
v_mmac_i32_16x16x32_i8 v[84:87], $128, v[22:23], v[84:87]
v_mmac_i32_16x16x32_i8 v[116:119], $160, v[20:21], v[116:119]
v_mmac_i32_16x16x32_i8 v[148:151], $160, v[22:23], v[148:151]
s_waitcnt lgkmcnt(1)
s_nop 0
v_mmac_i32_16x16x32_i8 v[56:59], $128, v[24:25], v[56:59]
v_mmac_i32_16x16x32_i8 v[88:91], $128, v[26:27], v[88:91]
v_mmac_i32_16x16x32_i8 v[120:123], $160, v[24:25], v[120:123]
v_mmac_i32_16x16x32_i8 v[152:155], $160, v[26:27], v[152:155]
s_waitcnt lgkmcnt(0)
s_nop 0
v_mmac_i32_16x16x32_i8 v[60:63], $128, v[28:29], v[60:63]
v_mmac_i32_16x16x32_i8 v[92:95], $128, v[30:31], v[92:95]
v_mmac_i32_16x16x32_i8 v[124:127], $160, v[28:29], v[124:127]
v_mmac_i32_16x16x32_i8 v[156:159], $160, v[30:31], v[156:159]
s_nop 0""",
        constraints=(
            "=&{v32},=&{v33},=&{v34},=&{v35},=&{v36},=&{v37},=&{v38},=&{v39},"
            "=&{v40},=&{v41},=&{v42},=&{v43},=&{v44},=&{v45},=&{v46},=&{v47},"
            "=&{v48},=&{v49},=&{v50},=&{v51},=&{v52},=&{v53},=&{v54},=&{v55},"
            "=&{v56},=&{v57},=&{v58},=&{v59},=&{v60},=&{v61},=&{v62},=&{v63},"
            "=&{v64},=&{v65},=&{v66},=&{v67},=&{v68},=&{v69},=&{v70},=&{v71},"
            "=&{v72},=&{v73},=&{v74},=&{v75},=&{v76},=&{v77},=&{v78},=&{v79},"
            "=&{v80},=&{v81},=&{v82},=&{v83},=&{v84},=&{v85},=&{v86},=&{v87},"
            "=&{v88},=&{v89},=&{v90},=&{v91},=&{v92},=&{v93},=&{v94},=&{v95},"
            "=&{v96},=&{v97},=&{v98},=&{v99},=&{v100},=&{v101},=&{v102},=&{v103},"
            "=&{v104},=&{v105},=&{v106},=&{v107},=&{v108},=&{v109},=&{v110},=&{v111},"
            "=&{v112},=&{v113},=&{v114},=&{v115},=&{v116},=&{v117},=&{v118},=&{v119},"
            "=&{v120},=&{v121},=&{v122},=&{v123},=&{v124},=&{v125},=&{v126},=&{v127},"
            "=&{v128},=&{v129},=&{v130},=&{v131},=&{v132},=&{v133},=&{v134},=&{v135},"
            "=&{v136},=&{v137},=&{v138},=&{v139},=&{v140},=&{v141},=&{v142},=&{v143},"
            "=&{v144},=&{v145},=&{v146},=&{v147},=&{v148},=&{v149},=&{v150},=&{v151},"
            "=&{v152},=&{v153},=&{v154},=&{v155},=&{v156},=&{v157},=&{v158},=&{v159},"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "v,v,v,v,v,v,v,v,"
            "0,1,2,3,4,5,6,7,"
            "8,9,10,11,12,13,14,15,"
            "16,17,18,19,20,21,22,23,"
            "24,25,26,27,28,29,30,31,"
            "32,33,34,35,36,37,38,39,"
            "40,41,42,43,44,45,46,47,"
            "48,49,50,51,52,53,54,55,"
            "56,57,58,59,60,61,62,63,"
            "64,65,66,67,68,69,70,71,"
            "72,73,74,75,76,77,78,79,"
            "80,81,82,83,84,85,86,87,"
            "88,89,90,91,92,93,94,95,"
            "96,97,98,99,100,101,102,103,"
            "104,105,106,107,108,109,110,111,"
            "112,113,114,115,116,117,118,119,"
            "120,121,122,123,124,125,126,127,"
            "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},"
            "~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},"
            "~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},"
            "~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},"
            "~{v160},~{memory}"
        ),
        args=[High, Low, Base, HE, HO, LE, LO],
        dtype=(gl.int32, gl.int32, gl.int32, gl.int32),
        is_pure=False,
        pack=32,
    )


@g.jit
def quantize_mla_p16_word(word, scale, NORMALIZED: gl.constexpr = False):
    # The low limb is the signed seven-bit remainder of the biased FP32 bits.
    biased_words = ()
    if NORMALIZED:
        # Pair FP32 FMAs without changing the P14 rounding or probability precision.
        raw_words = ()
        for i in gl.static_range(4):
            raw = ((word >> (16 * i)) & 65535).to(gl.uint32).to(gl.float32)
            raw_words += (raw.to(gl.uint32, bitcast=True),)
        scale_bits = scale.to(gl.uint32, bitcast=True).to(gl.uint64)
        scale_pair = scale_bits | (scale_bits << 32)
        bias_pair = gl.full(word.shape, 0x4780000047800000, gl.uint64, word.type.layout)
        for i in gl.static_range(2):
            raw_pair = raw_words[2 * i].to(gl.uint64) | (
                raw_words[2 * i + 1].to(gl.uint64) << 32
            )
            biased_pair = gl.inline_asm_elementwise(
                "v_pk_fma_f32 $0, $1, $2, $3",
                constraints="=v,v,v,v",
                args=[raw_pair, scale_pair, bias_pair],
                dtype=gl.uint64,
                is_pure=True,
                pack=1,
            )
            biased_words += (
                biased_pair.to(gl.uint32),
                (biased_pair >> 32).to(gl.uint32),
            )
    else:
        for i in gl.static_range(4):
            raw = ((word >> (16 * i)) & 65535).to(gl.uint32).to(gl.float32)
            normalized = gl.minimum(raw * scale, 127.0)
            biased = normalized + 65536.0
            biased_words += (biased.to(gl.uint32, bitcast=True),)
    pair_selector = gl.full(word.shape, 0x05040100, gl.uint32, word.type.layout)
    merge_selector = gl.full(word.shape, 0x06040200, gl.uint32, word.type.layout)
    sign_mask = gl.full(word.shape, 0x80808080, gl.uint32, word.type.layout)
    rounding_pair = gl.full(word.shape, 0x00400040, gl.uint32, word.type.layout)
    high, low = gl.inline_asm_elementwise(
        """v_perm_b32 v0, $3, $2, $6
v_perm_b32 v1, $5, $4, $6
v_perm_b32 $1, v1, v0, $7
v_lshlrev_b32 v2, 1, $1
v_bfi_b32 $1, $8, v2, $1
v_pk_add_u16 v0, v0, $9
v_pk_add_u16 v1, v1, $9
v_lshrrev_b32 v0, 7, v0
v_lshrrev_b32 v1, 7, v1
v_perm_b32 $0, v1, v0, $7""",
        constraints="=&v,=&v,v,v,v,v,v,v,v,v,~{v0},~{v1},~{v2}",
        args=[
            biased_words[0],
            biased_words[1],
            biased_words[2],
            biased_words[3],
            pair_selector,
            merge_selector,
            sign_mask,
            rounding_pair,
        ],
        dtype=(gl.uint32, gl.uint32),
        is_pure=True,
        pack=1,
    )
    if NORMALIZED:
        # Match minnum(NaN, 127) when normalization has nonfinite intermediates.
        is_nan = scale != scale
        return gl.where(is_nan, 0x7F7F7F7F, high), gl.where(is_nan, 0, low)
    return high, low


@g.jit
def expand_wide_mla_pv_word(value, N: gl.constexpr):
    BH: gl.constexpr = 128
    R: gl.constexpr = BlockedLayout([1, 1, 1], [16, 1, 4], [BH // 16, 1, 1], [0, 2, 1])
    C: gl.constexpr = BlockedLayout([1, 1], [16, 4], [BH // 16, 1], [0, 1])
    value = gl.convert_layout(value, SliceLayout(1, R))
    result = value[:, None, :] + gl.full((BH, N // 4, 4), 0, value.dtype, R)
    return gl.convert_layout(result.reshape((BH, N)), C)


@g.jit
def native_int8_mla_pv_wide(
    KV,
    Table,
    Lengths,
    Packed,
    Weight,
    RowFactor,
    Output,
    KV_STRIDES: gl.constexpr,
    TABLE_STRIDE: gl.constexpr,
    CAPACITY: gl.constexpr,
    LENGTH_STRIDE: gl.constexpr,
    H: gl.constexpr,
    SQ: gl.constexpr,
    BATCH: gl.constexpr = 0,
    PARTS: gl.constexpr = 1,
    Order=None,
    NORMALIZED: gl.constexpr = False,
):
    gl.static_assert(PARTS == 1)
    PAGE_GROUP: gl.constexpr = 2
    SHARED_BYTES: gl.constexpr = 16384 * PAGE_GROUP
    C: gl.constexpr = BlockedLayout([1, 1], [16, 4], [8, 1], [0, 1])
    W: gl.constexpr = BlockedLayout([1, 4], [4, 16], [8, 1], [1, 0])
    P3: gl.constexpr = BlockedLayout([1, 1, 2], [16, 4, 1], [8, 1, 1], [0, 2, 1])
    P2: gl.constexpr = SliceLayout(2, P3)
    dt, ht, batch = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    if Order is not None:
        batch = gl.load(Order + batch)
    mh: gl.constexpr = H * SQ
    local_heads = gl.arange(0, 128, layout=SliceLayout(1, C))
    heads = ht * 128 + local_heads
    groups = gl.arange(0, 4, layout=SliceLayout(0, C))
    ph = ht * 128 + gl.arange(0, 128, layout=SliceLayout(1, P2))
    pg = gl.arange(0, 4, layout=SliceLayout(0, P2))
    pj = gl.arange(0, 2, layout=SliceLayout(0, SliceLayout(1, P3)))
    length = gl.load(Lengths + batch * LENGTH_STRIDE)
    pages = gl.cdiv(length, 64)
    shared = gl.allocate_shared_memory(
        gl.uint8, (SHARED_BYTES,), SwizzledSharedLayout(16, 1, 1, [0])
    )
    paired = shared._reinterpret(
        gl.uint32, (32 * PAGE_GROUP, 128), SwizzledSharedLayout(4, 1, 8, [1, 0])
    )
    tail = shared.slice(SHARED_BYTES - 2048, 2048, 0)._reinterpret(
        gl.uint32, (128, 4), SwizzledSharedLayout(1, 1, 1, [0, 1])
    )
    tail.store(gl.full((128, 4), 0, gl.uint32, C))
    gl.thread_barrier()
    zero = gl.inline_asm_elementwise(
        "v_and_b32 $0, $1, 0",
        constraints="=v,v,~{memory}",
        args=[tail.load(C)],
        dtype=gl.uint32,
        is_pure=False,
        pack=1,
    )
    # Each shared halfword holds one feature from two adjacent INT8 tokens.
    source_pair = (local_heads[:, None] & 15) // 4 + groups[None, :] * 4
    base_raw = (
        source_pair * 512
        + (((local_heads[:, None] & 3) * 8) ^ ((source_pair & 7) * 8)) * 2
        + zero
    )
    base = expand_wide_mla_pv_word(base_raw, 128)
    pairs = gl.arange(0, 32, layout=SliceLayout(1, W))
    features = dt * 64 + gl.arange(0, 64, layout=SliceLayout(0, W))
    # Separate P14 limbs stay within INT32 at the 131072-token capacity limit.
    he = gl.full((128, 128), 0, gl.int32, C)
    ho = he
    le = he
    lo = he
    for group in tl.range(0, gl.cdiv(pages, PAGE_GROUP), loop_unroll_factor=1):
        gl.thread_barrier()
        for page_offset in gl.static_range(PAGE_GROUP):
            page_idx = group * PAGE_GROUP + page_offset
            page = gl.load(
                Table + batch * TABLE_STRIDE + page_idx, page_idx < pages, 0
            ).to(gl.int64)
            first_token = 2 * pairs
            first = buffer_load(
                KV.to(tl.pointer_type(gl.uint32)) + page * (KV_STRIDES[0] // 4),
                first_token[:, None] * (KV_STRIDES[1] // 4) + features[None, :],
                page_idx * 64 + first_token[:, None] < length,
                0,
            )
            second = buffer_load(
                KV.to(tl.pointer_type(gl.uint32)) + page * (KV_STRIDES[0] // 4),
                (first_token[:, None] + 1) * (KV_STRIDES[1] // 4) + features[None, :],
                page_idx * 64 + first_token[:, None] + 1 < length,
                0,
            )
            low_pair = interleave_mla_pv_bytes(first, second, 0x05010400)
            high_pair = interleave_mla_pv_bytes(first, second, 0x07030602)
            packed_pairs = gl.join(low_pair, high_pair).reshape((32, 128))
            paired.slice(page_offset * 32, 32, 0).store(packed_pairs)
        gl.thread_barrier()
        for page_offset in gl.static_range(PAGE_GROUP):
            page_idx = group * PAGE_GROUP + page_offset
            scale = buffer_load(
                Weight + (batch * CAPACITY + page_idx) * mh,
                ph,
                (ph < mh) & (page_idx < pages),
                0.0,
            )
            for half in gl.static_range(2):
                poffset = (
                    ph[:, None, None] * 16
                    + half * 8
                    + pg[None, :, None] * 2
                    + pj[None, None, :]
                )
                pwords = buffer_load(
                    Packed.to(tl.pointer_type(gl.uint64))
                    + (batch * CAPACITY + page_idx) * mh * 16,
                    poffset,
                    (ph[:, None, None] < mh) & (page_idx < pages),
                    0,
                )
                p0, p1 = gl.split(pwords)
                high0, low0 = quantize_mla_p16_word(p0, scale[:, None], NORMALIZED)
                high1, low1 = quantize_mla_p16_word(p1, scale[:, None], NORMALIZED)
                high = expand_wide_mla_pv_word(
                    high0.to(gl.uint64) | (high1.to(gl.uint64) << 32), 128
                )
                low = expand_wide_mla_pv_word(
                    low0.to(gl.uint64) | (low1.to(gl.uint64) << 32), 128
                )
                he, ho, le, lo = accumulate_mla_pv_limbs(
                    high, low, base + (2 * page_offset + half) * 8192, he, ho, le, lo
                )
    factor = buffer_load(RowFactor + batch * mh, heads, heads < mh, 0.0)
    even = (he.to(gl.float32) * 128.0 + le.to(gl.float32)) * factor[:, None]
    odd = (ho.to(gl.float32) * 128.0 + lo.to(gl.float32)) * factor[:, None]
    OUT: gl.constexpr = BlockedLayout([1, 2], [16, 4], [8, 1], [0, 1])
    result = gl.convert_layout(gl.join(even, odd).reshape((128, 256)), OUT)
    oh = ht * 128 + gl.arange(0, 128, layout=SliceLayout(1, OUT))
    of = dt * 256 + gl.arange(0, 256, layout=SliceLayout(0, OUT))
    gl.store(
        Output + (batch * mh + oh[:, None]) * 512 + of[None, :],
        result,
        oh[:, None] < mh,
    )
