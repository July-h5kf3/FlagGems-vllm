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

import triton.language as tl

D_CKV = 512  # content / V head dim

D_ROPE = 64  # rope tail dim

PAGE_SIZE = 64  # paged KV cache page size (= BK)

FP8_MAX = 448.0  # E4M3 dynamic range upper bound

LOG2E = 1.4426950408889634

LN2 = 0.6931471805599453

P_AMAX_FLOOR = 1e-26

TLE_FP8_BH = 64  # heads per iteration

K_CONTENT_TILE_HOST = 128

K_CONTENT_TILE = tl.constexpr(K_CONTENT_TILE_HOST)

DEFAULT_PAGES_PER_SPLIT = 2

MAX_SEQUENCE_LENGTH = 33280

TLE_LOG2E = tl.constexpr(LOG2E)

TLE_LN2 = tl.constexpr(LN2)

TLE_FP8_MAX = tl.constexpr(FP8_MAX)

TLE_P_AMAX_FLOOR = tl.constexpr(P_AMAX_FLOOR)

TLE_NEG_INF = tl.constexpr(float("-inf"))

ADAPTIVE_MODEL_MIN_PAGES = 69

ADAPTIVE_MIN_FIXED_PAGES = 4

ADAPTIVE_MAX_FIXED_PAGES = 32

ADAPTIVE_TAIL_WAVE_MAX_FIXED_PAGES = 34

ADAPTIVE_CTA_PENALTY = 0.5

CUDA_REF_FIXED_OVERHEAD_PAGES = 5

TLE_POS_INF = tl.constexpr(float("inf"))

D_QK = 576  # Q/K head dim (content 512 + rope 64)

TLE_FP8_BK = 64  # KV tokens per iteration (= PAGE_SIZE)

TLE_FP8_DPH = 256  # output 512 dim split into left/right halves of 256

COMBINE_BLOCK_SPLITS = 8

COMBINE_BLOCK_D = 128

CUDA_COARSE_COMBINE_BLOCK_SPLITS = 32

CUDA_COARSE_COMBINE_BLOCK_ROWS = 8

CUDA_COARSE_COMBINE_MIN_BATCH = 4

LSE_FINALIZE_BLOCK = 256
