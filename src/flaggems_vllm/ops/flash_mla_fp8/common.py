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


# flake8: noqa: E501,F841

from __future__ import annotations

import triton.language as tl

from flaggems_vllm.utils.triton_version_utils import has_triton_tle

HAS_TRITON = True
HAS_TLE = has_triton_tle(3, 6, 0)
if HAS_TLE:
    import triton.experimental.tle.language as tle
else:
    tle = None

D_CKV = 512  # content / V head dim

D_ROPE = 64  # rope tail dim

PAGE_SIZE = 64  # paged KV cache page size (= BK)

FP8_MAX = 448.0  # E4M3 dynamic range upper bound

LOG2E = 1.4426950408889634

LN2 = 0.6931471805599453

P_AMAX_FLOOR = 1e-26

TLE_FP8_BH = 64  # heads per iteration

K_CONTENT_TILE_HOST = 128

K_CONTENT_TILE = (
    tl.constexpr(K_CONTENT_TILE_HOST) if HAS_TRITON else K_CONTENT_TILE_HOST
)

DEFAULT_PAGES_PER_SPLIT = 2

MAX_SEQUENCE_LENGTH = 33280

NUM_SLOTS = (
    tl.constexpr(2) if HAS_TRITON else 2 if HAS_TRITON else 2 if HAS_TRITON else 2
)

_PACK32_F32_BASE_CONSTRAINTS = ",".join(["=r"] * 32 + ["f"] * 32 + ["r"] * 32)

_TLE_LOG2E = tl.constexpr(LOG2E)

_TLE_LN2 = tl.constexpr(LN2)

_TLE_FP8_MAX = tl.constexpr(FP8_MAX)

_TLE_P_AMAX_FLOOR = tl.constexpr(P_AMAX_FLOOR)

_TLE_NEG_INF = tl.constexpr(float("-inf"))

__all__ = ["HAS_TLE", "HAS_TRITON", "tle"]
