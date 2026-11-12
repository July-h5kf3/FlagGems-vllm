# Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple

import triton


def group_short_pages(
    rows: int, pages: int, programs: int
) -> Optional[Tuple[int, int, int, int]]:
    groups = math.gcd(rows, programs)
    group_rows, group_programs = rows // groups, programs // groups
    if group_rows == 1 or pages < group_programs:
        return None
    max_pages = triton.cdiv(group_rows * pages, group_programs)
    row_programs = triton.cdiv(pages, max_pages)
    row_slack = row_programs * max_pages - pages
    if row_slack == 0:
        return None
    # Use spare page capacity to align queries without lengthening the busiest CTA.
    spare_pages = group_programs * max_pages - group_rows * pages
    prefix_rows = min(group_rows - 1, spare_pages // row_slack)
    return (
        (group_rows, group_programs, prefix_rows, row_programs) if prefix_rows else None
    )
