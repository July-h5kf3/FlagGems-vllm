# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Enable the operator-scoped CommonIR bridge for CANN 9.0."""
from flaggems_vllm.runtime.backend._ascend.ops.marlin_w4a16 import common_ir as compat

compat.install_cann90_custom_op_compat()
