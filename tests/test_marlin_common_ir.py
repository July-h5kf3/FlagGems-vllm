# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression for CANN 9.0 side-effect custom ABI parsing."""

import importlib.util
from pathlib import Path


def test_custom_without_outputs_preserves_regions():
    source = (
        Path(__file__).resolve().parents[1]
        / "src/flaggems_vllm/runtime/backend/_ascend/ops/marlin_w4a16/common_ir.py"
    )
    spec = importlib.util.spec_from_file_location("marlin_common_ir_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mlir = """module {
  func.func @kernel(%pid: i32, %cond: i1, %buffer: memref<16xf32>) {
    scf.if %cond {
      hivm.hir.custom ins(%pid : i32) {symbol = "marlin_ft_cube_begin"}
    } else {
      hivm.hir.custom ins(%pid : i32) outs(%buffer : memref<16xf32>) {symbol = "marlin_other"}
    }
    hivm.hir.custom ins(%pid : i32) {symbol = "marlin_ft_cube_end"}
    return
  }
}
"""
    lowered = module.lower_custom_op_to_call(mlir)
    assert "} else {" in lowered
    assert lowered.count("func.call @") == 3
    assert "func.call @_mlir_ciface_marlin_ft_cube_begin(%pid) : (i32) -> ()" in lowered
    assert "func.call @_mlir_ciface_marlin_ft_cube_end(%pid) : (i32) -> ()" in lowered
    assert "hivm.hir.custom" not in lowered
