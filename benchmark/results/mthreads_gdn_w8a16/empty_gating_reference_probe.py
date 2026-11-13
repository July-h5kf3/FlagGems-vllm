"""Proposed acceptance-only binding; execute only after approval of the baseline packaging."""

import hashlib
import importlib.util
from pathlib import Path

root = Path(__file__).resolve().parent
source = root / "vendor-source-review/fused_gdn_gating.py"
assert (
    hashlib.sha256(source.read_bytes()).hexdigest()
    == "c592256d7ca4622207432d64cd3ff42495cddb6bad8721defd7604f03c258bf8"
)


def pytest_collection_modifyitems(items):
    spec = importlib.util.spec_from_file_location("gdn_vendor_gate_reference", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for item in items:
        if item.nodeid.startswith("benchmark/test_gdn_w8a16.py"):
            item.module._mthreads_gates = module.fused_gdn_gating
