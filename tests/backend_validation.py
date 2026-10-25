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

from importlib import import_module
from pathlib import Path
from types import ModuleType

import flaggems_vllm


def backend_validation_module(operator: str) -> ModuleType | None:
    vendor = import_module(
        f"flaggems_vllm.runtime.backend._{flaggems_vllm.vendor_name}"
    )
    source = Path(vendor.__file__).parent / "testing" / f"{operator}.py"
    if not source.is_file():
        return None
    return import_module(f"{vendor.__name__}.testing.{operator}")
