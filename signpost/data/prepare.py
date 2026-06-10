from __future__ import annotations

"""F3 package entry for dataset preparation.

The existing `scripts/prepare_datasets.py` already performs the heavy lifting.
This module keeps that script available while adding the planned package command:

    python -m signpost.data.prepare --validate-only
"""

import importlib.util
from pathlib import Path

from signpost.config.context import PROJECT_ROOT


def _load_script_module():
    script_path = PROJECT_ROOT / "scripts" / "prepare_datasets.py"
    spec = importlib.util.spec_from_file_location("_signpost_prepare_datasets", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load dataset preparation script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_script_module()
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())

