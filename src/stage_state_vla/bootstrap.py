"""Activate and validate the external RoboCasa Isaac-GR00T checkout."""

from __future__ import annotations

import importlib
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


EXPECTED_GR00T_COMMIT = "9d7d7a9eb7ad30bd8ce30448d9ab53a918b45b10"
SUPPORTED_TRANSFORMERS_PREFIX = "4.51."


def activate_gr00t(gr00t_root: Path) -> None:
    root = Path(gr00t_root).expanduser().resolve()
    sentinel = root / "gr00t/model/gr00t_n1.py"
    if not sentinel.is_file():
        raise FileNotFoundError(
            f"Invalid GR00T_ROOT={root}: expected {sentinel}. "
            "Run scripts/bootstrap_gr00t.sh or point GR00T_ROOT at the pinned RoboCasa fork."
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()


def validate_python_environment() -> None:
    try:
        transformers_version = version("transformers")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "transformers is not installed in the active Python environment"
        ) from exc
    if not transformers_version.startswith(SUPPORTED_TRANSFORMERS_PREFIX):
        raise RuntimeError(
            f"This GR00T fork requires Transformers {SUPPORTED_TRANSFORMERS_PREFIX}x; "
            f"found {transformers_version}."
        )
