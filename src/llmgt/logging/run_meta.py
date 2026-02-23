"""Write run metadata (configuration snapshot) to a JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_run_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
