from __future__ import annotations

import json
from pathlib import Path

from .records import EpisodeRecord


class JsonlLogger:
    def __init__(self, out_dir: Path, filename: str = "episodes.jsonl") -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / filename

    def log_episode(self, rec: EpisodeRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.model_dump(), ensure_ascii=False) + "\n")

