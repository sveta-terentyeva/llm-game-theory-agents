"""Append-mode JSONL logger for episode records."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .records import EpisodeRecord


class JsonlLogger:
    """Writes ``EpisodeRecord`` objects as one-JSON-per-line to a file.

    Thread-safe: concurrent calls to :meth:`log_episode` are serialised
    via an internal lock.

    Parameters
    ----------
    out_dir : Path
        Directory where the log file is created.
    filename : str
        Name of the JSONL file inside *out_dir*.
    overwrite : bool
        If *True* (default), delete any existing file before writing.
    """

    def __init__(self, out_dir: Path, filename: str = "episodes.jsonl", *, overwrite: bool = True) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / filename
        self._lock = threading.Lock()

        if overwrite and self.path.exists():
            self.path.unlink()

    def log_episode(self, rec: EpisodeRecord) -> None:
        """Append a single episode record as a JSON line (thread-safe)."""
        line = json.dumps(rec.model_dump(), ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
