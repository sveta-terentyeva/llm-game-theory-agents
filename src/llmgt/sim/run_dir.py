from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _git_sha_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out or "nogit"
    except Exception:
        return "nogit"


@dataclass(frozen=True)
class RunDir:
    root: Path

    @property
    def runs_dir(self) -> Path:
        return self.root

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"


def make_run_dir(base: Path = Path("data/runs"), tag: str = "run") -> RunDir:
    stamp = _utc_stamp()
    sha = _git_sha_short()
    root = base / f"{stamp}_{sha}_{tag}"
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    return RunDir(root=root)
