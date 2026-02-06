from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import os


@dataclass(frozen=True)
class RunDir:
    root: Path

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def make_run_dir(
    base: Path | None = None,
    tag: str = "run",
    create_standard_dirs: bool = True,
) -> RunDir:
    repo = _repo_root()
    if base is None:
        base = repo / "data" / "runs"

    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    salt = os.urandom(6)
    short = hashlib.sha1(salt).hexdigest()[:7]
    root = base / f"{ts}_{short}_{tag}"
    root.mkdir(parents=True, exist_ok=True)

    if create_standard_dirs:
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "figures").mkdir(parents=True, exist_ok=True)

    return RunDir(root=root)
