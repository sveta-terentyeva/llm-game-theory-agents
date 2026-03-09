"""Simple, dependency-free caching wrappers for LLM clients.

Goal
----
During sweeps, the same prompt can repeat (especially in deterministic parts
of the simulation). Caching makes reruns cheaper and speeds up debugging.

Design
------
- File-backed JSON cache (one entry per request key).
- Key is a stable hash of: backend/model + temperature + messages.
- Conservative: caches only successful completions.

Config (env)
------------
- ``LLMGT_LLM_CACHE=1`` enables caching.
- ``LLMGT_LLM_CACHE_DIR`` overrides cache directory.

This is intentionally lightweight and avoids extra dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, Any

from llmgt.llm.client import LLMMessage


class _LLMClientLike(Protocol):
    def complete(self, messages: Sequence[LLMMessage], *, temperature: float | None = None) -> str: ...


def _stable_request_key(*, model_id: str, temperature: float, messages: Sequence[LLMMessage]) -> str:
    payload = {
        "model": model_id,
        "temperature": float(temperature),
        "messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class FileLLMCache:
    """Very small JSON cache stored in a directory."""

    dir_path: Path

    def __post_init__(self) -> None:
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        # 2-level fan-out to avoid too many files in one directory
        return self.dir_path / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        p = self._path_for_key(key)
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            out = obj.get("completion")
            return str(out) if out is not None else None
        except Exception:
            return None

    def set(self, key: str, completion: str) -> None:
        p = self._path_for_key(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"completion": completion}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)


@dataclass
class CachedLLMClient:
    """Wraps an LLM client and caches `complete()` calls."""

    inner: Any
    cache: FileLLMCache
    model_id: str

    def complete(self, messages: Sequence[LLMMessage], *, temperature: float | None = None) -> str:
        # Preserve inner client's default behavior if temperature isn't supplied.
        temp = float(temperature) if temperature is not None else 0.0

        # For caching, we must include a concrete temperature in the key.
        # If temperature=None, we try to use inner's default attribute; else fall back to 0.0.
        if temperature is None:
            temp = float(getattr(self.inner, "temperature_default", 0.0))

        key = _stable_request_key(model_id=self.model_id, temperature=temp, messages=messages)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        out = self.inner.complete(messages, temperature=temp)
        # Cache only non-empty strings.
        if isinstance(out, str) and out.strip():
            self.cache.set(key, out)
        return out


def caching_enabled() -> bool:
    return os.getenv("LLMGT_LLM_CACHE", "0") == "1"


def default_cache_dir() -> Path:
    # Prefer project-local cache so results are reproducible per repo checkout.
    override = os.getenv("LLMGT_LLM_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.cwd() / ".llm_cache"

