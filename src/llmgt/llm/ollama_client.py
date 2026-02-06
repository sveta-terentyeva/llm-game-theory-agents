from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Sequence, Optional

from llmgt.llm.client import LLMMessage


@dataclass
class OllamaChatClient:
    """
    Local Ollama backend via HTTP API (default: http://localhost:11434).
    Uses POST /api/chat with stream=false.
    """
    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    temperature_default: float = 0.7
    num_predict: int = 128
    timeout_s: float = 120.0

    def complete(self, messages: Sequence[LLMMessage], *, temperature: float | None = None) -> str:
        temp = self.temperature_default if temperature is None else temperature

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {
                "temperature": float(temp),
                "num_predict": int(self.num_predict),
            },
        }

        url = self.host.rstrip("/") + "/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(
                f"Ollama HTTPError {e.code}: {e.reason}. Body: {body[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Is Ollama running? ({e})"
            ) from e

        try:
            obj = json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"Invalid JSON from Ollama. Raw: {raw[:500]}") from e

        # Expected: {"message": {"role":"assistant","content":"..."} , ...}
        content = (obj.get("message", {}) or {}).get("content", "")
        content = (content or "").strip()
        return content if content else "OK"
