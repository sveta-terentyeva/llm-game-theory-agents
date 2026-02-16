from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from llmgt.llm.client import LLMMessage


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return float(v)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return int(v)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v == "" else v


@dataclass
class OllamaChatClient:
    """
    Local Ollama backend via HTTP API (default: http://localhost:11434).
    Uses POST /api/chat with stream=false.
    """
    model: str = "mistral"
    host: str = "http://localhost:11434"
    temperature_default: float = 0.7
    num_predict: int = 256
    timeout_s: float = 240.0

    max_retries: int = 2
    retry_backoff_s: float = 0.5

    @classmethod
    def from_env(cls) -> "OllamaChatClient":
        return cls(
            model=_env_str("LLMGT_OLLAMA_MODEL", "mistral"),
            host=_env_str("LLMGT_OLLAMA_HOST", "http://localhost:11434"),
            temperature_default=_env_float("LLMGT_OLLAMA_TEMPERATURE", 0.7),
            num_predict=_env_int("LLMGT_OLLAMA_NUM_PREDICT", 256),
            timeout_s=_env_float("LLMGT_OLLAMA_TIMEOUT_S", 240.0),
            max_retries=_env_int("LLMGT_OLLAMA_MAX_RETRIES", 2),
            retry_backoff_s=_env_float("LLMGT_OLLAMA_RETRY_BACKOFF_S", 0.5),
        )

    def complete(self, messages: Sequence[LLMMessage], *, temperature: float | None = None) -> str:
        temp = self.temperature_default if temperature is None else float(temperature)

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {
                "temperature": temp,
                "num_predict": int(self.num_predict),
            },
        }

        url = self.host.rstrip("/") + "/api/chat"
        data = json.dumps(payload).encode("utf-8")

        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read().decode("utf-8")

                obj = json.loads(raw)
                content = self._extract_content(obj)
                content = content.strip()

                if not content:
                    raise RuntimeError(f"Ollama returned empty message.content. Raw keys={list(obj.keys())}")

                return content

            except urllib.error.HTTPError as e:
                body = ""
                try:
                    body = e.read().decode("utf-8")
                except Exception:
                    pass

                if e.code in (500, 502, 503, 504) and attempt < self.max_retries:
                    last_err = RuntimeError(f"Ollama HTTPError {e.code}: {e.reason}. Body: {body[:500]}")
                    time.sleep(self.retry_backoff_s * (2**attempt))
                    continue

                raise RuntimeError(
                    f"Ollama HTTPError {e.code}: {e.reason}. Body: {body[:500]}"
                ) from e

            except urllib.error.URLError as e:
                if attempt < self.max_retries:
                    last_err = RuntimeError(f"Cannot reach Ollama at {self.host} ({e}). Retrying...")
                    time.sleep(self.retry_backoff_s * (2**attempt))
                    continue

                raise RuntimeError(
                    f"Cannot reach Ollama at {self.host}. Is the container running and port forwarded? ({e})"
                ) from e

            except json.JSONDecodeError as e:
                raise RuntimeError("Invalid JSON from Ollama") from e

            except Exception as e:
                if attempt < self.max_retries:
                    last_err = e
                    time.sleep(self.retry_backoff_s * (2**attempt))
                    continue
                raise

        raise RuntimeError(f"Ollama request failed after retries: {last_err}")

    def _extract_content(self, obj: dict[str, Any]) -> str:
        msg = obj.get("message")
        if isinstance(msg, dict):
            c = msg.get("content")
            if isinstance(c, str):
                return c
        return ""

