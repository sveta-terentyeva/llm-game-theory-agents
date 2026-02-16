from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # noqa: F401
    import importlib
    transformers = importlib.import_module("transformers")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    AutoTokenizer = transformers.AutoTokenizer

from llmgt.llm.client import LLMMessage


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if not v else v


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if not v else int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if not v else float(v)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if not v:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass
class HuggingFaceChatClient:

    model_id: str
    max_new_tokens: int = 128
    temperature_default: float = 0.7
    top_p: float = 0.95
    do_sample_default: bool = True

    use_fast_tokenizer: bool = True

    _tokenizer: Optional[AutoTokenizer] = None
    _model: Optional[AutoModelForCausalLM] = None

    @classmethod
    def from_env(cls) -> "HuggingFaceChatClient":
        return cls(
            model_id=_env_str("LLMGT_HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
            max_new_tokens=_env_int("LLMGT_HF_MAX_NEW_TOKENS", 128),
            temperature_default=_env_float("LLMGT_HF_TEMPERATURE", 0.7),
            top_p=_env_float("LLMGT_HF_TOP_P", 0.95),
            do_sample_default=_env_bool("LLMGT_HF_DO_SAMPLE", True),
            use_fast_tokenizer=_env_bool("LLMGT_HF_USE_FAST_TOKENIZER", True),
        )

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        try:
            import torch  # noqa: F401
            import importlib
            transformers = importlib.import_module("transformers")
            AutoModelForCausalLM = transformers.AutoModelForCausalLM
            AutoTokenizer = transformers.AutoTokenizer

        except Exception as e:
            raise RuntimeError(
                "HuggingFace backend requires 'torch', 'transformers', and 'accelerate'. "
                "Install in Colab (recommended) or locally: pip install -U torch transformers accelerate"
            ) from e

        tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=self.use_fast_tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype="auto",
        )

        self._tokenizer = tok
        self._model = model

    def complete(self, messages: Sequence[LLMMessage], *, temperature: float = 0.7) -> str:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        temp = float(temperature) if temperature is not None else float(self.temperature_default)
        do_sample = self.do_sample_default and temp > 0.0

        # Convert to HF chat format
        hf_msgs = [{"role": m.role, "content": m.content} for m in messages]

        tok = self._tokenizer

        # Prefer chat template if available (most instruct models provide it)
        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            prompt_ids = tok.apply_chat_template(
                hf_msgs,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            text = ""
            for m in hf_msgs:
                text += f"{m['role'].upper()}: {m['content']}\n"
            text += "ASSISTANT: "
            prompt_ids = tok(text, return_tensors="pt").input_ids

        prompt_ids = prompt_ids.to(self._model.device)

        gen = self._model.generate(
            prompt_ids,
            max_new_tokens=int(self.max_new_tokens),
            do_sample=bool(do_sample),
            temperature=max(temp, 1e-6) if do_sample else 1.0,
            top_p=float(self.top_p),
            pad_token_id=tok.eos_token_id,
        )

        # Decode only the newly generated part
        new_tokens = gen[0, prompt_ids.shape[-1]:]
        out = tok.decode(new_tokens, skip_special_tokens=True).strip()

        return out
