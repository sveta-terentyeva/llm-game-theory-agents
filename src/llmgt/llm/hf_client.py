from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Any

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

    _tokenizer: Any = None
    _model: Any = None

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
            import importlib
            transformers = importlib.import_module("transformers")
            AutoModelForCausalLM = transformers.AutoModelForCausalLM
            AutoTokenizer = transformers.AutoTokenizer
        except Exception as e:
            raise RuntimeError(
                "HuggingFace backend requires 'torch', 'transformers', and usually 'accelerate'. "
                "In Colab: pip install -U torch transformers accelerate"
            ) from e

        tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=self.use_fast_tokenizer)

        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

        try:
            model.eval()
        except Exception:
            pass

        self._tokenizer = tok
        self._model = model

    def _build_inputs(self, messages: Sequence[LLMMessage]) -> tuple["torch.Tensor", "torch.Tensor"]:
        import torch  # type: ignore

        tok = self._tokenizer
        assert tok is not None

        hf_msgs = [{"role": m.role, "content": m.content} for m in messages]

        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            out = tok.apply_chat_template(
                hf_msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(out, torch.Tensor):
                input_ids = out
                attention_mask = torch.ones_like(input_ids)
                return input_ids, attention_mask

            input_ids = out["input_ids"]
            attention_mask = out.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            return input_ids, attention_mask

        text = ""
        for m in hf_msgs:
            text += f"{m['role'].upper()}: {m['content']}\n"
        text += "ASSISTANT: "

        enc = tok(text, return_tensors="pt")
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    def complete(self, messages: Sequence[LLMMessage], *, temperature: float = 0.7) -> str:
        self._ensure_loaded()
        tok = self._tokenizer
        model = self._model
        assert tok is not None
        assert model is not None

        import torch  # type: ignore

        temp = float(temperature) if temperature is not None else float(self.temperature_default)
        do_sample = bool(self.do_sample_default and temp > 0.0)

        input_ids, attention_mask = self._build_inputs(messages)

        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        with torch.inference_mode():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(self.max_new_tokens),
                do_sample=do_sample,
                temperature=max(temp, 1e-6) if do_sample else 1.0,
                top_p=float(self.top_p),
                pad_token_id=pad_token_id,
            )

        new_tokens = gen[0, input_ids.shape[-1]:]
        out = tok.decode(new_tokens, skip_special_tokens=True).strip()
        return out
