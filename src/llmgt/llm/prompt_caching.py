"""Helpers for OpenRouter prompt caching (Anthropic/Claude).

Goal
----
Create a large, stable *system* preamble that can be cached across many episodes.

Why this exists
---------------
Anthropic prompt caching has minimum cacheable prompt lengths (e.g. 2048 tokens
for Claude 3.5 Haiku). Many of our system prompts are shorter, which prevents
cache hits even if cache_control is attached.

We solve this by optionally prepending a reusable, content-neutral preamble to
system prompts when prompt caching is enabled.

Notes
-----
- This module does *not* add cache_control itself. That is handled by
  ``OpenRouterClient._serialize_messages``.
- Keep the preamble stable across runs to maximize cache hits.
"""

from __future__ import annotations

import os


def _count_tokens_approx(text: str) -> int:
    """Approximate token count.

    We prefer tiktoken when installed (dev dependency) but fall back to a stable
    heuristic so this module stays lightweight.

    This is only used to size the cached preamble;
    Anthropic/OpenRouter tokenization may differ slightly.
    """

    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Rough heuristic: ~4 chars/token for English-like text.
        return max(1, int(len(text) / 4))


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _build_default_cached_system_preamble() -> str:
    # Intentionally long (>= Claude cache minimum) but should not be excessively large.
    # Default target: ~2200 tokens (just above the 2048 minimum for Claude 3.5 Haiku).
    target_tokens = _env_int("LLMGT_CACHED_SYSTEM_PREAMBLE_TARGET_TOKENS", 2200)
    min_tokens = _env_int("LLMGT_CACHED_SYSTEM_PREAMBLE_MIN_TOKENS", 2048)
    # Ensure target is never below our minimum safety threshold.
    target_tokens = max(target_tokens, min_tokens)

    parts: list[str] = []

    parts.append(
        "You are an LLM agent participating in repeated game-theory experiments.\n"
        "You must follow instructions precisely and be consistent across turns.\n"
        "If the user requests a strict output format (e.g., a single action token or a single protocol line), "
        "you MUST comply and output nothing else.\n"
    )

    parts.append(
        "General rules (applies to all games):\n"
        "1) Never include markdown, code fences, or extra commentary unless explicitly requested.\n"
        "2) Never invent new actions beyond the allowed list.\n"
        "3) If uncertain, choose the safest valid output that matches the requested format.\n"
        "4) Do not repeat the prompt. Do not echo the conversation history.\n"
        "5) Be deterministic in formatting: no trailing spaces; no extra lines.\n"
    )

    parts.append(
        "Terminology and concepts (reference):\n"
        "- payoff: numeric utility assigned to an outcome.\n"
        "- strategy: a mapping from histories to actions.\n"
        "- best response: action that maximizes your payoff given the other agent's action.\n"
        "- Nash equilibrium: a profile where no agent can improve unilaterally.\n"
        "- Pareto improvement: makes at least one agent better off without making the other worse off.\n"
        "- fairness: can be interpreted as symmetry, equality, or minimizing regret depending on context.\n"
        "- commitment: credible plan that constrains future actions.\n"
        "- mixed strategy: probability distribution over actions.\n"
    )

    parts.append(
        "Protocol patterns you may be asked to follow (examples):\n"
        "A) Single-token action output:\n"
        "   - Output exactly one token from the valid action list, e.g. COOPERATE or DEFECT.\n"
        "B) Negotiation protocol output:\n"
        "   - agent_a: PROPOSE: (X,Y)\n"
        "   - agent_b: COUNTER: (X,Y) or ACCEPT: (X,Y)\n"
        "   Where X is agent_a's final action and Y is agent_b's final action.\n"
        "   Output exactly one line. No other words.\n"
    )

    parts.append(
        "Common formatting errors to avoid:\n"
        "- Adding explanations when only an action is requested.\n"
        "- Outputting multiple lines when only one line is requested.\n"
        "- Using parentheses/spaces incorrectly (must match the requested template).\n"
        "- Including JSON unless JSON is explicitly required.\n"
    )

    appendix_paragraph = (
        "Appendix (stable reference text for caching): In repeated strategic settings, agents may face tradeoffs "
        "between short-term gains and long-term cooperation. When interactions repeat, conditioning behavior on "
        "past actions can sustain cooperation if future payoffs are sufficiently valued. In one-shot games, a "
        "dominant strategy may exist; in coordination games, equilibrium selection can depend on focal points. "
        "In bargaining, proposals can be evaluated using fairness heuristics, outside options, and acceptance "
        "thresholds. In all cases, you must obey the allowed actions and the exact output format demanded by the "
        "prompt. This appendix is only a reference and should not be quoted unless asked.\n"
    )

    # Add as many stable paragraphs as needed to reach the target token count.
    # We cap at a reasonable number to avoid runaway loops if token counting is unavailable.
    base = "\n".join(parts).strip() + "\n"
    approx = _count_tokens_approx(base)
    max_repeats = 256
    repeats = 0
    while approx < target_tokens and repeats < max_repeats:
        parts.append(appendix_paragraph)
        repeats += 1
        # Recompute on the full text every few iterations; cheap at this size.
        if repeats % 4 == 0 or approx < min_tokens:
            approx = _count_tokens_approx("\n".join(parts).strip() + "\n")

    return "\n".join(parts).strip() + "\n"


def cached_system_preamble() -> str:
    """Return the cached system preamble.

    Can be overridden by setting ``LLMGT_CACHED_SYSTEM_PREAMBLE`` to a custom
    string (useful for thesis prompts).
    """

    override = os.getenv("LLMGT_CACHED_SYSTEM_PREAMBLE")
    if override:
        return override.strip() + "\n"
    return _build_default_cached_system_preamble()


def maybe_prepend_cached_preamble(system_prompt: str) -> str:
    """Prepend the cached preamble to *system_prompt* if enabled.

    Enable by setting ``LLMGT_OPENROUTER_CACHE_PREAMBLE=1``.

    This keeps default behavior unchanged unless explicitly enabled.
    """

    if os.getenv("LLMGT_OPENROUTER_CACHE_PREAMBLE") != "1":
        return system_prompt
    pre = cached_system_preamble()
    # Avoid double-prepending if user already concatenated it.
    if system_prompt.strip().startswith(pre.splitlines()[0]):
        return system_prompt
    return pre + "\n" + system_prompt

