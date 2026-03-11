"""Quick local check: does our *system* prompt exceed Claude cache minimums?

This script builds the same system prompts that agents use (including the optional
cached preamble controlled by env vars) and estimates token count.

We use tiktoken as a rough proxy. Exact Anthropic tokenization differs, but
this is still a good sanity check; we target a comfortable margin above 2048.

Usage:
  python scripts/check_system_prompt_tokens.py

Optional:
  LLMGT_CACHED_SYSTEM_PREAMBLE=... override the preamble text
  LLMGT_OPENROUTER_CACHE_PREAMBLE=1 to enable preamble injection
"""

from __future__ import annotations

import os

from llmgt.games.prisoners_dilemma import PrisonersDilemma
from llmgt.llm.prompt_caching import maybe_prepend_cached_preamble


def _count_tokens_approx(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Fallback: very rough heuristic
        return max(1, int(len(text) / 4))


def build_pd_system_prompt() -> str:
    game = PrisonersDilemma()
    allowed = list(game.actions())
    system = (
        "You are a game-theory agent negotiating with another agent.\n"
        f"Game: {game.name}\n"
        f"Valid actions: {allowed}\n"
        "Goal: propose or accept a plan.\n"
        "When proposing, use: PROPOSE: (X,Y)\n"
        "When accepting, use: ACCEPT: (X,Y)\n"
        "Here X is agent_a's final action and Y is agent_b's final action.\n"
    )
    return maybe_prepend_cached_preamble(system)


def main() -> None:
    print("LLMGT_OPENROUTER_CACHE_PREAMBLE=", os.getenv("LLMGT_OPENROUTER_CACHE_PREAMBLE"))
    system = build_pd_system_prompt()
    approx = _count_tokens_approx(system)

    print("System prompt chars:", len(system))
    print("Approx tokens (proxy):", approx)
    print("Meets Claude 3.5 Haiku min (2048)?", approx >= 2048)

    # Print a small head/tail for sanity without dumping everything
    head = "\n".join(system.splitlines()[:8])
    tail = "\n".join(system.splitlines()[-8:])
    print("\n--- system head ---\n" + head)
    print("\n--- system tail ---\n" + tail)


if __name__ == "__main__":
    main()

