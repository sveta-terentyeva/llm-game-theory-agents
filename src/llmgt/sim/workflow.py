from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

from llmgt.logging.records import ChatMessage

_TOKEN_RE = r"([A-Za-z0-9_]+)"
_PAIR_RE = rf"\(\s*{_TOKEN_RE}\s*,\s*{_TOKEN_RE}\s*\)"

_PROPOSE_RE = re.compile(rf"\bPROPOSE\s*:\s*{_PAIR_RE}\b", flags=re.IGNORECASE)
_COUNTER_RE = re.compile(rf"\bCOUNTER\s*:\s*{_PAIR_RE}\b", flags=re.IGNORECASE)
_ACCEPT_RE = re.compile(rf"\bACCEPT\s*:\s*{_PAIR_RE}\b", flags=re.IGNORECASE)


def _extract_last_pair(messages: Iterable[ChatMessage], pattern: re.Pattern[str]) -> Optional[Tuple[str, str]]:
    last: Optional[Tuple[str, str]] = None
    for m in messages:
        match = pattern.search(m.content or "")
        if match:
            last = (match.group(1), match.group(2))
    return last


def _extract_first_pair(messages: Iterable[ChatMessage], pattern: re.Pattern[str]) -> Optional[Tuple[str, str]]:
    for m in messages:
        match = pattern.search(m.content or "")
        if match:
            return (match.group(1), match.group(2))
    return None


def extract_last_proposal(messages: Iterable[ChatMessage]) -> Optional[Tuple[str, str]]:
    return _extract_last_pair(messages, _PROPOSE_RE)


def extract_last_counter(messages: Iterable[ChatMessage]) -> Optional[Tuple[str, str]]:
    return _extract_last_pair(messages, _COUNTER_RE)


def extract_accepted_pair(messages: Iterable[ChatMessage]) -> Optional[Tuple[str, str]]:
    return _extract_last_pair(messages, _ACCEPT_RE)


def extract_first_accepted_pair(messages: Iterable[ChatMessage]) -> Optional[Tuple[str, str]]:
    return _extract_first_pair(messages, _ACCEPT_RE)


def workflow_has_agreement(messages: Iterable[ChatMessage]) -> bool:
    return extract_first_accepted_pair(messages) is not None