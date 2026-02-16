from __future__ import annotations

import re
from typing import Iterable

from llmgt.logging.records import ChatMessage

_PAIR_RE = r"\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)"
_PROPOSE_RE = re.compile(r"\bPROPOSE\s*:\s*" + _PAIR_RE)
_COUNTER_RE = re.compile(r"\bCOUNTER\s*:\s*" + _PAIR_RE)
_ACCEPT_RE = re.compile(r"\bACCEPT\s*:\s*" + _PAIR_RE)


def extract_last_proposal(messages: Iterable[ChatMessage]) -> tuple[str, str] | None:
    last = None
    for m in messages:
        match = _PROPOSE_RE.search(m.content)
        if match:
            last = (match.group(1), match.group(2))
    return last


def extract_last_counter(messages: Iterable[ChatMessage]) -> tuple[str, str] | None:
    last = None
    for m in messages:
        match = _COUNTER_RE.search(m.content)
        if match:
            last = (match.group(1), match.group(2))
    return last


def extract_accepted_pair(messages: Iterable[ChatMessage]) -> tuple[str, str] | None:
    last = None
    for m in messages:
        match = _ACCEPT_RE.search(m.content)
        if match:
            last = (match.group(1), match.group(2))
    return last

def workflow_has_agreement(messages: Iterable[ChatMessage]) -> bool:
    """
    Backward-compatible helper for older code paths.
    Returns True if there is any ACCEPT:(X,Y) in the transcript.
    """
    return extract_accepted_pair(messages) is not None