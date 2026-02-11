from __future__ import annotations

import re
from typing import Iterable

from llmgt.logging.records import ChatMessage

_PROPOSE_RE = re.compile(r"\bPROPOSE:\s*\(\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\)")
_COUNTER_RE = re.compile(r"\bCOUNTER:\s*\(\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\)")
_ACCEPT_RE = re.compile(r"\bACCEPT:\s*\(\s*([A-Za-z]+)\s*,\s*([A-Za-z]+)\s*\)")


def extract_last_proposal(messages: Iterable[ChatMessage]) -> tuple[str, str] | None:
    last = None
    for m in messages:
        if m.role != "agent_a":
            continue
        match = _PROPOSE_RE.search(m.content)
        if match:
            last = (match.group(1), match.group(2))
    return last


def extract_last_counter(messages: Iterable[ChatMessage]) -> tuple[str, str] | None:
    last = None
    for m in messages:
        if m.role != "agent_b":
            continue
        match = _COUNTER_RE.search(m.content)
        if match:
            last = (match.group(1), match.group(2))
    return last


def extract_accepted_pair(messages: Iterable[ChatMessage]) -> tuple[str, str] | None:
    last = None
    for m in messages:
        if m.role != "agent_b":
            continue
        match = _ACCEPT_RE.search(m.content)
        if match:
            last = (match.group(1), match.group(2))
    return last

def workflow_has_agreement(messages: Iterable[ChatMessage]) -> bool:
    return extract_accepted_pair(messages) is not None
