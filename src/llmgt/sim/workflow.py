"""Workflow protocol extraction (PROPOSE / COUNTER / ACCEPT).

These helpers scan a conversation for structured negotiation markers and
return the extracted action pairs.  Used by both the simulation runner
and the metrics module.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from llmgt.logging.records import ChatMessage

_PAIR_RE = r"\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)"

_PROPOSE_RE = re.compile(r"(?i)\bpropose\b\s*:?\s*" + _PAIR_RE)
_COUNTER_RE = re.compile(r"(?i)\b(counter|counter[-\s]?propose|counterproposal)\b\s*:?\s*" + _PAIR_RE)
_ACCEPT_RE = re.compile(r"(?i)\b(accept|accepted|agree|agreed)\b\s*:?\s*" + _PAIR_RE)

_LEGACY_PROPOSE_RE = re.compile(r"\bPROPOSE\s*:\s*" + _PAIR_RE)
_LEGACY_COUNTER_RE = re.compile(r"\bCOUNTER\s*:\s*" + _PAIR_RE)
_LEGACY_ACCEPT_RE = re.compile(r"\bACCEPT\s*:\s*" + _PAIR_RE)


def _iter_text(messages: Iterable[ChatMessage]) -> Iterable[str]:
    for m in messages:
        if m and isinstance(m.content, str):
            yield m.content


def extract_last_proposal(messages: Iterable[ChatMessage]) -> Optional[tuple[str, str]]:
    last: Optional[tuple[str, str]] = None
    for txt in _iter_text(messages):
        m = _PROPOSE_RE.search(txt) or _LEGACY_PROPOSE_RE.search(txt)
        if m:
            last = (m.group(1), m.group(2))
    return last


def extract_last_counter(messages: Iterable[ChatMessage]) -> Optional[tuple[str, str]]:
    last: Optional[tuple[str, str]] = None
    for txt in _iter_text(messages):
        m = _COUNTER_RE.search(txt) or _LEGACY_COUNTER_RE.search(txt)
        if m:
            if m.re is _COUNTER_RE:
                last = (m.group(2), m.group(3))
            else:
                last = (m.group(1), m.group(2))
    return last


def extract_accepted_pair(messages: Iterable[ChatMessage]) -> Optional[tuple[str, str]]:
    last: Optional[tuple[str, str]] = None
    for txt in _iter_text(messages):
        m = _ACCEPT_RE.search(txt) or _LEGACY_ACCEPT_RE.search(txt)
        if m:
            if m.re is _ACCEPT_RE:
                last = (m.group(2), m.group(3))
            else:
                last = (m.group(1), m.group(2))
    return last


def workflow_has_agreement(messages: Iterable[ChatMessage]) -> bool:
    return extract_accepted_pair(messages) is not None
