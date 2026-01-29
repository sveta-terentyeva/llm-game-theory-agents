from __future__ import annotations

import re
from typing import Iterable, Optional

from llmgt.logging.records import ChatMessage

_PAIR_RE = re.compile(r"\(([A-Za-z]+)\s*,\s*([A-Za-z]+)\)")
_ACCEPT_RE = re.compile(r"\bACCEPT\b\s*:\s*" + _PAIR_RE.pattern)
_PROPOSE_RE = re.compile(r"\bPROPOSE\b\s*:\s*" + _PAIR_RE.pattern)


def extract_last_proposal(messages: Iterable[ChatMessage]) -> Optional[tuple[str, str]]:
    last = None
    for m in messages:
        mm = _PROPOSE_RE.search(m.content)
        if mm:
            last = (mm.group(1), mm.group(2))
    return last


def extract_accepted_pair(messages: Iterable[ChatMessage]) -> Optional[tuple[str, str]]:
    for m in messages:
        mm = _ACCEPT_RE.search(m.content)
        if mm:
            return (mm.group(1), mm.group(2))
    return None
