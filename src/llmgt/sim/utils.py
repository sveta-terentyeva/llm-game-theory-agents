from __future__ import annotations

import uuid


def new_episode_id(prefix: str = "ep") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"
