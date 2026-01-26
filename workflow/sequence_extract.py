from __future__ import annotations

import json
import re
from typing import Any


_JSON_FENCE_RE = re.compile(r"```json\s*(?P<body>.*?)\s*```", re.IGNORECASE | re.DOTALL)


def extract_repro_sequence_from_finish_message(
    finish_message: str,
    *,
    token: str = "<LEAK_SEQUENCE_READY>",
) -> dict[str, Any] | None:
    """
    Extract a `repro_sequence.json` payload from a model finish(message=...) text.

    Workflow controller intended usage:
    - Check for the token to know the agent claims the sequence is ready.
    - Parse the fenced JSON block for a machine-executable sequence.
    """
    if token not in (finish_message or ""):
        return None

    m = _JSON_FENCE_RE.search(finish_message)
    if not m:
        return None

    body = m.group("body").strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None

