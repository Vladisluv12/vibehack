"""Credential redaction for note text. Defense in depth, not a guarantee."""
from __future__ import annotations

import os
import re
from typing import Any

_SECRET_FIELD = re.compile(
    r"(?:^|_)(?:key|api_?key|apikey|private_?key|secret|password|passwd|credential|"
    r"authorization|access_?token|refresh_?token|personal_?access_?token|token)(?:$|_)", re.I)
_REASON_FIELD = re.compile(r"^(?:reasoning|thinking|thoughts?|analysis|chain_of_thought)(?:$|_)", re.I)
_PREFIX_TOKEN = re.compile(
    r"\b(?:sk[-_][A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|"
    r"github_pat_[A-Za-z0-9_]{8,}|AIza[A-Za-z0-9_-]{12,}|"
    r"eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,})\b")
_BEARER = re.compile(r"(?i)\bbearer\s+[^\s\"'<>]+")
_ASSIGN_SECRET = re.compile(
    r"(?i)\b((?:[A-Za-z0-9_]*_)?(?:api[_-]?key|token|secret|password|"
    r"authorization))([\"']?\s*[:=]\s*[\"']?)([^\s\"',;}]+)")


def redact(value: Any) -> Any:
    """Recursively redact credentials and remove private reasoning dumps.

    Counts named input_tokens/output_tokens are metrics, not credentials.
    This is defense in depth, not a promise arbitrary text can never be secret.
    """
    secrets = sorted({v for k, v in os.environ.items()
                      if _SECRET_FIELD.search(k) and len(v) >= 6}, key=len, reverse=True)

    def visit(item: Any) -> Any:
        if isinstance(item, dict):
            result = {}
            for key, val in item.items():
                name = str(key)
                safe_key = visit(key) if isinstance(key, str) else key
                if _REASON_FIELD.search(name) and not name.endswith("tokens"):
                    continue
                if _SECRET_FIELD.search(name) and not (
                    isinstance(val, (int, float)) and name.lower().endswith("tokens")
                ):
                    result[safe_key] = "[REDACTED]"
                else:
                    result[safe_key] = visit(val)
            return result
        if isinstance(item, (list, tuple)):
            return [visit(val) for val in item]
        if isinstance(item, str):
            for secret in secrets:
                item = item.replace(secret, "[REDACTED]")
            item = _BEARER.sub("Bearer [REDACTED]", item)
            item = _PREFIX_TOKEN.sub("[REDACTED]", item)
            return _ASSIGN_SECRET.sub(r"\1\2[REDACTED]", item)
        return item

    return visit(value)

