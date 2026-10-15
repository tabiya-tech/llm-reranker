"""JSON log formatting (see README "Errors and logging")."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonLogFormatter(logging.Formatter):
    """One JSON object per record. Fields via ``extra={"fields": {...}}``.

    Never put full job/prompt text, raw LLM responses, or secrets in a message or a
    field — see README "What must not be logged".
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                entry.setdefault(key, value)
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)
