"""Structured JSON logging with secret redaction and request-id correlation."""
from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_REDACTIONS = [
    (re.compile(r"(?i)\b(api[_-]?key|token|password|passwd|secret|authorization|access_token)\b(\s*[=:]\s*)(\"?)[^\s,\"'}]+"),
     r"\1\2\3***"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-~+/=]+"), "Bearer ***"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{10,}"), "sk-***"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"), "***jwt***"),
]

_EXTRA_FIELDS = ("event", "tool", "status", "duration_ms", "request_id", "error_type", "pending_id", "session_id")


def redact(text: object) -> str:
    """Remove obvious secrets (tokens, passwords, API keys, JWTs) from any string."""
    out = str(text)
    for pattern, repl in _REDACTIONS:
        out = pattern.sub(repl, out)
    return out


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        for key in _EXTRA_FIELDS:
            if hasattr(record, key):
                value = getattr(record, key)
                payload[key] = redact(value) if isinstance(value, str) else value
        payload.setdefault("request_id", request_id_var.get())
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", log_file: str = "") -> None:
    root = logging.getLogger("mlops_agent")
    root.setLevel(level.upper())
    root.handlers.clear()
    root.propagate = False
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"mlops_agent.{name}")
