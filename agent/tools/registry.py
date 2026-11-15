"""Tool registry: validation, authorization, confirmation, timeouts, structured logging."""
from __future__ import annotations

import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from agent.config import Settings
from agent.logging_utils import get_logger, redact, request_id_var
from agent.schemas import ToolResult
from agent.tools.base import Tool, ToolError

log = get_logger("tools")


@dataclass
class PendingAction:
    pending_id: str
    tool: str
    args: dict[str, Any]
    parsed: BaseModel
    created: float = field(default_factory=time.time)
    request_id: str = ""


class ToolRegistry:
    def __init__(self, settings: Settings, tools: list[Tool] | None = None):
        self.settings = settings
        self._tools: dict[str, Tool] = {}
        self._pending: dict[str, PendingAction] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")
        for tool in tools or []:
            self.register(tool)

    # ----------------------------------------------------------------- registration
    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "module": t.module, "mutating": t.mutating, "description": t.description}
            for t in self._tools.values()
        ]

    # ----------------------------------------------------------------- execution
    def execute(self, name: str, args: dict[str, Any] | None, request_id: str | None = None,
                confirmed: bool = False) -> ToolResult:
        request_id = request_id or request_id_var.get()
        start = time.perf_counter()

        def finish(ok: bool, data: dict | None = None, error: str | None = None,
                   error_type: str | None = None) -> ToolResult:
            duration = round((time.perf_counter() - start) * 1000, 2)
            status = "ok" if ok else (error_type or "error")
            log.info(
                "tool_call",
                extra={"event": "tool_call", "tool": name, "status": status,
                       "duration_ms": duration, "request_id": request_id, "error_type": error_type},
            )
            return ToolResult(ok=ok, tool=name, data=data, error=redact(error) if error else None,
                              error_type=error_type, duration_ms=duration, request_id=request_id)

        tool = self._tools.get(name)
        if tool is None:
            return finish(False, error=f"Unknown tool '{name}'", error_type="unknown_tool")

        try:
            parsed = tool.input_model.model_validate(args or {})
        except ValidationError as exc:
            return finish(False, error=f"Invalid arguments: {str(exc)[:400]}", error_type="validation_error")

        try:
            if tool.precheck:
                tool.precheck(parsed)
        except ToolError as exc:
            return finish(False, error=str(exc), error_type=exc.error_type)

        if tool.mutating and not confirmed:
            pending = self._create_pending(tool, parsed, args or {}, request_id)
            return finish(
                False,
                data={"pending_id": pending.pending_id, "tool": tool.name, "arguments": parsed.model_dump(),
                      "expires_in_s": self.settings.confirmation_ttl_s},
                error="This operation changes system state and requires explicit user confirmation.",
                error_type="confirmation_required",
            )

        timeout = tool.timeout_s or self.settings.tool_timeout_s
        future = self._executor.submit(tool.func, parsed)
        try:
            data = future.result(timeout=timeout)
        except FutureTimeout:
            future.cancel()
            return finish(False, error=f"Tool '{name}' timed out after {timeout}s", error_type="timeout")
        except ToolError as exc:
            return finish(False, error=str(exc), error_type=exc.error_type)
        except Exception as exc:  # noqa: BLE001 - last-resort boundary, message is redacted
            log.exception("tool_unhandled_error", extra={"tool": name, "request_id": request_id})
            return finish(False, error=f"{type(exc).__name__}: {exc}", error_type="internal_error")
        return finish(True, data=data)

    # ----------------------------------------------------------------- confirmation flow
    def _create_pending(self, tool: Tool, parsed: BaseModel, args: dict, request_id: str) -> PendingAction:
        pending = PendingAction(secrets.token_urlsafe(6), tool.name, args, parsed, request_id=request_id)
        with self._lock:
            self._purge()
            self._pending[pending.pending_id] = pending
        return pending

    def _purge(self) -> None:
        cutoff = time.time() - self.settings.confirmation_ttl_s
        for key in [k for k, v in self._pending.items() if v.created < cutoff]:
            del self._pending[key]

    def pending_actions(self) -> list[dict[str, Any]]:
        with self._lock:
            self._purge()
            return [{"pending_id": p.pending_id, "tool": p.tool, "arguments": p.parsed.model_dump()}
                    for p in self._pending.values()]

    def confirm(self, pending_id: str, request_id: str | None = None) -> ToolResult:
        """Execute a previously proposed mutating action. Called only from a human-initiated path."""
        request_id = request_id or request_id_var.get()
        with self._lock:
            self._purge()
            pending = self._pending.pop(pending_id, None)
        if pending is None:
            return ToolResult(ok=False, tool="confirm", error="Unknown or expired confirmation id.",
                              error_type="validation_error", request_id=request_id)
        log.info("confirmation_accepted", extra={"event": "confirmation", "tool": pending.tool,
                                                 "pending_id": pending_id, "request_id": request_id})
        return self.execute(pending.tool, pending.args, request_id=request_id, confirmed=True)

    def reject(self, pending_id: str) -> bool:
        with self._lock:
            return self._pending.pop(pending_id, None) is not None
