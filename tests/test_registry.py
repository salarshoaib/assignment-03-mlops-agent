import io
import json
import logging
import time

from agent.logging_utils import JsonFormatter, redact
from agent.schemas import StrictModel
from agent.tools.base import PermissionDenied, ServiceUnavailable, Tool
from agent.tools.registry import ToolRegistry


class In(StrictModel):
    x: int = 1


def make(settings, func, **kw):
    return ToolRegistry(settings, [Tool("t", "test", "test", In, func, **kw)])


def test_successful_execution_and_structured_log(settings):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    lg = logging.getLogger("mlops_agent.tools")
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    try:
        res = make(settings, lambda a: {"double": a.x * 2}).execute("t", {"x": 4}, request_id="req-123")
    finally:
        lg.removeHandler(handler)
    assert res.ok and res.data == {"double": 8} and res.request_id == "req-123"
    rec = json.loads(stream.getvalue().strip().splitlines()[-1])
    assert rec["tool"] == "t" and rec["status"] == "ok" and rec["request_id"] == "req-123"
    assert isinstance(rec["duration_ms"], float)


def test_unknown_tool(settings):
    r = make(settings, lambda a: {}).execute("nope", {})
    assert not r.ok and r.error_type == "unknown_tool"


def test_invalid_arguments_rejected(settings):
    r = make(settings, lambda a: {}).execute("t", {"x": "not-an-int"})
    assert r.error_type == "validation_error"
    r = make(settings, lambda a: {}).execute("t", {"x": 1, "extra": 2})
    assert r.error_type == "validation_error"


def test_tool_timeout(settings):
    def slow(a):
        time.sleep(0.5)
        return {}
    r = make(settings, slow, timeout_s=0.05).execute("t", {})
    assert not r.ok and r.error_type == "timeout"


def test_service_failure_is_structured(settings):
    def boom(a):
        raise ServiceUnavailable("airflow down")
    r = make(settings, boom).execute("t", {})
    assert r.error_type == "service_unavailable" and "airflow down" in r.error


def test_unexpected_exception_is_contained_and_redacted(settings):
    def boom(a):
        raise RuntimeError("login failed password=hunter2 token=abcdef")
    r = make(settings, boom).execute("t", {})
    assert r.error_type == "internal_error"
    assert "hunter2" not in r.error and "abcdef" not in r.error


def test_mutating_tool_requires_confirmation(settings):
    calls = []
    reg = make(settings, lambda a: calls.append(a) or {"done": True}, mutating=True)
    r = reg.execute("t", {"x": 2})
    assert r.error_type == "confirmation_required" and not calls
    pid = r.data["pending_id"]
    done = reg.confirm(pid)
    assert done.ok and len(calls) == 1
    assert reg.confirm(pid).error_type == "validation_error"  # single use


def test_precheck_runs_before_confirmation(settings):
    def deny(a):
        raise PermissionDenied("nope")
    reg = make(settings, lambda a: {}, mutating=True, precheck=deny)
    r = reg.execute("t", {})
    assert r.error_type == "permission_denied" and not reg.pending_actions()


def test_pending_actions_expire(settings):
    reg = make(settings, lambda a: {}, mutating=True)
    pid = reg.execute("t", {}).data["pending_id"]
    reg._pending[pid].created -= 10_000
    assert reg.confirm(pid).error_type == "validation_error"


def test_redact_and_formatter():
    assert "s3cr3t" not in redact("password=s3cr3t and Authorization: Bearer abc.def.ghi")
    assert "sk-" not in redact("key sk-ABCDEFGHIJKLMNOPQRS").replace("sk-***", "")
