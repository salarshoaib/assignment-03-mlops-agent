import requests

from agent.tools import build_registry
from agent.tools.airflow_tools import AirflowClient
from agent.tools.base import ServiceUnavailable

from .conftest import FakeAirflow


def reg(settings, fake):
    return build_registry(settings, airflow_client=fake)


def test_list_dags_marks_approved(settings):
    d = reg(settings, FakeAirflow()).execute("airflow_list_dags", {}).data
    assert {x["dag_id"]: x["approved_for_agent"] for x in d["dags"]} == {"training_pipeline": True, "other_dag": False}


def test_trigger_unapproved_dag_rejected_without_confirmation(settings):
    fake = FakeAirflow()
    res = reg(settings, fake).execute("airflow_trigger_dag", {"dag_id": "other_dag"}, confirmed=True)
    assert res.error_type == "permission_denied" and not any(c[0] == "POST" for c in fake.calls)


def test_trigger_unapproved_conf_key_rejected(settings):
    res = reg(settings, FakeAirflow()).execute("airflow_trigger_dag", {"dag_id": "training_pipeline", "conf": {"cmd": "rm -rf /"}})
    assert res.error_type == "permission_denied"


def test_trigger_requires_confirmation_then_posts(settings):
    fake = FakeAirflow()
    r = reg(settings, fake)
    first = r.execute("airflow_trigger_dag", {"dag_id": "training_pipeline", "conf": {"inject_failure": "quality"}})
    assert first.error_type == "confirmation_required" and not any(c[0] == "POST" for c in fake.calls)
    done = r.confirm(first.data["pending_id"])
    assert done.ok and done.data["triggered"] is True
    post = [c for c in fake.calls if c[0] == "POST"][0]
    assert post[2]["conf"] == {"inject_failure": "quality"}


def test_failed_tasks_identified(settings):
    d = reg(settings, FakeAirflow("failed")).execute("airflow_get_failed_tasks", {"dag_id": "training_pipeline"}).data
    assert [t["task_id"] for t in d["failed_tasks"]] == ["train_model"]
    assert [t["task_id"] for t in d["upstream_failed_tasks"]] == ["evaluate_and_log", "quality_gate"]
    assert d["run_state"] == "failed"


def test_task_logs_redacted_with_error_lines(settings):
    d = reg(settings, FakeAirflow("failed")).execute("airflow_get_task_logs", {
        "dag_id": "training_pipeline", "dag_run_id": "manual__1", "task_id": "quality_gate"})
    assert d.ok and "abc123SECRET" not in str(d.data) and d.data["error_lines"]


def test_dag_runs_status(settings):
    d = reg(settings, FakeAirflow("success")).execute("airflow_get_dag_runs", {"dag_id": "training_pipeline"}).data
    assert d["runs"][0]["state"] == "success"


def test_airflow_down_is_service_unavailable(settings):
    res = reg(settings, FakeAirflow(fail=ServiceUnavailable("Airflow unreachable"))).execute("airflow_get_dag_runs", {"dag_id": "training_pipeline"})
    assert res.error_type == "service_unavailable"


class Resp:
    def __init__(self, code, body):
        self.status_code, self._b, self.content, self.text = code, body, b"x", str(body)

    def json(self):
        return self._b


class FakeSession:
    def __init__(self):
        self.auth_calls, self.seen = 0, []

    def post(self, url, json=None, timeout=None):
        self.auth_calls += 1
        return Resp(201, {"access_token": f"tok{self.auth_calls}"})

    def request(self, method, url, headers=None, **kw):
        self.seen.append(headers["Authorization"])
        if len(self.seen) == 1:
            return Resp(401, {})
        return Resp(200, {"dags": []})


def test_client_authenticates_and_refreshes_token_on_401(settings):
    s = FakeSession()
    out = AirflowClient(settings, s).request("GET", "/dags")
    assert out == {"dags": []} and s.auth_calls == 2 and s.seen == ["Bearer tok1", "Bearer tok2"]


def test_client_connection_error(settings):
    class Dead:
        def post(self, *a, **k):
            raise requests.ConnectionError("refused")
        get = post
    try:
        AirflowClient(settings, Dead()).request("GET", "/dags")
    except ServiceUnavailable as exc:
        assert "unreachable" in str(exc)
    else:
        raise AssertionError("expected ServiceUnavailable")
