from fastapi.testclient import TestClient

from agent.config import Settings
from agent.llm import LLMResponse
from agent.main import create_app
from agent.orchestrator import Orchestrator
from agent.tools import build_registry


class Echo:
    def complete(self, system, messages, tools):
        return LLMResponse([{"type": "text", "text": "hello"}])


def client(token=""):
    s = Settings(agent_api_token=token)
    return TestClient(create_app(Orchestrator(build_registry(s), Echo(), s), s))


def test_health_and_tools():
    c = client()
    assert c.get("/health").json()["status"] == "ok"
    names = {t["name"] for t in c.get("/tools").json()}
    assert {"dvc_get_dataset_version", "mlflow_find_best_run", "airflow_trigger_dag", "k8s_diagnose_deployment"} <= names
    assert {t["module"] for t in c.get("/tools").json()} == {"dvc", "mlflow", "airflow", "kubernetes"}


def test_chat_returns_answer_and_request_id():
    r = client().post("/chat", json={"message": "hi"})
    assert r.status_code == 200 and r.json()["answer"] == "hello" and r.headers["x-request-id"]


def test_input_validation():
    assert client().post("/chat", json={"message": ""}).status_code == 422
    assert client().post("/chat", json={"message": "x", "evil": 1}).status_code == 422


def test_bearer_auth_enforced_when_configured():
    c = client("s3cret")
    assert c.post("/chat", json={"message": "hi"}).status_code == 401
    assert c.post("/chat", json={"message": "hi"}, headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_confirm_unknown_id():
    r = client().post("/confirm", json={"pending_id": "unknown-id"})
    assert r.status_code == 200 and "Could not execute" in r.json()["answer"]
