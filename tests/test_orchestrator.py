import json

from agent.llm import LLMResponse
from agent.offline_llm import OfflineClient
from agent.orchestrator import Orchestrator
from agent.tools import build_registry

from .conftest import DownMlflow, FakeAirflow, FakeApps, FakeCore, make_deployment, make_pod


class Scripted:
    """LLM stand-in that replays prepared responses."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), 0

    def complete(self, system, messages, tools):
        self.calls += 1
        return self.responses.pop(0)


def use(name, args, i="t1"):
    return LLMResponse([{"type": "tool_use", "id": i, "name": name, "input": args}], "tool_use")


def text(t):
    return LLMResponse([{"type": "text", "text": t}])


def full(settings, git_repo, mlflow_client, airflow=None, apps=None, core=None):
    from agent.config import Settings
    s = Settings(repo_root=git_repo, tool_timeout_s=10)
    apps = apps or FakeApps(make_deployment())
    reg = build_registry(s, mlflow_client=mlflow_client, airflow_client=airflow or FakeAirflow(),
                         core_api=core or FakeCore([make_pod("a"), make_pod("b")]), apps_api=apps)
    return Orchestrator(reg, OfflineClient(s), s), reg, apps


MANDATORY = ("Find the best recent model, identify the dataset version used to train it, check the deployment "
             "status, and explain whether retraining is necessary.")


def test_tool_selection_and_execution(settings, git_repo, mlflow_client):
    from agent.config import Settings
    s = Settings(repo_root=git_repo)
    reg = build_registry(s, mlflow_client=mlflow_client, core_api=FakeCore([make_pod("a")]), apps_api=FakeApps())
    orch = Orchestrator(reg, Scripted(use("k8s_get_deployed_version", {}), text("done")), s)
    r = orch.chat("what is deployed?")
    assert r.answer == "done" and [t.tool for t in r.tool_calls] == ["k8s_get_deployed_version"] and r.tool_calls[0].ok


def test_mandatory_multi_tool_scenario_uses_all_four_and_never_triggers(settings, git_repo, mlflow_client):
    airflow = FakeAirflow("success")
    orch, _, _ = full(settings, git_repo, mlflow_client, airflow)
    r = orch.chat(MANDATORY)
    tools = [t.tool for t in r.tool_calls]
    for needed in ("mlflow_find_best_run", "dvc_get_dataset_for_run", "k8s_get_deployed_version", "airflow_get_dag_runs"):
        assert needed in tools
    assert "airflow_trigger_dag" not in tools and not any(c[0] == "POST" for c in airflow.calls)
    for section in ("Known facts", "Uncertainties", "Recommended actions"):
        assert section in r.answer
    assert "No evidence that retraining is necessary" in r.answer
    assert "data-v1" in r.answer  # dataset version resolved through DVC provenance


def test_retraining_recommended_only_with_evidence_and_still_not_triggered(settings, git_repo, mlflow_client):
    airflow = FakeAirflow("failed")
    orch, _, _ = full(settings, git_repo, mlflow_client, airflow)
    r = orch.chat(MANDATORY)
    assert "Evidence supports retraining" in r.answer and "train_model" in r.answer
    assert "airflow_trigger_dag" not in [t.tool for t in r.tool_calls]
    assert not any(c[0] == "POST" for c in airflow.calls)


def test_agent_does_not_invent_results_when_tool_unavailable(settings, git_repo):
    from agent.config import Settings
    s = Settings(repo_root=git_repo, tool_timeout_s=10)
    reg = build_registry(s, mlflow_client=DownMlflow())
    r = Orchestrator(reg, OfflineClient(s), s).chat("Find the best model by f1")
    assert [t.ok for t in r.tool_calls] == [False]
    assert "cannot report anything" in r.answer and "None - no tool returned usable data" in r.answer
    assert "f1=" not in r.answer and "SUPERSECRET" not in r.answer


def test_llm_backend_failure_makes_no_claims(settings, git_repo, mlflow_client):
    class Broken:
        def complete(self, *a):
            raise RuntimeError("api key sk-ABCDEFGHIJKLMNOP invalid")
    from agent.config import Settings
    s = Settings(repo_root=git_repo)
    r = Orchestrator(build_registry(s), Broken(), s).chat("status?")
    assert "cannot make operational claims" in r.answer and "sk-ABCDEFGHIJKLMNOP" not in r.answer and r.tool_calls == []


def test_trigger_flow_needs_human_confirmation(settings, git_repo, mlflow_client):
    airflow = FakeAirflow()
    orch, reg, _ = full(settings, git_repo, mlflow_client, airflow)
    r = orch.chat("Trigger the model training pipeline and report whether execution succeeded.")
    assert r.pending_actions and "airflow_trigger_dag" in [t.tool for t in r.tool_calls]
    assert not any(c[0] == "POST" for c in airflow.calls)
    out = orch.confirm(r.pending_actions[0]["pending_id"])
    assert "Executed" in out.answer and any(c[0] == "POST" for c in airflow.calls)


def test_llm_cannot_bypass_namespace_policy(settings, git_repo, mlflow_client):
    from agent.config import Settings
    s = Settings(repo_root=git_repo)
    apps = FakeApps()
    reg = build_registry(s, core_api=FakeCore(), apps_api=apps)
    llm = Scripted(use("k8s_restart_deployment", {"namespace": "kube-system"}), text("ok"))
    r = Orchestrator(reg, llm, s).chat("restart everything in kube-system")
    assert r.tool_calls[0].error_type == "permission_denied" and apps.patches == [] and r.pending_actions == []


def test_diagnosis_scenario_unhealthy_pods(settings, git_repo, mlflow_client):
    core = FakeCore([make_pod("a", ready=False, restarts=3, waiting="CrashLoopBackOff", last_term="OOMKilled")],
                    logs="ERROR MODEL LOAD FAILED")
    orch, _, _ = full(settings, git_repo, mlflow_client, core=core, apps=FakeApps(make_deployment(ready=0)))
    r = orch.chat("Why is my prediction API unavailable?")
    assert "k8s_diagnose_deployment" in [t.tool for t in r.tool_calls]
    assert "CrashLoopBackOff" in r.answer and "OOMKilled" in r.answer


def test_tool_results_serialised_without_secrets(settings, git_repo, mlflow_client):
    orch, _, _ = full(settings, git_repo, mlflow_client)
    from agent.schemas import ToolResult
    s = orch._serialize(ToolResult(ok=False, tool="x", error="boom", error_type="internal_error"))
    assert json.loads(s)["error"] == "boom"


def test_compare_prompt_recommends_candidate_with_caveats(settings, git_repo, mlflow_client):
    orch, _, _ = full(settings, git_repo, mlflow_client)
    r = orch.chat("Compare the last five training runs and recommend the best candidate based on F1 score.")
    tools = [t.tool for t in r.tool_calls]
    assert "mlflow_compare_runs" in tools and "mlflow_find_best_run" in tools
    assert "run-2" in r.answer and "Uncertainties" in r.answer


def test_reproduce_prompt_returns_concrete_steps(settings, git_repo, mlflow_client):
    orch, _, _ = full(settings, git_repo, mlflow_client)
    r = orch.chat("Which dataset version was used for the best model by f1? How can I reproduce it?")
    assert "dvc_explain_reproduce" in [t.tool for t in r.tool_calls]
    assert "dvc checkout" in r.answer and "data-v1" in r.answer


def test_restart_in_foreign_namespace_via_chat_is_refused(settings, git_repo, mlflow_client):
    orch, _, apps = full(settings, git_repo, mlflow_client)
    r = orch.chat("Restart the ml-inference deployment in the kube-system namespace")
    restart = [t for t in r.tool_calls if t.tool == "k8s_restart_deployment"][0]
    assert restart.error_type == "permission_denied" and apps.patches == [] and r.pending_actions == []
