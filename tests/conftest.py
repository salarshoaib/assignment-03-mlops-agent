"""Shared fixtures: temp git+DVC repo, real (sqlite) MLflow, and fakes for Airflow and Kubernetes."""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agent.config import Settings
from agent.tools.base import ServiceUnavailable

MD5_V1, MD5_V2 = "a" * 32, "b" * 32


def git(cwd, *args):
    subprocess.run(["git", "-c", "user.email=t@t.io", "-c", "user.name=tester", *args], cwd=cwd, check=True,
                   capture_output=True)


def write_dvc(repo, md5, size):
    f = repo / "data" / "raw" / "dataset.csv.dvc"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"outs:\n- md5: {md5}\n  size: {size}\n  hash: md5\n  path: dataset.csv\n")


@pytest.fixture
def settings(tmp_path):
    return Settings(repo_root=tmp_path, airflow_password="pw", mlflow_tracking_uri="sqlite:///unused",
                    tool_timeout_s=5.0)


@pytest.fixture
def git_repo(tmp_path):
    """Repo with dataset versions data-v1 (md5 a..) and data-v2 (md5 b..)."""
    git(tmp_path, "init", "-b", "main")
    write_dvc(tmp_path, MD5_V1, 1000)
    for cmd in (("add", "."), ("commit", "-m", "data v1"), ("tag", "data-v1")):
        git(tmp_path, *cmd)
    write_dvc(tmp_path, MD5_V2, 1500)
    for cmd in (("add", "."), ("commit", "-m", "data v2"), ("tag", "data-v2")):
        git(tmp_path, *cmd)
    return tmp_path


@pytest.fixture
def mlflow_client(tmp_path_factory):
    from mlflow.tracking import MlflowClient

    base = tmp_path_factory.mktemp("mlflow")
    client = MlflowClient(tracking_uri=f"sqlite:///{base}/m.db")
    eid = client.create_experiment("mlops-agent", artifact_location=(base / "art").as_uri())
    spec = [("run-1", 0.84, 0.85, "FINISHED", MD5_V1, 100), ("run-2", 0.91, 0.92, "FINISHED", MD5_V1, 200),
            ("run-3", 0.97, 0.97, "FAILED", MD5_V1, 300), ("run-4", 0.905, 0.91, "FINISHED", MD5_V1, 300)]
    for name, f1, acc, status, md5, n in spec:
        run = client.create_run(eid, tags={"mlflow.runName": name, "dvc.md5": md5,
                                           "dvc.dataset_path": "data/raw/dataset.csv", "pipeline": "manual"})
        client.log_metric(run.info.run_id, "f1", f1)
        client.log_metric(run.info.run_id, "accuracy", acc)
        client.log_param(run.info.run_id, "n_estimators", n)
        client.log_param(run.info.run_id, "max_depth", 6)
        client.set_terminated(run.info.run_id, status)
    client.eid = eid
    return client


# ---------------------------------------------------------------- Airflow fake
class FakeAirflow:
    def __init__(self, run_state="success", fail=None):
        self.calls, self.run_state, self.fail = [], run_state, fail

    def request(self, method, path, params=None, json=None, accept="application/json", timeout=15):
        self.calls.append((method, path, json))
        if self.fail:
            raise self.fail
        if method == "POST" and path.endswith("/dagRuns"):
            return {"dag_run_id": "manual__new", "state": "queued", "conf": (json or {}).get("conf")}
        if path == "/dags":
            return {"total_entries": 2, "dags": [{"dag_id": "training_pipeline", "is_paused": False, "tags": [{"name": "mlops"}]},
                                                  {"dag_id": "other_dag", "is_paused": False, "tags": []}]}
        if path.endswith("/taskInstances"):
            states = (["success", "success", "failed", "upstream_failed", "upstream_failed"] if self.run_state == "failed"
                      else ["success"] * 5)
            ids = ["validate_dataset", "preprocess_data", "train_model", "evaluate_and_log", "quality_gate"]
            return {"task_instances": [{"task_id": i, "state": s, "try_number": 3 if s == "failed" else 1,
                                        "max_tries": 2} for i, s in zip(ids, states, strict=True)]}
        if "/logs/" in path:
            return {"content": ["INFO start", "ERROR quality gate FAILED: f1 0.88 < 0.999 token=abc123SECRET"]}
        if "/dagRuns/" in path:
            return {"dag_run_id": path.rsplit("/", 1)[1], "state": self.run_state}
        if path.endswith("/dagRuns"):
            return {"dag_runs": [{"dag_run_id": "manual__1", "state": self.run_state,
                                  "start_date": "2026-11-09T10:00:00Z"}]}
        raise AssertionError(f"unexpected path {path}")


# ---------------------------------------------------------------- Kubernetes fakes
def make_pod(name, phase="Running", ready=True, restarts=0, waiting=None, last_term=None, image="mlops-inference:1.0.0"):
    cs = SimpleNamespace(
        name="inference", ready=ready, restart_count=restarts, image=image,
        state=SimpleNamespace(waiting=SimpleNamespace(reason=waiting, message=None) if waiting else None, terminated=None),
        last_state=SimpleNamespace(terminated=SimpleNamespace(reason=last_term, exit_code=137) if last_term else None))
    probe = SimpleNamespace(http_get=SimpleNamespace(path="/ready", port="http"))
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, creation_timestamp=datetime(2026, 11, 9, tzinfo=timezone.utc)),
        spec=SimpleNamespace(node_name="node1", containers=[SimpleNamespace(
            name="inference", readiness_probe=probe,
            liveness_probe=SimpleNamespace(http_get=SimpleNamespace(path="/health", port="http")), startup_probe=None)]),
        status=SimpleNamespace(phase=phase, container_statuses=[cs],
                               conditions=[SimpleNamespace(type="Ready", status="True" if ready else "False")]))


def make_deployment(name="ml-inference", desired=2, ready=2, image="mlops-inference:1.0.0"):
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, labels={"app.kubernetes.io/version": "1.0.0"}),
        spec=SimpleNamespace(replicas=desired, selector=SimpleNamespace(match_labels={"app": "ml-inference"}),
                             template=SimpleNamespace(spec=SimpleNamespace(containers=[SimpleNamespace(
                                 name="inference", image=image,
                                 env_from=[SimpleNamespace(config_map_ref=SimpleNamespace(name="cfg"))])]))),
        status=SimpleNamespace(ready_replicas=ready, available_replicas=ready, updated_replicas=desired, conditions=[]))


class FakeCore:
    def __init__(self, pods=None, events=None, logs="INFO ok"):
        self.pods, self.events, self.logs = pods or [], events or [], logs

    def list_namespaced_pod(self, ns, label_selector=None):
        return SimpleNamespace(items=self.pods)

    def read_namespaced_pod(self, name, ns):
        return next(p for p in self.pods if p.metadata.name == name)

    def read_namespaced_pod_log(self, name, ns, **kw):
        return self.logs

    def list_namespaced_event(self, ns):
        return SimpleNamespace(items=self.events)

    def list_namespaced_service(self, ns):
        svc = SimpleNamespace(metadata=SimpleNamespace(name="ml-inference"), spec=SimpleNamespace(
            type="ClusterIP", cluster_ip="10.0.0.1", selector={"app": "ml-inference"},
            ports=[SimpleNamespace(port=80, target_port="http", protocol="TCP")]))
        return SimpleNamespace(items=[svc])

    def read_namespaced_endpoints(self, name, ns):
        ready = [p for p in self.pods if p.status.conditions[0].status == "True"]
        return SimpleNamespace(subsets=[SimpleNamespace(addresses=[1] * len(ready),
                                                        not_ready_addresses=[1] * (len(self.pods) - len(ready)))])

    def read_namespaced_config_map(self, name, ns):
        return SimpleNamespace(data={"MODEL_VERSION": "1", "DATASET_VERSION": "data-v2", "SECRET_KEY": "nope"})


class FakeApps:
    def __init__(self, deployment=None):
        self.deployment, self.patches, self.scales = deployment or make_deployment(), [], []

    def list_namespaced_deployment(self, ns):
        return SimpleNamespace(items=[self.deployment])

    def read_namespaced_deployment(self, name, ns):
        return self.deployment

    def patch_namespaced_deployment(self, name, ns, body):
        self.patches.append((name, ns, body))

    def patch_namespaced_deployment_scale(self, name, ns, body):
        self.scales.append((name, ns, body))


def make_event(obj, reason, message, etype="Warning"):
    return SimpleNamespace(last_timestamp=datetime(2026, 11, 10, tzinfo=timezone.utc), event_time=None, type=etype,
                           reason=reason, involved_object=SimpleNamespace(name=obj), message=message, count=3,
                           metadata=SimpleNamespace(creation_timestamp=None))


class DownMlflow:
    """MLflow client whose server is unreachable."""

    def __getattr__(self, item):
        def boom(*a, **k):
            raise ConnectionError("connection refused token=SUPERSECRET")
        return boom


_ = ServiceUnavailable
