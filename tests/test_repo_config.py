import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "kubernetes"


def load(name):
    return list(yaml.safe_load_all((K / name).read_text()))[0]


def test_deployment_requirements():
    d = load("deployment.yaml")
    assert d["spec"]["replicas"] >= 2
    c = d["spec"]["template"]["spec"]["containers"][0]
    assert c["readinessProbe"]["httpGet"]["path"] == "/ready" and c["livenessProbe"]["httpGet"]["path"] == "/health"
    for kind in ("requests", "limits"):
        assert {"cpu", "memory"} <= set(c["resources"][kind])


def test_service_and_configmap():
    assert load("service.yaml")["spec"]["type"] == "ClusterIP"
    cm = load("configmap.yaml")
    assert cm["data"]["MODEL_PATH"]
    assert not any(w in k.upper() for k in cm["data"] for w in ("PASSWORD", "SECRET", "TOKEN", "KEY"))


def test_all_manifests_use_approved_namespace():
    for f in ("deployment", "service", "configmap", "rbac"):
        for doc in yaml.safe_load_all((K / f"{f}.yaml").read_text()):
            assert doc["metadata"]["namespace"] == "mlops-agent"


def test_dag_defines_required_stages_in_order_with_retries():
    src = (ROOT / "dags" / "training_pipeline.py").read_text()
    tree = ast.parse(src)
    names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    for stage in ("validate_dataset", "preprocess_data", "train_model", "evaluate_and_log", "quality_gate"):
        assert stage in names
    assert '"retries": 2' in src and "AirflowFailException" in src
    chain = src.index("quality_gate(evaluate_and_log(train_model(preprocess_data(validate_dataset()))))")
    assert chain > 0


def test_ci_runs_lint_and_tests_on_pull_requests():
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert "pull_request" in ci[True] or "pull_request" in ci.get("on", {})
    text = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "ruff check" in text and "pytest" in text


def test_no_secrets_committed():
    for f in ROOT.rglob("*"):
        if f.is_file() and f.suffix in {".py", ".yaml", ".yml", ".md"} and ".git/" not in str(f) and "tests" not in f.parts:
            t = f.read_text(errors="ignore")
            assert "sk-ant-" not in t, f
