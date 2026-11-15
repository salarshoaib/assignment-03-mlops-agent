# Assignment 03 - Production-Ready MLOps AI Agent

A conversational agent that **uses real infrastructure** (DVC, MLflow, Apache Airflow 3, Kubernetes) through
structured, permissioned tools and answers only from tool evidence.

```
User -> FastAPI (validation) -> Orchestrator (LLM tool selection) -> Tool registry (schema check, allow-lists,
confirmation, timeouts, structured logs) -> DVC | MLflow | Airflow | Kubernetes -> evidence-based answer
```
See `architecture.md` for the full design and `docs/` for demo scripts, failure injections and the report template.

## Submission

* **Technical report:** [`docs/REPORT.md`](docs/REPORT.md) — architecture, tool permissions, all evidence
  screenshots inline, the five tested conversations with tool traces, failure investigations, limitations.
* **Verification checklist:** [`docs/CHECKLIST.md`](docs/CHECKLIST.md) — every step of the brief, checked.
* **Evidence:** `screenshots/` (26 screenshots) and `screenshots/evidence/*.json` (raw tool traces).


## 0. Prerequisites
Python 3.11+, Git, Docker Desktop, `kubectl`, `kind` (or minikube). Optional: Anthropic API key
(without it the agent uses a deterministic offline planner - same tools, same evidence rules).

## 1. Python environment
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                      # edit; never commit .env
pytest                                                    # 78 tests, everything mocked / local
ruff check .
```

## 2. DVC dataset versions (Task 1)
```bash
python scripts/bootstrap_dvc.py --version 1     # git init, dvc init, local remote, dvc add, tag data-v1, dvc push
```
Dataset pointer `data/raw/dataset.csv.dvc` is committed to git; data lives in the DVC remote (`../dvc-remote`).

## 3. MLflow + Airflow (Tasks 2 and 3)
```bash
docker compose up -d --build        # MLflow :5000 (set MLFLOW_PORT=5001 if 5000 is taken), Airflow :8080 (admin/admin)
python scripts/run_experiments.py --count 5          # 5 runs on data-v1, tagged with DVC md5/path + git commit
python scripts/bootstrap_dvc.py --version 2          # new dataset version
python scripts/run_experiments.py --count 3          # 3 runs on data-v2
```
MLflow UI: http://localhost:5000 - Airflow UI: http://localhost:8080 (DAG `training_pipeline`, unpaused by default).
Training the model that Kubernetes serves: `python training/train.py --run-name final --no-gate` (writes `models/model.joblib`).

## 4. Kubernetes (Task 4)
```bash
kind create cluster --name mlops
docker build -t mlops-inference:1.0.0 .             # needs models/model.joblib from step 3
kind load docker-image mlops-inference:1.0.0 --name mlops
kubectl apply -f kubernetes/namespace.yaml -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml
kubectl -n mlops-agent get pods,svc
kubectl -n mlops-agent port-forward svc/ml-inference 8000:80    # then curl localhost:8000/ready
```
Optional least-privilege identity: `kubernetes/rbac.yaml`.

## 5. Run the agent
```bash
python -m agent.cli                                  # interactive terminal client (/tools, /confirm <id>, /quit)
uvicorn agent.main:app --port 8001                   # REST: POST /chat, POST /confirm, GET /tools, GET /health
python scripts/smoke_test_agent.py                   # runs the 5 demo scenarios, saves JSON evidence
```
Mutating operations (Kubernetes restart/scale, Airflow trigger) never execute directly: the agent returns a
`pending_id`; **you** confirm with `/confirm <id>` (CLI) or `POST /confirm`. The LLM cannot confirm on its own.

## 6. Tool inventory
| Module | Tools |
|---|---|
| DVC | list_tracked_datasets, get_dataset_version, compare_versions, detect_changes, explain_reproduce, get_dataset_for_run, restore_dataset_version |
| MLflow | list_experiments, list_runs, get_run_details, find_best_run, compare_runs, compare_experiments, get_model_registry, list_run_artifacts |
| Airflow | list_dags, get_dag_runs, trigger_dag*, get_task_instances, get_failed_tasks, get_task_logs, wait_for_dag_run |
| Kubernetes | list_deployments, list_pods, get_pod_status, get_pod_logs, list_services, get_events, find_unhealthy_pods, get_deployed_version, diagnose_deployment, restart_deployment*, scale_deployment* |

`*` = mutating, requires human confirmation. Kubernetes: approved namespace only. Airflow: approved DAG and conf keys only.
DVC/git: allow-listed read-only sub-commands, no shell.

## 7. Safety summary
Evidence-only answers (system prompt + deterministic report builder) - allow-listed commands - namespace/DAG policy checked
**before** confirmation - confirmation tokens (single use, 5-min TTL) - secrets redacted from logs, tool output and errors -
per-tool timeouts - structured JSON logs (tool, status, duration, request id) in `logs/agent.jsonl`.

## 8. Tests
`pytest` runs 78 tests: tool selection/execution, validation, timeouts, service failures, no-invention when a tool is
down, unauthorised Kubernetes changes rejected, confirmation flow, log redaction, real MLflow (sqlite) logic, temp git/DVC repos,
pipeline stages + quality gate, inference API, manifests, DAG structure, CI config.
