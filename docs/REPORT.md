# Technical Report — MLOps Agent

All evidence below was produced on this machine; screenshots are in `screenshots/` and the
machine-readable agent evidence (tool traces) is in `screenshots/evidence/`.

## 1. System architecture

```
client -> FastAPI (agent)   pydantic validation (extra=forbid), bearer token, X-Request-ID
            |
        Orchestrator        LLM (Anthropic tool use)  OR  OfflineClient (deterministic planner)
            |
        ToolRegistry        schema -> precheck (namespace/DAG/conf allow-list) -> confirmation gate
            |               -> timeout -> redaction -> JSON log
      +-----+-----+---------+---------+
      DVC   MLflow  Airflow  Kubernetes
```

33 tools are registered (DVC, MLflow, Airflow, Kubernetes). Every call is schema-validated and
written to `logs/agent.jsonl` as one JSON object per line (`ts`, `level`, `tool`, `status`,
`duration_ms`, `request_id`, `error_type`).

## 2. Tool permissions

| Class | Behaviour |
|---|---|
| Read-only (get/list/find/diff) | executed immediately after schema validation |
| Mutating (`airflow_trigger_dag`, `airflow_restart_deployment`, `k8s_restart_deployment`, `dvc_checkout`, …) | create a pending action, return `confirmation_required`; only `/confirm <id>` (human path) executes it |
| Namespace allow-list | only `mlops-agent`; `kube-system` is refused **before** a pending action exists |
| DAG allow-list | only `training_pipeline` |
| Conf allow-list | only `inject_failure` with values `quality` (used by the failure demo) |

## 3. Evidence

| # | File | Shows |
|---|---|---|
| 01 | `01_pytest_pass.png` | `78 passed` |
| 02 | `02_dvc_git_log.png` | `git log --oneline --decorate` with `tag: data-v1` |
| 03 | `03_dvc_file.png` | `dataset.csv.dvc` + `dvc status` (up to date) |
| 04 | `04_mlflow_runs.png` | runs table with **f1** and **accuracy** columns (5 runs, data-v1) |
| 05 | `05_mlflow_run_tags.png` | `dvc.md5`, `dvc.dataset_path`, `dvc.dataset_version`, `git.commit` |
| 06 | `06_dvc_two_versions.png` | `git tag` + `dvc diff data-v1 data-v2` |
| 07 | `07_mlflow_two_datasets.png` | 8 runs with `dvc.dataset_version` = data-v1 / data-v2 |
| 08 | `08_airflow_dag_graph.png` | Graph tab, five stages |
| 09 | `09_airflow_success_run.png` | green run, all tasks succeeded |
| 10 | `10_airflow_failed_run.png` | quality-gate failure run (failed) |
| 11 | `11_mlflow_registry.png` | model registry with `@candidate` alias |
| 12 | `12_k8s_healthy.png` | `kubectl get deploy,pods,svc`, 2/2 ready |
| 13 | `13_k8s_describe.png` | `describe deploy` with probes and resources |
| 14 | `14_inference_ready.png` | `/docs` POST /predict with ten zeros |
| 15–19 | `15…19` | agent scenarios S1, S2, S3, S5 |
| 18/21 | `18_agent_s4_diagnosis.png`, `21_failA_agent_diagnosis.png` | S4 / failure-A diagnosis |
| 20 | `20_failA_inject_and_k8s.png` | pods `0/1` after broken `MODEL_PATH` |
| 22 | `22_failC_airflow_agent.png` | agent explains the quality-gate failure |
| 23 | `23_failF_mlflow_down.png` | agent degrades gracefully with MLflow stopped |
| 24 | `24_rejected_namespace.png` | `kube-system` restart refused (`permission_denied`) |
| 25 | `25_structured_logs.png` | `logs/agent.jsonl` tail |
| 26 | `26_github_ci_green.png` | PR #1, `lint-and-test` succeeded |

## 4. Five tested conversations

Full tool traces: `screenshots/evidence/1_dataset_provenance.json` … `5_multi_tool.json`.

**S1 — provenance.** Tools: `mlflow_find_best_run`, `dvc_get_dataset_for_run`. Best run by f1 is
`run-6` (f1 0.9020, accuracy 0.900); provenance tags give dataset `data/raw/dataset.csv`, md5
`670d7a2ffe…`, `data-v1`, git commit `08323a3`. Reproduction = `dvc checkout data-v1` then
`python scripts/run_experiments.py`.

**S2 — selection.** Tools: `mlflow_find_best_run`. The last five runs are compared on f1; runs on
different dataset versions are flagged as not directly comparable, and the top-2 gap is reported as
possibly within noise.

**S3 — trigger + confirm.** Tool: `airflow_trigger_dag` returned `confirmation_required`
(pending id). After `/confirm`, the run was created (`state: queued`) and the follow-up question
reported the run as **success**.

**S4 — diagnosis.** Tools: `k8s_get_deployed_version`, `k8s_find_unhealthy_pods`,
`k8s_diagnose_deployment`, `k8s_get_pod_logs`.

**S5 — multi-tool.** Five tools in one turn: `mlflow_find_best_run`, `k8s_get_deployed_version`,
`k8s_find_unhealthy_pods`, `airflow_get_dag_runs`, `dvc_get_dataset_for_run`. Answer: best f1 model
is `run-6` (0.9020) trained on data-v1; deployment `ml-inference` runs image
`mlops-inference:1.0.0` with 2/2 replicas ready; no failed quality gate or policy breach, so
retraining is not required, while the dataset change (`data-v2` tracked) is a reason to evaluate.

## 5. Failure investigations

### Failure A — broken model path (also scenario 4)
* Symptom: `kubectl get pods` shows `0/1`; the prediction API is unavailable.
* Evidence: configmap `MODEL_PATH=/app/models/does_not_exist.joblib`; pod events show readiness and
  liveness probe failures on `/health` and `/ready` with `connection refused`; container restarted
  with exit code 137.
* Root cause: the deployment points at a model artefact that does not exist, so the process never
  serves traffic and probes remove the pod from Service endpoints.
* Resolution: `python scripts/failure_k8s.py restore` re-applies the configmap and deployment;
  pods return to `1/1`.

### Failure C — quality gate
* Trigger: DAG run with config `{"inject_failure": "quality"}`.
* Evidence: run `failed`; task `quality_gate` failed with `quality gate FAILED: f1 0.8552 < …`;
  all earlier tasks succeeded.
* Root cause: the gate blocks promotion when the metric threshold is not met — by design no
  retries (`AirflowFailException`), so downstream stays blocked and the `candidate` alias is not moved.
* Resolution: no infrastructure fix needed; re-run without the injection.

### Failure F — MLflow unavailable
* Symptom: `docker compose stop mlflow`, then "Find the best model by f1".
* Evidence: `mlflow_find_best_run` returned `status=timeout` after 30 s; the answer reports
  "None — no tool returned usable data" instead of inventing runs.
* Resolution: `docker compose start mlflow`; the tracker recovers.

## 6. Known limitations
* Sessions and pending actions are in memory (single process); a restart drops them.
* A timed-out tool thread is not killed.
* Airflow log evidence depends on API availability.
* No drift / online metrics; offline test-set metrics only.
* The local Docker host (8 GB RAM) is the limiting factor for running MLflow, Airflow and kind at
  the same time; DAG runs take longer under load.
