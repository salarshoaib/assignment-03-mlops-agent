# Assignment 03 â€” Production-Ready MLOps AI Agent

**Student:** Salar Shoaib â€” i200830 (i200830@nu.edu.pk)
**Repository:** https://github.com/salarshoaib/assignment-03-mlops-agent
**Evidence:** every screenshot referenced below is committed in `screenshots/`; the raw tool traces
are in `screenshots/evidence/*.json`; the agent's JSON log is written to `logs/agent.jsonl`.

---

## 1. System architecture

```
User
 â”‚
 â–¼
FastAPI  (agent/main.py)      pydantic request validation (extra=forbid), bearer token, X-Request-ID
 â”‚
 â–¼
Orchestrator (agent/orchestrator.py)
 â”‚        â”œâ”€â”€ LLM path: Anthropic tool use (agent/llm.py)
 â”‚        â””â”€â”€ Offline path: deterministic planner (agent/offline_planner.py) â€” used here, no API key
 â–¼
ToolRegistry (agent/tools/registry.py)
 â”‚   1. schema validation   2. precheck (namespace / DAG / conf allow-lists)
 â”‚   3. confirmation gate for mutating tools   4. timeout   5. redaction   6. JSON log
 â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
 DVC     MLflow     Airflow 3  Kubernetes (kind)
```

**Design decisions**

* **Provenance link.** Training writes `dvc.md5`, `dvc.dataset_path`, `dvc.dataset_version` and
  `git.commit` as MLflow tags. `dvc_get_dataset_for_run` reads them back and searches the git
  history of the `.dvc` file for the commit that holds that md5, so "which dataset trained run X"
  is answered from recorded provenance, never from memory.
* **Best â‰  production-ready.** `mlflow_find_best_run` returns readiness checks (finished,
  thresholds, provenance, registry/alias) plus caveats (noise, no online metrics).
* **Pipeline safety.** Validation and the quality gate raise `AirflowFailException` (no retries,
  downstream `upstream_failed`); the registry alias `candidate` is set only by the gate after it
  passes. Other tasks use `retries=2` with exponential back-off.
* **Confirmation outside the LLM.** Mutating tools create a pending action; only
  `Orchestrator.confirm` / `POST /confirm` (human path) executes it. Authorisation is checked
  *before* a pending action exists.
* **Retraining policy.** Recommended only with a failed DAG run / quality gate, a breached
  freshness policy (`MODEL_MAX_AGE_DAYS`), or sub-threshold metrics. A dataset change is reported
  as a consideration, not a trigger.
* **Offline planner.** Same registry, same rules; makes the no-invention guarantee testable without
  an API key.

---

## 2. Tool definitions and permissions

33 tools are registered. All inputs are pydantic models with `extra="forbid"`, so an unknown
argument is rejected before any infrastructure call.

| Group | Tool | Mutating |
|---|---|---|
| DVC | `dvc_list_tracked_datasets`, `dvc_get_dataset_version`, `dvc_compare_versions`, `dvc_detect_changes`, `dvc_get_dataset_for_run`, `dvc_explain_reproduce` | no |
| DVC | `dvc_restore_dataset_version` | **yes â€” confirmation required** |
| MLflow | `mlflow_list_experiments`, `mlflow_list_runs`, `mlflow_get_run_details`, `mlflow_compare_runs`, `mlflow_compare_experiments`, `mlflow_find_best_run`, `mlflow_get_model_registry`, `mlflow_list_run_artifacts` | no |
| Airflow | `airflow_list_dags`, `airflow_get_dag_runs`, `airflow_get_task_instances`, `airflow_get_failed_tasks`, `airflow_get_task_logs`, `airflow_wait_for_dag_run` | no |
| Airflow | `airflow_trigger_dag` | **yes â€” confirmation required** |
| Kubernetes | `k8s_list_pods`, `k8s_list_deployments`, `k8s_list_services`, `k8s_get_pod_status`, `k8s_get_pod_logs`, `k8s_get_events`, `k8s_diagnose_deployment`, `k8s_find_unhealthy_pods`, `k8s_get_deployed_version` | no |
| Kubernetes | `k8s_restart_deployment`, `k8s_scale_deployment` | **yes â€” confirmation required** |

**Permission rules enforced in the registry**

| Rule | Enforced where | Demonstrated by |
|---|---|---|
| Namespace allow-list: only `mlops-agent` | `k8s_precheck` before a pending action is created | `screenshots/24_rejected_namespace.png` â€” `kube-system` refused with `permission_denied`, allowed namespaces `['mlops-agent']` |
| DAG allow-list: only `training_pipeline` | `dag_precheck` | tests `test_dag_not_allowed` |
| Conf allow-list: only `inject_failure` with `quality` | `conf_precheck` | tests `test_conf_not_allowed` |
| Confirmation for every mutating tool | `ToolRegistry._create_pending` | `screenshots/17_agent_s3_trigger_confirm.png` |
| Timeout per call (default 30 s) | `ToolRegistry.execute` | `screenshots/23_failF_mlflow_down.png` |
| Redaction of credentials in output | `redact()` | tests `test_redaction` |
| One JSON line per call | `configure_logging` | `screenshots/25_structured_logs.png` |

---

## 3. Evidence screenshots

### 3.1 Environment and tests

`pytest` â€” **78 passed** (final code, final run):

![pytest](../screenshots/01_pytest_pass.png)

### 3.2 Dataset versioning with DVC (Task 1)

`git log --oneline --decorate` shows the version tag `data-v1`:

![git log with data-v1](../screenshots/02_dvc_git_log.png)

The `.dvc` pointer file and `dvc status` (dataset up to date):

![dvc file and status](../screenshots/03_dvc_file.png)

Two versions in git, and `dvc diff data-v1 data-v2`:

![dvc diff between versions](../screenshots/06_dvc_two_versions.png)

### 3.3 Experiment tracking (Task 2)

Five runs on `data-v1`, table configured to show **f1** and **accuracy**:

![MLflow runs table](../screenshots/04_mlflow_runs.png)

A run page showing the provenance tags `dvc.md5`, `dvc.dataset_path`, `dvc.dataset_version`,
`git.commit`:

![MLflow run tags](../screenshots/05_mlflow_run_tags.png)

Eight runs across both dataset versions â€” the `dvc.dataset_version` column separates them:

![MLflow runs for two datasets](../screenshots/07_mlflow_two_datasets.png)

### 3.4 Pipeline orchestration (Task 3)

Graph view of the five-stage DAG:

![Airflow DAG graph](../screenshots/08_airflow_dag_graph.png)

A successful run â€” every task succeeded:

![Airflow successful run](../screenshots/09_airflow_success_run.png)

A failed run â€” the quality gate rejected the model:

![Airflow failed run](../screenshots/10_airflow_failed_run.png)

Model registry: `ml-inference-model`, version 13 carries the `@candidate` alias set by the gate:

![MLflow model registry](../screenshots/11_mlflow_registry.png)

### 3.5 Kubernetes deployment (Task 4)

`kubectl get deploy,pods,svc` â€” 2/2 replicas ready:

![k8s healthy](../screenshots/12_k8s_healthy.png)

`kubectl describe deploy` â€” probes and resource requests/limits:

![k8s describe](../screenshots/13_k8s_describe.png)

Inference API reachable through the service, `POST /predict` with ten zeros:

![inference ready](../screenshots/14_inference_ready.png)

### 3.6 Agent scenarios

S1 â€” dataset provenance and reproduction:

![S1 provenance](../screenshots/15_agent_s1_provenance.png)

S2 â€” comparison and candidate recommendation:

![S2 selection](../screenshots/16_agent_s2_selection.png)

S3 â€” trigger, pending id, `/confirm`, status:

![S3 trigger and confirm](../screenshots/17_agent_s3_trigger_confirm.png)

S5 â€” multi-tool answer with the tool trace and the three sections:

![S5 multi-tool](../screenshots/19_agent_s5_multitool.png)

S4 / failure A â€” diagnosis:

![S4 diagnosis](../screenshots/18_agent_s4_diagnosis.png)

![failure A diagnosis](../screenshots/21_failA_agent_diagnosis.png)

### 3.7 Failure injection

Pods drop to `0/1` after the broken `MODEL_PATH` is applied:

![failure A injection](../screenshots/20_failA_inject_and_k8s.png)

Quality-gate failure explained by the agent:

![failure C](../screenshots/22_failC_airflow_agent.png)

MLflow stopped â€” the agent reports the tool failure instead of inventing runs:

![failure F](../screenshots/23_failF_mlflow_down.png)

### 3.8 Safety and observability

Namespace `kube-system` refused before a pending action exists:

![rejected namespace](../screenshots/24_rejected_namespace.png)

Structured JSON log (`logs/agent.jsonl`):

![structured logs](../screenshots/25_structured_logs.png)

### 3.9 CI

Pull request #1 with the `lint-and-test` workflow green:

![CI green](../screenshots/26_github_ci_green.png)

---

## 4. Five tested conversations

Tool traces below are copied from `screenshots/evidence/*.json` (produced by
`scripts/smoke_test_agent.py` against `uvicorn agent.main:app --port 8001`).

### S1 â€” â€œWhich dataset version was used to train the best model by f1? How can I reproduce it?â€

**Tools:** `mlflow_find_best_run` (15.0 s) â†’ `dvc_list_tracked_datasets` (8.8 s) â†’
`dvc_get_dataset_for_run` (2.9 s) â†’ `dvc_explain_reproduce` (0.4 s)

**Answer (excerpt):**
> Best of 10 recent finished runs by f1 is `run-6` (f1=0.9020, accuracy=0.900).
> Readiness: run_finished=True, meets_min_f1=True, meets_min_accuracy=True,
> has_dataset_provenance=True, registered_in_model_registry=True,
> has_candidate_or_champion_alias=False.
> DVC: that run recorded `data/raw/dataset.csv` md5 `679b7e2ff6` (version data-v1, git commit 08323a3).
> Reproduce: `git checkout 08323a3b6b -- data/raw/dataset.csv.dvc`, `dvc checkout`, `dvc pull`,
> `md5sum data/raw/dataset.csv`; non-destructive alternative
> `dvc get . data/raw/dataset.csv --rev 08323a3b6b -o artifacts/restored/dataset.csv`.
> **Uncertainties:** offline metrics only; top-2 f1 differ by 0.0064 (may be noise).

### S2 â€” â€œCompare the last five training runs and recommend the best candidate based on F1.â€

**Tools:** `mlflow_list_runs` (0.6 s) â†’ `mlflow_find_best_run(recent_n=5)` (2.1 s) â†’
`airflow_get_dag_runs` (7.7 s) â†’ `mlflow_compare_runs` (1.6 s) â†’
`airflow_get_failed_tasks` (0.9 s) â†’ `airflow_get_task_logs` (1.3 s)

**Answer (excerpt):**
> Best of 5 recent finished runs by f1 is `airflow-â€¦L8yidEyF` (f1=0.8552, accuracy=0.860).
> Readiness: registered_in_model_registry=False, has_candidate_or_champion_alias=False.
> Best f1/precision/recall among the compared runs: `0ab68625`; parameters that differ: `max_depth`.
> Airflow: the last run is failed â€” task `quality_gate` failed,
> `quality gate FAILED: f1 0.8552 < 0.999`.
> **Recommended action:** do not pick solely on the top metric; check dataset, thresholds and
> registry state.

### S3 â€” â€œTrigger the model training pipeline and report whether execution succeeded.â€

**Tools:** `airflow_get_dag_runs` â†’ `airflow_trigger_dag` â†’ (after `/confirm`) `airflow_get_dag_runs`

**Trace (from the JSON evidence):**

```json
{"tool": "airflow_trigger_dag", "arguments": {"dag_id": "training_pipeline", "conf": {}},
 "ok": false, "error_type": "confirmation_required", "duration_ms": 0.11}
```

```json
{"pending_id": "_hrFakdW", "tool": "airflow_trigger_dag",
 "arguments": {"dag_id": "training_pipeline", "conf": {}}, "expires_in_s": 300}
```

**Behaviour:** the agent refuses to execute and returns a pending id; after `/confirm <id>` the run
is created (`state: queued`) and the follow-up question reports the run state from Airflow
(`success` in `screenshots/17_agent_s3_trigger_confirm.png`).

### S4 â€” â€œWhy is my prediction API unavailable?â€ (failure A, also scenario 4)

**Tools:** `k8s_get_deployed_version` (5.2 s) â†’ `k8s_find_unhealthy_pods` (0.03 s) â†’
`k8s_diagnose_deployment` (0.6 s) â†’ `k8s_get_pod_logs`

**Answer (excerpt):**
> ConfigMap `MODEL_PATH=/app/models/does_not_exist.joblib`, `MODEL_VERSION=1`.
> Pods unhealthy: `inference` restarted with last termination `Error, exit=137`.
> Events: liveness `GET /health` â†’ `connection refused` / `context deadline exceeded`;
> readiness `GET /ready` â†’ `connection refused`; pods removed from Service endpoints.
> **Recommended actions:** resolve the deployment problem first â€” it is an infrastructure issue,
> not a modelling issue.

### S5 â€” â€œFind the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.â€

**Tools (7 calls, one turn):** `mlflow_find_best_run` (1.8 s) â†’ `k8s_get_deployed_version` (0.02 s) â†’
`k8s_find_unhealthy_pods` (0.01 s) â†’ `airflow_get_dag_runs` (0.08 s) â†’
`dvc_get_dataset_for_run` (0.3 s) â†’ `airflow_get_failed_tasks` (0.06 s) â†’
`airflow_get_task_logs` (0.05 s)

**Answer (excerpt):**
> Best of 10 recent finished runs by f1 is `run-6` (0.9020) trained on `data-v1`.
> Kubernetes: `ml-inference`, image `mlops-inference:1.0.0` (label 1.0.0), 2/2 replicas ready.
> Airflow: last run failed â€” `quality gate FAILED: f1 0.8552 < 0.999`.
> **Retraining assessment:** evidence supports retraining (the latest DAG run failed at the
> quality gate); the agent will only trigger it on an explicit, confirmed request.
> **Uncertainties:** offline metrics only; dataset changed since training is a reason to evaluate,
> not proof of degradation.

---

## 5. Failure investigations

### Failure A â€” broken model path (scenario 4)

| | |
|---|---|
| **Symptom** | `kubectl get pods` â†’ `0/1`; the prediction API is unavailable |
| **Injection** | `python scripts/failure_k8s.py bad-model-path` patches the ConfigMap to `MODEL_PATH=/app/models/does_not_exist.joblib` and restarts the Deployment |
| **Evidence** | pod events: readiness/liveness probes on `/ready` and `/health` failing with `connection refused`; container restart with `exit=137`; `screenshots/20_failA_inject_and_k8s.png`, `screenshots/18_agent_s4_diagnosis.png` |
| **Root cause** | the process never loads a model, so it never serves traffic; the probes then remove the pod from the Service endpoints |
| **Resolution** | `python scripts/failure_k8s.py restore` re-applies `configmap.yaml` and `deployment.yaml`; pods return to `1/1` |

### Failure C â€” quality gate

| | |
|---|---|
| **Symptom** | DAG run ends `failed` at `quality_gate` |
| **Injection** | trigger with `{"inject_failure": "quality"}` (conf allow-list permits exactly this) |
| **Evidence** | `quality gate FAILED: f1 0.8552 < â€¦`; earlier tasks succeeded; `screenshots/10_airflow_failed_run.png`, `screenshots/22_failC_airflow_agent.png` |
| **Root cause** | the gate blocks promotion when the metric threshold is not met; `AirflowFailException` disables retries so downstream tasks stay `upstream_failed` and the `candidate` alias is not moved |
| **Resolution** | no infrastructure fix required â€” re-run without the injection |

### Failure F â€” MLflow unavailable

| | |
|---|---|
| **Symptom** | â€œFind the best model by f1â€ returns no data |
| **Injection** | `docker compose stop mlflow` |
| **Evidence** | `mlflow_find_best_run` â†’ `status=timeout` after 30 s; the answer says â€œNone â€” no tool returned usable dataâ€ instead of inventing runs; `screenshots/23_failF_mlflow_down.png` |
| **Resolution** | `docker compose start mlflow`; the tracker recovers and the same question succeeds |

---

## 6. Tests, lint and CI

* `pytest` â†’ **78 passed** (`screenshots/01_pytest_pass.png`). Suite is fully offline: every
  infrastructure call is mocked or local.
* CI (`/.github/workflows/ci.yml`) runs lint + tests on every pull request; PR #1 is green
  (`screenshots/26_github_ci_green.png`, job `lint-and-test`, 1 m 33 s) and was merged.

---

## 7. Known limitations and improvements

* Sessions and pending actions are held in memory (single process); a restart drops them. A shared
  store (Redis/DB) would be the next step.
* A timed-out tool thread is not killed â€” the call returns, but the worker keeps running.
* Airflow log evidence depends on the REST API being reachable.
* No drift or online metrics: only offline test-set metrics are recorded, so every answer carries
  that caveat.
* The DVC remote in this setup is local (`../dvc-remote`), not a shared object store.
* The demo host has 8 GB of RAM; MLflow + Airflow + a kind cluster together occasionally starve the
  Docker VM, which shows up as pod restarts and slower DAG runs.

---

## 8. Note on evidence timing

The screenshots were taken in the order of the brief: scenarios S1/S2/S3/S5 before the failure
injections, then failure A (S4), failure C and failure F. The JSON evidence in
`screenshots/evidence/` was captured **after** failure C, so S2 and S5 there correctly report the
failed quality-gate run and conclude that retraining is justified â€” the earlier screenshots show
the same questions answered before that run existed. Both are real outputs from the same agent;
only the cluster/experiment state differs.
