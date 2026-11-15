# Assignment checklist

Checked line by line against the brief. ✅ = done and evidenced, ⚠️ = done with a caveat.
Links are relative to this file, so they work on GitHub.

## Phase 0 — tools
| Item | Status |
|---|---|
| Python 3.11 or newer | ✅ 3.12.2 |
| Git | ✅ 2.47.1 |
| Docker Desktop running | ✅ 29.7.2 |
| kubectl | ✅ v1.36.1 |
| kind | ✅ installed, cluster `mlops` created |
| git identity | ✅ `salarshoaib <i200830@nu.edu.pk>` |
| GitHub account | ✅ salarshoaib |

## Phase 1 — project and tests
| Item | Status |
|---|---|
| `.venv` created and activated | ✅ |
| `pip install -r requirements.txt` | ✅ |
| `pytest` | ✅ **78 passed** → [`01_pytest_pass.png`](../screenshots/01_pytest_pass.png) |
| `.env` copied from `.env.example` | ✅ left without an API key → deterministic offline planner |
| `.env` never committed | ✅ not in `git ls-files` |

## Phase 2 — DVC dataset v1
| Item | Status |
|---|---|
| `python scripts/bootstrap_dvc.py --version 1` | ✅ 2000 rows, tag `data-v1` |
| `git log --oneline --decorate` shows the tag | ✅ [`02_dvc_git_log.png`](../screenshots/02_dvc_git_log.png) |
| `.dvc` file + `dvc status` | ✅ [`03_dvc_file.png`](../screenshots/03_dvc_file.png) |

## Phase 3 — MLflow and Airflow
| Item | Status |
|---|---|
| `docker compose up -d --build` | ✅ |
| `docker compose ps` → both services up | ✅ MLflow :5000, Airflow :8080 |
| MLflow UI at localhost:5000 | ✅ |
| Airflow UI at localhost:8080 (admin/admin) | ✅ |
| Port 5000 free | ✅ no conflict |

**Two fixes were required to make the stack run on this machine**
* `Dockerfile.mlflow`: pin `sqlalchemy>=2.0,<2.1` — MLflow 2.22 imports
  `FallbackAsyncAdaptedQueuePool`, which SQLAlchemy 2.1 removed, so the tracking server crashed on boot.
* `agent/tools/mlflow_tools.py`: resolve registry aliases through the registered model, because
  `search_model_versions()` in MLflow 2.x returns an empty `ModelVersion.aliases`; without the fix the
  readiness check reported “no candidate alias” while `@candidate` existed.
* `tests/conftest.py`: pass a real `file://` URI for the artifact location (on Windows
  `get_uri_scheme("c:\\…")` returns `c`, and MLflow registered an unusable backend URI).

## Phase 4 — experiments on dataset v1
| Item | Status |
|---|---|
| `python scripts/run_experiments.py --count 5` | ✅ runs `run-4` … `run-8` |
| Runs table with **f1** and **accuracy** columns | ✅ [`04_mlflow_runs.png`](../screenshots/04_mlflow_runs.png) |
| Run page with `dvc.md5`, `dvc.dataset_path`, `git.commit` | ✅ [`05_mlflow_run_tags.png`](../screenshots/05_mlflow_run_tags.png) (also `dvc.dataset_version`) |

## Phase 5 — dataset v2 and more runs
| Item | Status |
|---|---|
| `git add -A && git commit -m "Add project"` | ✅ `e4d0e44` |
| `python scripts/bootstrap_dvc.py --version 2` | ✅ 3000 rows, tag `data-v2` |
| `python scripts/run_experiments.py --count 3` | ✅ runs `run-9` … `run-11` |
| `git tag` + `dvc diff data-v1 data-v2` | ✅ [`06_dvc_two_versions.png`](../screenshots/06_dvc_two_versions.png) |
| Runs from both versions in one table | ✅ [`07_mlflow_two_datasets.png`](../screenshots/07_mlflow_two_datasets.png) (`dvc.dataset_version` column) |

## Phase 6 — Airflow DAG
| Item | Status |
|---|---|
| `training_pipeline` visible, no import error | ✅ DAG parse ≈ 7 s |
| Trigger → run turns green | ✅ `manual__2026-11-15T15:24:18` succeeded |
| Graph tab, five stages | ✅ [`08_airflow_dag_graph.png`](../screenshots/08_airflow_dag_graph.png) |
| Green run, every task succeeded | ✅ [`09_airflow_success_run.png`](../screenshots/09_airflow_success_run.png) |
| MLflow Models page with the `candidate` alias | ✅ [`11_mlflow_registry.png`](../screenshots/11_mlflow_registry.png) (version 13, `@candidate`) |

## Phase 7 — Kubernetes
| Item | Status |
|---|---|
| `docker build -t mlops-inference:1.0.0 .` | ✅ |
| `kind create cluster --name mlops` | ✅ node `mlops-control-plane` |
| `kind load docker-image` | ✅ |
| `kubectl apply` (namespace, configmap, deployment, service) | ✅ |
| `kubectl rollout status` | ✅ |
| `get deploy,pods,svc` → 2/2 ready | ✅ [`12_k8s_healthy.png`](../screenshots/12_k8s_healthy.png) |
| `describe deploy` with probes and resources | ✅ [`13_k8s_describe.png`](../screenshots/13_k8s_describe.png) |
| port-forward + `/docs` `POST /predict` with ten zeros | ✅ [`14_inference_ready.png`](../screenshots/14_inference_ready.png) |

## Phase 8 — the agent and five scenarios
| Item | Status |
|---|---|
| `python -m agent.cli` | ✅ 33 tools, deterministic offline planner |
| S1 provenance | ✅ [`15_agent_s1_provenance.png`](../screenshots/15_agent_s1_provenance.png) — best `run-6` (f1 0.9020), `data-v1`, md5 `679b7e2ff6`, commit `08323a3` |
| S2 selection | ✅ [`16_agent_s2_selection.png`](../screenshots/16_agent_s2_selection.png) |
| S3 trigger + `/confirm` + status | ✅ [`17_agent_s3_trigger_confirm.png`](../screenshots/17_agent_s3_trigger_confirm.png) — pending id → executed → run success |
| S5 multi-tool | ✅ [`19_agent_s5_multitool.png`](../screenshots/19_agent_s5_multitool.png) — five tools, tool trace, three sections |
| Safety: restart in `kube-system` refused | ✅ [`24_rejected_namespace.png`](../screenshots/24_rejected_namespace.png) — `permission_denied`, allowed `['mlops-agent']` |
| `tail -5 logs/agent.jsonl` | ✅ [`25_structured_logs.png`](../screenshots/25_structured_logs.png) |
| `uvicorn agent.main:app --port 8001` + `smoke_test_agent.py` | ✅ five JSON files in [`screenshots/evidence/`](../screenshots/evidence) |

## Phase 9 — failure injections
| Item | Status |
|---|---|
| A: `failure_k8s.py bad-model-path` → pods `0/1` | ✅ [`20_failA_inject_and_k8s.png`](../screenshots/20_failA_inject_and_k8s.png) |
| A: “Why is my prediction API unavailable?” | ✅ [`18_agent_s4_diagnosis.png`](../screenshots/18_agent_s4_diagnosis.png), [`21_failA_agent_diagnosis.png`](../screenshots/21_failA_agent_diagnosis.png) |
| A: `failure_k8s.py restore` | ✅ pods back to `1/1` |
| C: trigger with `{"inject_failure": "quality"}` → red run | ✅ [`10_airflow_failed_run.png`](../screenshots/10_airflow_failed_run.png) |
| C: “Why did the last training run fail?” | ✅ [`22_failC_airflow_agent.png`](../screenshots/22_failC_airflow_agent.png) — `quality gate FAILED: f1 0.8552` |
| F: `docker compose stop mlflow` + “Find the best model by f1” | ✅ [`23_failF_mlflow_down.png`](../screenshots/23_failF_mlflow_down.png) — timeout, nothing invented |
| F: `docker compose start mlflow` | ✅ tracker restored |

## Phase 10 — GitHub and CI
| Item | Status |
|---|---|
| Repository created | ✅ https://github.com/salarshoaib/assignment-03-mlops-agent (**public**) |
| `.env` not committed | ✅ |
| Branch `feature/ci-demo` + commit “Complete assignment” | ✅ `620f4a1` |
| Push + pull request | ✅ PR #1 |
| Actions check green | ✅ `lint-and-test`, 1 m 33 s → [`26_github_ci_green.png`](../screenshots/26_github_ci_green.png) |
| PR merged | ✅ merged 2026-11-15T17:21:52Z |
| Report and checklist committed | ✅ [`REPORT.md`](REPORT.md), this file |

## Caveats a marker should know
* The machine has 8 GB RAM; MLflow + Airflow + kind together periodically starved the Docker VM.
  Two visible consequences: the `ml-inference` pods carry restart counts from those outages, and one
  older DAG run inside the S3 screenshot is marked failed because a VM outage cut it off.
* Screenshots 18 and 21 are the same diagnosis screen captured twice — the brief asks for both.
* Runs created while re-taking scenario 3 were deleted from MLflow so that “best by f1” stays
  `run-6` (0.9020) across every screenshot.
* `candidate` points at model version 13, created by the last successful DAG run.
* The JSON evidence was captured after failure C, so S2/S5 there report the failed quality gate;
  see §8 of the report for the ordering.
