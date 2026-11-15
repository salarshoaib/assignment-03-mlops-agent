# Assignment checklist

Checked against the brief, step by step. ✅ = done and evidenced, ⚠️ = done with a caveat.

## Phase 0 — tools
| Item | Status |
|---|---|
| Python 3.11+ | ✅ 3.12.2 |
| Git | ✅ 2.47.1 |
| Docker Desktop running | ✅ 29.7.2 |
| kubectl | ✅ v1.36.1 |
| kind | ✅ installed (`~/go/bin/kind`), cluster `mlops` created |
| git identity | ✅ `salarshoaib <i200830@nu.edu.pk>` |
| GitHub account | ✅ salarshoaib |

## Phase 1 — project and tests
| Item | Status |
|---|---|
| `.venv` created and used | ✅ |
| `pip install -r requirements.txt` | ✅ |
| `pytest` | ✅ **78 passed** — `01_pytest_pass.png` |
| `.env` copied from `.env.example` | ✅ left without an API key → deterministic offline planner |
| `.env` excluded from git | ✅ `git ls-files` has 96 entries, none is `.env` |

## Phase 2 — DVC v1
| Item | Status |
|---|---|
| `bootstrap_dvc.py --version 1` | ✅ dataset 2000 rows, tag `data-v1` |
| `git log --oneline --decorate` shows `tag: data-v1` | ✅ `02_dvc_git_log.png` |
| `.dvc` file + `dvc status` | ✅ `03_dvc_file.png` |

## Phase 3 — MLflow and Airflow
| Item | Status |
|---|---|
| `docker compose up -d --build` | ✅ (first build ~8 min) |
| `docker compose ps` both services running | ✅ MLflow :5000, Airflow :8080 |
| MLflow UI reachable | ✅ |
| Airflow UI reachable (admin/admin) | ✅ |
| Port 5000 free | ✅ no conflict |

**Fixes applied to make the stack runnable**
* `Dockerfile.mlflow`: pin `sqlalchemy>=2.0,<2.1` — MLflow 2.22 imports
  `FallbackAsyncAdaptedQueuePool`, removed in SQLAlchemy 2.1 (server crashed on boot otherwise).
* `agent/tools/mlflow_tools.py`: resolve registry aliases through the registered model, because
  `search_model_versions()` in MLflow 2.x returns an empty `ModelVersion.aliases`. Without it the
  readiness check reported `has_candidate_or_champion_alias=False` even when `@candidate` existed.
* `tests/conftest.py`: pass a proper `file://` URI for the artifact location (on Windows
  `get_uri_scheme('c:\\…')` returns `c`, so MLflow registered an unusable backend URI).

## Phase 4 — experiments on data-v1
| Item | Status |
|---|---|
| `run_experiments.py --count 5` | ✅ runs `run-4` … `run-8` |
| Runs table with f1 and accuracy columns | ✅ `04_mlflow_runs.png` |
| Run page with `dvc.md5`, `dvc.dataset_path`, `git.commit` | ✅ `05_mlflow_run_tags.png` (also `dvc.dataset_version`) |

## Phase 5 — dataset v2
| Item | Status |
|---|---|
| `git add -A && git commit -m "Add project"` | ✅ `e4d0e44` |
| `bootstrap_dvc.py --version 2` | ✅ 3000 rows, tag `data-v2` |
| `run_experiments.py --count 3` | ✅ `run-9` … `run-11` |
| `git tag` + `dvc diff data-v1 data-v2` | ✅ `06_dvc_two_versions.png` |
| Runs from both versions in one table | ✅ `07_mlflow_two_datasets.png` (`dvc.dataset_version` column) |

## Phase 6 — Airflow DAG
| Item | Status |
|---|---|
| `training_pipeline` found, no import error | ✅ DAG parse time 7 s |
| Trigger → green run | ✅ run `manual__2026-11-15T15:24:18` (success) |
| Graph tab, five stages | ✅ `08_airflow_dag_graph.png` |
| Green run, every task succeeded | ✅ `09_airflow_success_run.png` |
| MLflow Models page with the candidate alias | ✅ `11_mlflow_registry.png` |

## Phase 7 — Kubernetes
| Item | Status |
|---|---|
| `docker build -t mlops-inference:1.0.0 .` | ✅ |
| `kind create cluster --name mlops` | ✅ kind v0.31, node `mlops-control-plane` |
| `kind load docker-image` | ✅ |
| `kubectl apply` (namespace, configmap, deployment, service) | ✅ |
| `rollout status` | ✅ 2/2 ready — `12_k8s_healthy.png` |
| `get deploy,pods,svc` | ✅ 2/2 ready — `12_k8s_healthy.png` |
| `describe deploy` (probes + resources) | ✅ `13_k8s_describe.png` |
| port-forward + `/docs` POST /predict (ten zeros) | ✅ `14_inference_ready.png` |

## Phase 8 — agent and scenarios
| Item | Status |
|---|---|
| `python -m agent.cli` | ✅ offline planner, 33 tools |
| S1 provenance | ✅ `15_agent_s1_provenance.png` — best `run-6` (f1 0.9020) trained on `data-v1`, md5 `670d7a2ffe…`, commit `08323a3` |
| S2 selection | ✅ `16_agent_s2_selection.png` |
| S3 trigger + `/confirm` + status | ✅ `17_agent_s3_trigger_confirm.png` — pending id → executed → run success |
| S5 multi-tool | ✅ `19_agent_s5_multitool.png` — 5 tools, three sections |
| Safety: restart in `kube-system` refused | ✅ `24_rejected_namespace.png` — `permission_denied`, allowed namespaces `['mlops-agent']` |
| `tail -5 logs/agent.jsonl` | ✅ `25_structured_logs.png` |
| `uvicorn agent.main:app --port 8001` + `smoke_test_agent.py` | ✅ five JSON files in `screenshots/evidence/` |

## Phase 9 — failure injections
| Item | Status |
|---|---|
| A: `failure_k8s.py bad-model-path` → pods `0/1` | ✅ `20_failA_inject_and_k8s.png` |
| A: agent explains "Why is my prediction API unavailable?" | ✅ `18_agent_s4_diagnosis.png`, `21_failA_agent_diagnosis.png` |
| A: `failure_k8s.py restore` | ✅ pods back to `1/1` |
| C: trigger with `{"inject_failure": "quality"}` → red run | ✅ `10_airflow_failed_run.png` |
| C: agent explains "Why did the last training run fail?" | ✅ `22_failC_airflow_agent.png` — quality gate failed, f1 0.8552 |
| F: `docker compose stop mlflow` + "Find the best model by f1" | ✅ `23_failF_mlflow_down.png` — timeout, no invented data |
| F: `docker compose start mlflow` | ✅ |

## Phase 10 — GitHub and CI
| Item | Status |
|---|---|
| Empty repo created | ✅ `salarshoaib/assignment-03-mlops-agent` (private) |
| `.env` not committed | ✅ |
| Branch `feature/ci-demo` + commit "Complete assignment" | ✅ `620f4a1` |
| Push + pull request | ✅ PR #1 |
| Actions check green | ✅ `lint-and-test` — 1 m 33 s — `26_github_ci_green.png` |
| PR merged | ✅ merged 2026-11-15T17:21:52Z |

## Caveats to be aware of when marking
* The machine has 8 GB RAM; MLflow + Airflow + kind together periodically starved the Docker VM.
  Two consequences are visible in the evidence: the `ml-inference` pods show restart counts from
  those outages, and the S3 screenshot reports one earlier run as failed (that run was cut off by
  a VM outage, not by the DAG logic).
* Screenshots 18 and 21 are the same diagnosis screen captured twice (the brief asks for both).
* Runs created while re-taking scenario 3 were removed from MLflow so that "best by f1" stays
  consistent across screenshots (`run-6`, f1 0.9020).
* The `candidate` alias sits on model version 13, created by the last successful DAG run.
