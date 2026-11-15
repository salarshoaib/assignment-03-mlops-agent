# Five demo scenarios (use `python -m agent.cli` or `scripts/smoke_test_agent.py`)

| # | Prompt | Expected tool evidence |
|---|---|---|
| 1 Dataset provenance | `Which dataset version was used to train the best model by f1? How can I reproduce it?` | mlflow_find_best_run -> dvc_get_dataset_for_run (+ dvc_explain_reproduce with that run_id) |
| 2 Model selection | `Compare the last five training runs and recommend the best candidate based on F1 score.` | mlflow_list_runs -> mlflow_compare_runs |
| 3 Pipeline execution | `Trigger the model training pipeline and report whether execution succeeded.` | airflow_trigger_dag (pending) -> `/confirm <id>` -> `Check status of the latest training run` |
| 4 Deployment diagnosis | `Why is my prediction API unavailable?` | k8s_get_deployed_version, k8s_find_unhealthy_pods, k8s_diagnose_deployment |
| 5 Multi-tool | `Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.` | MLflow + DVC + Kubernetes + Airflow, NO trigger |

Reproduce a dataset version for real: `python -m agent.cli` then call tool via prompt
`Restore dataset version data-v1` (or run `dvc get . data/raw/dataset.csv --rev data-v1 -o artifacts/restored/x.csv`
and compare `md5sum` with the md5 reported).
