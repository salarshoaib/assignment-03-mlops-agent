# Deliberate failure injections (need >= 2; 4 provided)

| Failure | Inject | Ask the agent | Expected diagnosis |
|---|---|---|---|
| A. Missing model file | `python scripts/failure_k8s.py bad-model-path` | `Why is my prediction API unavailable?` | pods Running but not Ready; Event "Readiness probe failed ... 503"; log `MODEL LOAD FAILED`; ConfigMap MODEL_PATH wrong |
| B. Bad image tag | `python scripts/failure_k8s.py bad-image` | same | ImagePullBackOff finding, image `mlops-inference:9.9.9`; old pods still serving (maxUnavailable 0) |
| C. Quality gate | Airflow UI/agent: trigger with conf `{"inject_failure": "quality"}` | `Why did the last training run fail?` | task `quality_gate` failed (no retry), `evaluate_and_log` succeeded, nothing promoted (no `candidate` alias) |
| D. Validation | trigger with conf `{"inject_failure": "validation"}` | same | `validate_dataset` failed, all downstream `upstream_failed`, no MLflow run |
| E. Transient training error | conf `{"inject_failure": "train"}` | same | `train_model` retried (try 3/3) then failed |
| F. Service outage | `docker compose stop mlflow` | `Find the best model` | tool error `service_unavailable`; answer states it cannot report; no metrics invented |

Restore Kubernetes: `python scripts/failure_k8s.py restore`.
Write for each failure: symptom -> evidence (tool calls) -> root cause -> resolution (put in the technical report).
