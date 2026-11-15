# Architecture

```
            +-------------------+
 user ----> | FastAPI (agent)   |  validation (pydantic, extra=forbid), bearer token, X-Request-ID
            +---------+---------+
                      v
            +-------------------+      LLM (Anthropic tool use)  or  OfflineClient (deterministic)
            | Orchestrator      | <--> system prompt: evidence-only, no auto-retrain, 3-section answers
            +---------+---------+
                      v
            +-------------------+  1 schema validation  2 precheck (namespace / DAG / conf allow-list)
            | ToolRegistry      |  3 confirmation gate (mutating)  4 timeout  5 redaction  6 JSON log
            +--+-----+-----+--+-+
               v     v     v  v
             DVC  MLflow Airflow Kubernetes
```

## Design decisions
* **Provenance link:** training logs `dvc.md5`, `dvc.dataset_path`, `dvc.dataset_version`, `git.commit` as MLflow tags.
  `dvc_get_dataset_for_run` reads them back and searches git history of the `.dvc` file for the commit holding that md5,
  so "which dataset version trained run X" is answered from recorded provenance, not memory.
* **Best != production-ready:** `mlflow_find_best_run` returns readiness checks (finished, thresholds, provenance,
  registry/alias) and caveats (noise, no online metrics).
* **Pipeline safety:** validation and quality gate raise `AirflowFailException` (no retries, downstream `upstream_failed`);
  registry alias `candidate` is set only by the gate after it passes. Other tasks use retries=2 with exponential back-off.
* **Confirmation outside the LLM:** mutating tools create a pending action; only `Orchestrator.confirm` / `POST /confirm`
  (human path) executes it. Authorisation (namespace/DAG) is checked before a pending action is even created.
* **Retraining policy:** recommended only with failed DAG run/quality gate, breached freshness policy
  (`MODEL_MAX_AGE_DAYS`) or sub-threshold metrics; dataset change is reported as a consideration, not a trigger.
* **Offline planner:** same registry and rules; lets CI and demos run without an API key and makes the no-invention
  guarantee testable.
* **Known limitations:** in-memory sessions/pending actions (single process); timed-out tool threads are not killed;
  Airflow logs depend on API availability; no drift/online metrics; DVC remote is local.
