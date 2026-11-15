"""Airflow 3 training DAG: validate -> preprocess -> train -> evaluate/log (MLflow) -> quality gate.

Design notes
* retries=2 with exponential back-off for transient failures (training, MLflow I/O).
* Validation and the quality gate raise AirflowFailException => NO retries (deterministic failures),
  and downstream tasks get 'upstream_failed' so nothing is promoted.
* Promotion (registry alias 'candidate') happens ONLY inside the quality-gate task after it passes.
* Trigger conf can inject failures for demos: {"inject_failure": "validation" | "quality" | "train"}.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.exceptions import AirflowFailException
from airflow.sdk import dag, task

try:
    from airflow.sdk import get_current_context
except ImportError:  # pragma: no cover
    from airflow.operators.python import get_current_context

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/airflow/project"))
sys.path.insert(0, str(PROJECT_ROOT))

from training import pipeline_steps as ps  # noqa: E402

log = logging.getLogger("airflow.task")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

DEFAULT_PARAMS = {
    "data_path": "data/raw/dataset.csv", "experiment_name": "mlops-agent", "n_estimators": 200, "max_depth": 8,
    "min_f1": 0.80, "min_accuracy": 0.80, "inject_failure": "none",
}


def _p() -> dict:
    return {**DEFAULT_PARAMS, **get_current_context()["params"]}


def _work_dir() -> Path:
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", get_current_context()["run_id"])
    return PROJECT_ROOT / "artifacts" / "airflow" / run_id


@dag(
    dag_id="training_pipeline", schedule=None, catchup=False, max_active_runs=1,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc), tags=["mlops", "training"], params=DEFAULT_PARAMS,
    default_args={"owner": "mlops", "retries": 2, "retry_delay": timedelta(seconds=20),
                  "retry_exponential_backoff": True, "max_retry_delay": timedelta(minutes=2)},
    doc_md=__doc__,
)
def training_pipeline():
    @task(retries=0)
    def validate_dataset() -> dict:
        p = _p()
        min_rows = 10**9 if p["inject_failure"] == "validation" else 200
        try:
            info = ps.validate_dataset(str(PROJECT_ROOT / p["data_path"]), min_rows=min_rows)
        except ps.DataValidationError as exc:
            log.error("dataset validation failed: %s", exc)
            raise AirflowFailException(f"Dataset validation failed: {exc}") from exc
        log.info("dataset validated: %s", info)
        return {"data_path": str(PROJECT_ROOT / p["data_path"]), **info}

    @task
    def preprocess_data(v: dict) -> dict:
        out = ps.preprocess(v["data_path"], str(_work_dir()))
        log.info("preprocessed: %s", out)
        return {**out, "data_path": v["data_path"]}

    @task
    def train_model(prep: dict) -> dict:
        p = _p()
        if p["inject_failure"] == "train":
            raise RuntimeError("Injected training failure (simulated transient error)")
        out = ps.train_model(prep["train_path"], str(_work_dir() / "model.joblib"),
                             int(p["n_estimators"]), int(p["max_depth"]))
        log.info("trained: %s", out)
        return {**out, "test_path": prep["test_path"], "data_path": prep["data_path"]}

    @task
    def evaluate_and_log(tr: dict) -> dict:
        ctx, p = get_current_context(), _p()
        out = ps.evaluate_and_log(
            tr["test_path"], tr["model_path"], tr["params"], p["data_path"], TRACKING_URI, p["experiment_name"],
            run_name=f"airflow-{ctx['run_id'][-12:]}",
            extra_tags={"pipeline": "airflow", "airflow.dag_run_id": ctx["run_id"]}, repo_root=PROJECT_ROOT)
        log.info("logged to MLflow: %s", out)
        return out

    @task(retries=0)
    def quality_gate(ev: dict) -> dict:
        p = _p()
        min_f1 = 0.999 if p["inject_failure"] == "quality" else float(p["min_f1"])
        try:
            res = ps.quality_gate(ev["run_id"], TRACKING_URI, min_f1, float(p["min_accuracy"]))
        except ps.QualityGateError as exc:
            log.error("%s", exc)
            raise AirflowFailException(str(exc)) from exc
        log.info("quality gate passed, promoted: %s", res)
        return res

    quality_gate(evaluate_and_log(train_model(preprocess_data(validate_dataset()))))


training_pipeline()
