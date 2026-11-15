"""Pure-Python pipeline stages shared by the CLI (train.py) and the Airflow DAG.

validate -> preprocess -> train -> evaluate_and_log -> quality_gate
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import yaml

log = logging.getLogger("training")
FEATURES = [f"f{i}" for i in range(10)]
TARGET = "target"
MODEL_NAME = "ml-inference-model"
# NOTE: pandas / scikit-learn are imported inside the functions that need them, not at
# module level. The Airflow DAG imports this module while parsing, and DagBag has a hard
# 30s import timeout; pandas+sklearn alone take ~50s to import in the Airflow image.


class DataValidationError(Exception):
    """Dataset failed validation - not retryable."""


class QualityGateError(Exception):
    """Model failed the quality gate - promotion must not happen."""


# ----------------------------------------------------------------------- provenance
def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_provenance(data_path: str, repo_root: str | Path = ".") -> dict[str, str]:
    """DVC + git provenance recorded as MLflow tags (what the agent later reads back)."""
    root = Path(repo_root).resolve()
    p = Path(data_path)
    full = p if p.is_absolute() else root / p
    dvc_file = Path(str(full) + ".dvc")
    md5 = None
    if dvc_file.exists():
        outs = (yaml.safe_load(dvc_file.read_text()) or {}).get("outs", [])
        md5 = outs[0].get("md5") if outs else None
    if md5 is None and full.exists():
        md5 = _md5(full)
    rel = str(p.relative_to(root)) if p.is_absolute() and root in p.parents else str(p)
    tags = {"dvc.dataset_path": rel.replace("\\", "/"), "dvc.md5": md5 or "unknown"}
    commit = _git(root, "rev-parse", "HEAD")
    if commit:
        tags["git.commit"] = commit
    version = _git(root, "describe", "--tags", "--match", "data-v*", "--abbrev=0")
    if version:
        tags["dvc.dataset_version"] = version
    return tags


# ----------------------------------------------------------------------- stages
def validate_dataset(path: str, min_rows: int = 200, max_null_fraction: float = 0.01,
                     min_class_fraction: float = 0.1) -> dict[str, Any]:
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise DataValidationError(f"dataset not found: {path}")
    df = pd.read_csv(p)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    if len(df) < min_rows:
        raise DataValidationError(f"too few rows: {len(df)} < {min_rows}")
    null_frac = float(df[FEATURES + [TARGET]].isna().mean().max())
    if null_frac > max_null_fraction:
        raise DataValidationError(f"null fraction {null_frac:.3f} exceeds {max_null_fraction}")
    class_frac = float(df[TARGET].value_counts(normalize=True).min())
    if class_frac < min_class_fraction:
        raise DataValidationError(f"minority class fraction {class_frac:.3f} < {min_class_fraction}")
    log.info("validation passed: %d rows", len(df))
    return {"rows": len(df), "null_fraction": null_frac, "minority_class_fraction": class_frac}


def preprocess(path: str, out_dir: str, test_size: float = 0.2, seed: int = 42) -> dict[str, Any]:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path).dropna().drop_duplicates()
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[TARGET])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    return {"train_path": str(out / "train.csv"), "test_path": str(out / "test.csv"),
            "n_train": len(train), "n_test": len(test)}


def train_model(train_path: str, model_out: str, n_estimators: int = 200, max_depth: int = 8,
                seed: int = 42) -> dict[str, Any]:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    df = pd.read_csv(train_path)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1)
    t0 = time.perf_counter()
    model.fit(df[FEATURES], df[TARGET])
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    return {"model_path": model_out, "train_seconds": round(time.perf_counter() - t0, 3),
            "params": {"n_estimators": n_estimators, "max_depth": max_depth, "seed": seed}}


def evaluate_and_log(test_path: str, model_path: str, params: dict[str, Any], data_path: str,
                     tracking_uri: str, experiment: str, run_name: str | None = None,
                     extra_tags: dict[str, str] | None = None, repo_root: str | Path = ".") -> dict[str, Any]:
    import mlflow
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    df = pd.read_csv(test_path)
    model = joblib.load(model_path)
    pred = model.predict(df[FEATURES])
    metrics = {"accuracy": accuracy_score(df[TARGET], pred), "f1": f1_score(df[TARGET], pred),
               "precision": precision_score(df[TARGET], pred), "recall": recall_score(df[TARGET], pred)}
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.set_tags({**collect_provenance(data_path, repo_root), **(extra_tags or {})})
        mlflow.log_artifact(model_path, artifact_path="model")
        log.info("logged run %s metrics=%s", run.info.run_id, json.dumps(metrics))
        return {"run_id": run.info.run_id, "metrics": {k: float(v) for k, v in metrics.items()}}


def quality_gate(run_id: str, tracking_uri: str, min_f1: float, min_accuracy: float,
                 model_name: str = MODEL_NAME, alias: str = "candidate") -> dict[str, Any]:
    """Block promotion unless metrics pass. On pass: register the model and set the 'candidate' alias."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    m = client.get_run(run_id).data.metrics
    failures = []
    if m.get("f1", 0.0) < min_f1:
        failures.append(f"f1 {m.get('f1', 0.0):.4f} < {min_f1}")
    if m.get("accuracy", 0.0) < min_accuracy:
        failures.append(f"accuracy {m.get('accuracy', 0.0):.4f} < {min_accuracy}")
    if failures:
        client.set_tag(run_id, "quality_gate", "failed")
        raise QualityGateError("quality gate FAILED: " + "; ".join(failures))
    client.set_tag(run_id, "quality_gate", "passed")
    try:
        client.create_registered_model(model_name)
    except Exception:  # noqa: BLE001 - already exists
        pass
    mv = client.create_model_version(model_name, source=f"{client.get_run(run_id).info.artifact_uri}/model",
                                     run_id=run_id)
    client.set_registered_model_alias(model_name, alias, mv.version)
    return {"promoted": True, "model_name": model_name, "version": mv.version, "alias": alias}
