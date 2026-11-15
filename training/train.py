"""Run the full training pipeline locally (same stages as the Airflow DAG).

python training/train.py --run-name run-1 --n-estimators 100 --max-depth 6
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# MLflow prints a "🏃 View run ..." banner to stdout. On Windows the console defaults to
# cp1252, which cannot encode that emoji, so the process dies AFTER the run is logged.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from training import pipeline_steps as ps  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/dataset.csv")
    ap.add_argument("--work-dir", default="artifacts/local")
    ap.add_argument("--model-out", default="models/model.joblib")
    ap.add_argument("--metrics-out", default="metrics.json")
    ap.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "mlops-agent"))
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-f1", type=float, default=float(os.getenv("MIN_F1", "0.80")))
    ap.add_argument("--min-accuracy", type=float, default=float(os.getenv("MIN_ACCURACY", "0.80")))
    ap.add_argument("--no-gate", action="store_true", help="log to MLflow but skip registry promotion")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    ps.validate_dataset(a.data)
    prep = ps.preprocess(a.data, a.work_dir, seed=a.seed)
    trained = ps.train_model(prep["train_path"], a.model_out, a.n_estimators, a.max_depth, a.seed)
    logged = ps.evaluate_and_log(prep["test_path"], trained["model_path"], trained["params"], a.data,
                                 a.tracking_uri, a.experiment, a.run_name, {"pipeline": "manual"})
    Path(a.metrics_out).write_text(json.dumps(logged["metrics"], indent=2))
    print(json.dumps(logged, indent=2))
    if not a.no_gate:
        try:
            print(json.dumps(ps.quality_gate(logged["run_id"], a.tracking_uri, a.min_f1, a.min_accuracy), indent=2))
        except ps.QualityGateError as exc:
            print(f"QUALITY GATE FAILED: {exc}")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
