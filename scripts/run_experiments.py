"""Run a small grid of training runs (each logged to MLflow with DVC provenance tags).

  python scripts/run_experiments.py --count 5
Run names continue from the existing count: run-1, run-2, ...
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRID = list(itertools.product([50, 100, 200, 300], [3, 6, 10]))

# MLflow prints a "🏃 View run ..." banner; cp1252 (Windows console default) cannot encode it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def existing_runs(uri: str, experiment: str) -> int:
    from mlflow.tracking import MlflowClient

    c = MlflowClient(tracking_uri=uri)
    exp = c.get_experiment_by_name(experiment)
    return len(c.search_runs([exp.experiment_id], max_results=1000)) if exp else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    ap.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "mlops-agent"))
    a = ap.parse_args()
    start = existing_runs(a.tracking_uri, a.experiment)
    for i in range(a.count):
        n_est, depth = GRID[(start + i) % len(GRID)]
        name = f"run-{start + i + 1}"
        subprocess.run([sys.executable, "training/train.py", "--run-name", name, "--n-estimators", str(n_est),
                        "--max-depth", str(depth), "--tracking-uri", a.tracking_uri, "--experiment", a.experiment],
                       cwd=ROOT, check=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
