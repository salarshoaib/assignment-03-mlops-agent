"""Generate a synthetic tabular dataset (10 numeric features + binary target). Used to create DVC versions."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification


def generate(rows: int, seed: int, class_sep: float, flip_y: float) -> pd.DataFrame:
    X, y = make_classification(n_samples=rows, n_features=10, n_informative=6, n_redundant=2,
                               class_sep=class_sep, flip_y=flip_y, random_state=seed)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    df["target"] = y
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/dataset.csv")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--class-sep", type=float, default=1.0)
    ap.add_argument("--flip-y", type=float, default=0.03)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    generate(a.rows, a.seed, a.class_sep, a.flip_y).to_csv(a.out, index=False)
    print(f"wrote {a.rows} rows to {a.out}")


if __name__ == "__main__":
    main()
