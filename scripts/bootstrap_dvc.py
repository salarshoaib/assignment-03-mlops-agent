"""Create dataset version N under DVC + git (tag data-vN) and push it to a local DVC remote.

  python scripts/bootstrap_dvc.py --version 1
  ... train some runs ...
  python scripts/bootstrap_dvc.py --version 2     # new data (more rows, different seed)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {  # version -> generator args
    1: ["--rows", "2000", "--seed", "1", "--class-sep", "1.0", "--flip-y", "0.03"],
    2: ["--rows", "3000", "--seed", "2", "--class-sep", "0.9", "--flip-y", "0.05"],
    3: ["--rows", "3500", "--seed", "3", "--class-sep", "0.8", "--flip-y", "0.08"],
}


def run(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True, choices=sorted(PROFILES))
    ap.add_argument("--remote", default=str(ROOT.parent / "dvc-remote"))
    a = ap.parse_args()

    if not (ROOT / ".git").exists():
        run("git", "init", "-b", "main")
    if not (ROOT / ".dvc").exists():
        run("dvc", "init")
        run("dvc", "remote", "add", "-d", "localremote", a.remote)
        run("git", "add", ".dvc", ".dvcignore")
        run("git", "commit", "-m", "Initialise DVC", check=False)

    run(sys.executable, "training/generate_data.py", "--out", "data/raw/dataset.csv", *PROFILES[a.version])
    run("dvc", "add", "data/raw/dataset.csv")
    run("dvc", "push")
    run("git", "add", "data/raw/dataset.csv.dvc", "data/raw/.gitignore")
    run("git", "commit", "-m", f"data: dataset version {a.version}", check=False)
    run("git", "tag", "-f", f"data-v{a.version}")
    print(f"\nDataset version data-v{a.version} created and pushed to {a.remote}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
