"""Task 1 - DVC data management tools.

Only an allow-list of read-only git/dvc sub-commands can ever be executed (no shell, no arbitrary args).
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from agent.config import Settings
from agent.schemas import (
    DvcCompareInput,
    DvcDatasetInput,
    DvcReproduceInput,
    DvcRestoreInput,
    DvcRunInput,
    EmptyInput,
)
from agent.tools.base import PermissionDenied, ServiceUnavailable, Tool, ToolInputError

_SKIP_DIRS = {".git", ".dvc", ".venv", "venv", "node_modules", "artifacts", "mlruns", "__pycache__"}
DVC_ALLOWED = [("status",), ("diff",), ("data", "status"), ("list",), ("get",), ("remote", "list"), ("version",)]
GIT_ALLOWED = {"log", "rev-parse", "describe", "show", "status", "tag", "cat-file"}


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner:
    """Executes approved git/dvc commands without a shell."""

    def __init__(self, cwd: Path, timeout: float = 25.0):
        self.cwd = cwd
        self.timeout = timeout

    @staticmethod
    def _validate(binary: str, args: list[str]) -> None:
        if any(("\n" in a or "\x00" in a) for a in args):
            raise PermissionDenied("Illegal characters in command arguments")
        if binary == "git":
            if not args or args[0] not in GIT_ALLOWED:
                raise PermissionDenied(f"git sub-command not approved: {args[:1]}")
            if args[0] == "tag" and not (len(args) > 1 and args[1] in {"--points-at", "--list", "-l"}):
                raise PermissionDenied("Only read-only 'git tag' usage is approved")
        elif binary == "dvc":
            if not any(tuple(args[: len(p)]) == p for p in DVC_ALLOWED):
                raise PermissionDenied(f"dvc sub-command not approved: {args[:2]}")
        else:
            raise PermissionDenied(f"binary not approved: {binary}")

    def run(self, binary: str, args: list[str], timeout: float | None = None) -> CommandResult:
        self._validate(binary, args)
        try:
            proc = subprocess.run(  # noqa: S603 - allow-listed, shell=False
                [binary, *args], cwd=self.cwd, capture_output=True, text=True,
                timeout=timeout or self.timeout, shell=False, check=False,
            )
        except FileNotFoundError as exc:
            raise ServiceUnavailable(f"'{binary}' executable not found in PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise ServiceUnavailable(f"'{binary}' timed out") from exc
        return CommandResult(proc.returncode, proc.stdout, proc.stderr)


def md5_of_file(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - DVC itself uses md5 content hashes
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DvcService:
    def __init__(self, settings: Settings, runner: CommandRunner | None = None, mlflow_client: Any = None):
        self.settings = settings
        self.root = settings.repo_root
        self.runner = runner or CommandRunner(self.root)
        self._mlflow_client = mlflow_client

    # ------------------------------------------------------------------ helpers
    def _safe_rel(self, dataset_path: str | None) -> str:
        rel = dataset_path or self.settings.dvc_default_dataset
        p = PurePosixPath(rel.replace("\\", "/"))
        if p.is_absolute() or ".." in p.parts:
            raise PermissionDenied("Dataset path must be relative and inside the repository")
        return str(p)

    def _parse_dvc_yaml(self, text: str) -> list[dict[str, Any]]:
        doc = yaml.safe_load(text) or {}
        return [o for o in doc.get("outs", []) if isinstance(o, dict)]

    def _entry_from_dvc_text(self, text: str) -> dict[str, Any]:
        outs = self._parse_dvc_yaml(text)
        if not outs:
            raise ToolInputError("No outs found in .dvc file")
        o = outs[0]
        return {"md5": o.get("md5"), "size": o.get("size"), "nfiles": o.get("nfiles")}

    def list_tracked(self) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for dvc_file in sorted(self.root.rglob("*.dvc")):
            if any(part in _SKIP_DIRS for part in dvc_file.relative_to(self.root).parts):
                continue
            try:
                outs = self._parse_dvc_yaml(dvc_file.read_text())
            except (OSError, yaml.YAMLError):
                continue
            for o in outs:
                path = (dvc_file.parent / o.get("path", "")).relative_to(self.root).as_posix()
                found.append({
                    "dataset_path": path, "dvc_file": dvc_file.relative_to(self.root).as_posix(),
                    "md5": o.get("md5"), "size": o.get("size"), "nfiles": o.get("nfiles"),
                    "tracked_by": "dvc_file",
                })
        lock = self.root / "dvc.lock"
        if lock.exists():
            try:
                stages = (yaml.safe_load(lock.read_text()) or {}).get("stages", {})
            except yaml.YAMLError:
                stages = {}
            for stage, spec in stages.items():
                for o in spec.get("outs", []) or []:
                    found.append({"dataset_path": o.get("path"), "dvc_file": "dvc.lock", "stage": stage,
                                  "md5": o.get("md5"), "size": o.get("size"), "tracked_by": "pipeline"})
        return found

    def _dvc_file_for(self, dataset_path: str) -> str:
        return f"{dataset_path}.dvc"

    def _git(self, *args: str) -> CommandResult:
        return self.runner.run("git", list(args))

    def _git_show(self, rev: str, path: str) -> str:
        res = self._git("show", f"{rev}:{path}")
        if res.returncode != 0:
            raise ToolInputError(f"'{path}' not found at revision '{rev}': {res.stderr.strip()[:200]}")
        return res.stdout

    def _resolve_commit(self, rev: str) -> str:
        res = self._git("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
        if res.returncode != 0:
            raise ToolInputError(f"Unknown git revision '{rev}'")
        return res.stdout.strip()

    def _commit_info(self, dvc_file: str) -> dict[str, Any]:
        res = self._git("log", "-1", "--format=%H%x1f%h%x1f%aI%x1f%s", "--", dvc_file)
        if res.returncode != 0 or not res.stdout.strip():
            return {"commit": None, "note": "No git history for this .dvc file (not committed yet?)"}
        full, short, date, subject = res.stdout.strip().split("\x1f", 3)
        tag = self._git("describe", "--tags", "--match", "data-v*", "--abbrev=0", full)
        tags_here = self._git("tag", "--points-at", full)
        return {
            "commit": full, "commit_short": short, "commit_date": date, "commit_subject": subject,
            "dataset_version": tag.stdout.strip() if tag.returncode == 0 and tag.stdout.strip() else None,
            "tags_on_commit": [t for t in tags_here.stdout.split() if t],
        }

    def find_commit_for_md5(self, dvc_file: str, md5: str, limit: int = 100) -> dict[str, Any] | None:
        res = self._git("log", f"-n{limit}", "--format=%H", "--", dvc_file)
        for commit in res.stdout.split():
            try:
                if self._entry_from_dvc_text(self._git_show(commit, dvc_file)).get("md5") == md5:
                    info = self._git("log", "-1", "--format=%h%x1f%aI%x1f%s", commit).stdout.strip().split("\x1f", 2)
                    tags = self._git("tag", "--points-at", commit).stdout.split()
                    return {"commit": commit, "commit_short": info[0], "date": info[1], "subject": info[2],
                            "tags": tags}
            except ToolInputError:
                continue
        return None

    # ------------------------------------------------------------------ tools
    def list_datasets(self, _: EmptyInput) -> dict[str, Any]:
        datasets = self.list_tracked()
        return {"count": len(datasets), "datasets": datasets, "default_dataset": self.settings.dvc_default_dataset}

    def get_version(self, args: DvcDatasetInput) -> dict[str, Any]:
        path = self._safe_rel(args.dataset_path)
        dvc_file = self._dvc_file_for(path)
        f = self.root / dvc_file
        if not f.exists():
            raise ToolInputError(f"'{dvc_file}' not found - dataset is not tracked by DVC")
        entry = self._entry_from_dvc_text(f.read_text())
        out = {"dataset_path": path, "dvc_file": dvc_file, **entry}
        out.update(self._commit_info(dvc_file))
        remotes = self.runner.run("dvc", ["remote", "list"]) if self._have_dvc() else None
        out["dvc_remotes"] = remotes.stdout.strip().splitlines() if remotes and remotes.returncode == 0 else []
        return out

    def _have_dvc(self) -> bool:
        try:
            return self.runner.run("dvc", ["version"]).returncode == 0
        except ServiceUnavailable:
            return False

    def compare_versions(self, args: DvcCompareInput) -> dict[str, Any]:
        path = self._safe_rel(args.dataset_path)
        dvc_file = self._dvc_file_for(path)
        a_commit, b_commit = self._resolve_commit(args.rev_a), self._resolve_commit(args.rev_b)
        a = self._entry_from_dvc_text(self._git_show(a_commit, dvc_file))
        b = self._entry_from_dvc_text(self._git_show(b_commit, dvc_file))
        out: dict[str, Any] = {
            "dataset_path": path,
            "a": {"rev": args.rev_a, "commit": a_commit, **a},
            "b": {"rev": args.rev_b, "commit": b_commit, **b},
            "changed": a["md5"] != b["md5"],
        }
        if a.get("size") is not None and b.get("size") is not None:
            out["size_delta_bytes"] = b["size"] - a["size"]
        try:
            diff = self.runner.run("dvc", ["diff", a_commit, b_commit, "--json"])
            if diff.returncode == 0 and diff.stdout.strip():
                import json
                out["dvc_diff"] = json.loads(diff.stdout)
        except (ServiceUnavailable, ValueError):
            out["dvc_diff_note"] = "dvc diff unavailable; comparison based on recorded md5 hashes in git history"
        return out

    def detect_changes(self, args: DvcDatasetInput) -> dict[str, Any]:
        path = self._safe_rel(args.dataset_path)
        dvc_file = self._dvc_file_for(path)
        f = self.root / dvc_file
        if not f.exists():
            raise ToolInputError(f"'{dvc_file}' not found - dataset is not tracked by DVC")
        recorded = self._entry_from_dvc_text(f.read_text())["md5"]
        ws = self.root / path
        out: dict[str, Any] = {"dataset_path": path, "recorded_md5": recorded}
        if not ws.exists():
            out.update(workspace_present=False, changed=None,
                       note="Data file is not in the workspace (run `dvc checkout` or `dvc pull`).")
        elif ws.is_file() and not str(recorded).endswith(".dir"):
            current = md5_of_file(ws)
            out.update(workspace_present=True, workspace_md5=current, changed=current != recorded)
        else:
            out.update(workspace_present=True, changed=None, note="Directory outputs need `dvc status`.")
        try:
            import json
            st = self.runner.run("dvc", ["data", "status", "--json"])
            if st.returncode == 0 and st.stdout.strip():
                out["dvc_data_status"] = json.loads(st.stdout)
        except (ServiceUnavailable, ValueError):
            out["dvc_data_status_note"] = "dvc CLI unavailable; used direct md5 comparison"
        return out

    def explain_reproduce(self, args: DvcReproduceInput) -> dict[str, Any]:
        path = self._safe_rel(args.dataset_path)
        dvc_file = self._dvc_file_for(path)
        rev, resolved, recorded_md5 = args.rev, None, None
        if args.run_id and not rev:
            prov = self.dataset_for_run(DvcRunInput(run_id=args.run_id))
            rev = (prov.get("resolved_commit") or {}).get("commit")
            path = prov["recorded"]["dataset_path"] or path
            dvc_file = self._dvc_file_for(path)
            recorded_md5 = prov["recorded"]["md5"]
            if not rev:
                raise ToolInputError("Could not resolve a git commit that contains the recorded dataset hash")
        if not rev:
            raise ToolInputError("Provide 'rev' (git tag/commit) or 'run_id'")
        resolved = self._resolve_commit(rev)
        entry = self._entry_from_dvc_text(self._git_show(resolved, dvc_file))
        steps = [
            f"git checkout {resolved[:10]} -- {dvc_file}      # restore the pointer file for that version",
            f"dvc checkout {path}                            # link data from the local DVC cache",
            f"dvc pull {path}                                # if not cached locally, fetch from the remote",
            f"md5sum {path}                                  # must equal {entry['md5']}",
            f"git checkout HEAD -- {dvc_file} && dvc checkout {path}   # return to the latest version",
        ]
        return {
            "dataset_path": path, "revision": rev, "commit": resolved, "expected_md5": entry["md5"],
            "recorded_md5_matches": (recorded_md5 == entry["md5"]) if recorded_md5 else None,
            "steps": steps,
            "non_destructive_alternative": f"dvc get . {path} --rev {resolved[:10]} -o artifacts/restored/{PurePosixPath(path).name}",
        }

    def dataset_for_run(self, args: DvcRunInput) -> dict[str, Any]:
        """Read the DVC provenance recorded by training on an MLflow run and verify it against git history."""
        client = self._client()
        run = None
        if args.run_id:
            try:
                run = client.get_run(args.run_id)
            except Exception as exc:  # noqa: BLE001
                raise ToolInputError(f"MLflow run '{args.run_id}' not found: {exc}") from exc
        elif args.run_name:
            safe = args.run_name.replace("'", "")
            exps = [e.experiment_id for e in client.search_experiments()]
            runs = client.search_runs(exps, filter_string=f"tags.mlflow.runName = '{safe}'", max_results=1,
                                      order_by=["attributes.start_time DESC"])
            run = runs[0] if runs else None
            if run is None:
                raise ToolInputError(f"No MLflow run named '{args.run_name}'")
        else:
            raise ToolInputError("Provide run_id or run_name")
        tags = run.data.tags
        recorded = {
            "dataset_path": tags.get("dvc.dataset_path"), "md5": tags.get("dvc.md5"),
            "dataset_version": tags.get("dvc.dataset_version"), "git_commit": tags.get("git.commit"),
        }
        out: dict[str, Any] = {"run_id": run.info.run_id, "run_name": tags.get("mlflow.runName"),
                               "recorded": recorded}
        if not recorded["md5"] or not recorded["dataset_path"]:
            out["provenance_missing"] = True
            out["note"] = "Run has no DVC provenance tags; dataset version cannot be determined."
            return out
        dvc_file = self._dvc_file_for(recorded["dataset_path"])
        out["resolved_commit"] = self.find_commit_for_md5(dvc_file, recorded["md5"])
        f = self.root / dvc_file
        if f.exists():
            current = self._entry_from_dvc_text(f.read_text())["md5"]
            out["current"] = {"md5": current, "matches_recorded": current == recorded["md5"]}
        return out

    def restore_dataset(self, args: DvcRestoreInput) -> dict[str, Any]:
        """Non-destructive reproduction: export a dataset version into artifacts/restored/<rev>/."""
        path = self._safe_rel(args.dataset_path)
        commit = self._resolve_commit(args.rev)
        out_dir = self.root / "artifacts" / "restored" / commit[:10]
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / PurePosixPath(path).name
        res = self.runner.run("dvc", ["get", ".", path, "--rev", commit, "-o", str(target)], timeout=120)
        if res.returncode != 0:
            raise ServiceUnavailable(f"dvc get failed: {res.stderr.strip()[:300]}")
        expected = self._entry_from_dvc_text(self._git_show(commit, self._dvc_file_for(path)))["md5"]
        actual = md5_of_file(target) if target.is_file() else None
        return {"restored_to": str(target.relative_to(self.root)), "commit": commit, "expected_md5": expected,
                "actual_md5": actual, "verified": actual == expected}

    def _client(self) -> Any:
        if self._mlflow_client is None:
            from mlflow.tracking import MlflowClient
            self._mlflow_client = MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri)
        return self._mlflow_client


def build_tools(settings: Settings, runner: CommandRunner | None = None, mlflow_client: Any = None) -> list[Tool]:
    svc = DvcService(settings, runner, mlflow_client)
    return [
        Tool("dvc_list_tracked_datasets", "List datasets tracked by DVC (.dvc files and pipeline outputs) with md5 hashes.",
             "dvc", EmptyInput, svc.list_datasets),
        Tool("dvc_get_dataset_version", "Get the current DVC-tracked version of a dataset: md5, git commit, version tag, remotes.",
             "dvc", DvcDatasetInput, svc.get_version),
        Tool("dvc_compare_versions", "Compare two dataset versions (git revisions or tags such as data-v1 vs data-v2).",
             "dvc", DvcCompareInput, svc.compare_versions),
        Tool("dvc_detect_changes", "Detect whether the workspace data differs from the version recorded by DVC.",
             "dvc", DvcDatasetInput, svc.detect_changes),
        Tool("dvc_explain_reproduce", "Explain exactly how to reproduce a previous dataset version, by git revision or MLflow run_id.",
             "dvc", DvcReproduceInput, svc.explain_reproduce),
        Tool("dvc_get_dataset_for_run", "Return the DVC dataset version/hash/commit recorded for an MLflow run (by run_id or run_name).",
             "dvc", DvcRunInput, svc.dataset_for_run),
        Tool("dvc_restore_dataset_version", "Reproduce a past dataset version into artifacts/restored/ (non-destructive) and verify its md5.",
             "dvc", DvcRestoreInput, svc.restore_dataset, timeout_s=120),
    ]
