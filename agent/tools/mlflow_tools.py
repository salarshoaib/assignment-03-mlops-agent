"""Task 2 - MLflow experiment intelligence tools (uses MlflowClient, never scrapes the UI)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent.config import Settings
from agent.schemas import (
    MlBestRunInput,
    MlCompareExperimentsInput,
    MlCompareRunsInput,
    MlListExperimentsInput,
    MlListRunsInput,
    MlRegistryInput,
    MlRunInput,
)
from agent.tools.base import ServiceUnavailable, Tool, ToolInputError

PROVENANCE_TAGS = ("dvc.dataset_path", "dvc.md5", "dvc.dataset_version", "git.commit", "pipeline", "airflow.dag_run_id")


def _iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat() if ms else None


class MlflowService:
    def __init__(self, settings: Settings, client: Any = None):
        self.settings = settings
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from mlflow.tracking import MlflowClient
            self._client = MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri)
        return self._client

    def _call(self, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except ToolInputError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ServiceUnavailable(f"MLflow request failed: {type(exc).__name__}: {str(exc)[:200]}") from exc

    def _experiment_id(self, name: str | None) -> str:
        name = name or self.settings.mlflow_experiment
        exp = self._call(self.client.get_experiment_by_name, name)
        if exp is None:
            raise ToolInputError(f"Experiment '{name}' not found")
        return exp.experiment_id

    @staticmethod
    def _summ(run: Any) -> dict[str, Any]:
        tags = run.data.tags
        return {
            "run_id": run.info.run_id, "run_name": tags.get("mlflow.runName"), "status": run.info.status,
            "start_time": _iso(run.info.start_time), "end_time": _iso(run.info.end_time),
            "metrics": dict(run.data.metrics), "params": dict(run.data.params),
            "provenance": {k: tags.get(k) for k in PROVENANCE_TAGS if tags.get(k) is not None},
        }

    # ------------------------------------------------------------------ tools
    def list_experiments(self, args: MlListExperimentsInput) -> dict[str, Any]:
        exps = self._call(self.client.search_experiments, max_results=args.max_results)
        return {"count": len(exps), "experiments": [
            {"experiment_id": e.experiment_id, "name": e.name, "lifecycle_stage": e.lifecycle_stage} for e in exps]}

    def list_runs(self, args: MlListRunsInput) -> dict[str, Any]:
        eid = self._experiment_id(args.experiment_name)
        runs = self._call(self.client.search_runs, [eid], max_results=args.max_results,
                          order_by=["attributes.start_time DESC"])
        return {"experiment": args.experiment_name or self.settings.mlflow_experiment,
                "count": len(runs), "runs": [self._summ(r) for r in runs]}

    def get_run(self, args: MlRunInput) -> dict[str, Any]:
        try:
            run = self.client.get_run(args.run_id)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "not found" in msg.lower() or "RESOURCE_DOES_NOT_EXIST" in msg:
                raise ToolInputError(f"Run '{args.run_id}' not found") from exc
            raise ServiceUnavailable(f"MLflow request failed: {type(exc).__name__}: {msg[:200]}") from exc
        out = self._summ(run)
        try:
            out["artifacts"] = [{"path": a.path, "is_dir": a.is_dir, "size": a.file_size}
                                for a in self.client.list_artifacts(args.run_id)]
        except Exception as exc:  # noqa: BLE001
            out["artifacts"] = None
            out["artifacts_error"] = f"{type(exc).__name__}: {str(exc)[:150]}"
        out["tags"] = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.") or k == "mlflow.runName"}
        return out

    def best_run(self, args: MlBestRunInput) -> dict[str, Any]:
        eid = self._experiment_id(args.experiment_name)
        runs = self._call(self.client.search_runs, [eid], max_results=args.recent_n,
                          order_by=["attributes.start_time DESC"])
        scored = [r for r in runs if r.info.status == "FINISHED" and args.metric in r.data.metrics]
        skipped = [{"run_id": r.info.run_id, "status": r.info.status, "reason":
                    "not FINISHED" if r.info.status != "FINISHED" else f"metric '{args.metric}' not logged"}
                   for r in runs if r not in scored]
        if not scored:
            raise ToolInputError(f"No FINISHED runs with metric '{args.metric}' among the last {len(runs)} runs")
        reverse = args.mode == "max"
        scored.sort(key=lambda r: r.data.metrics[args.metric], reverse=reverse)
        best = scored[0]
        summary = self._summ(best)
        versions = self._versions_for_run(best.info.run_id)
        checks, caveats = self._readiness(best, versions)
        ranking = [{"run_id": r.info.run_id, "run_name": r.data.tags.get("mlflow.runName"),
                    args.metric: r.data.metrics[args.metric], "start_time": _iso(r.info.start_time)} for r in scored]
        if len(scored) > 1:
            gap = abs(ranking[0][args.metric] - ranking[1][args.metric])
            if gap <= 0.01:
                caveats.append(f"Top-2 {args.metric} values differ by only {gap:.4f}; the ranking may be within noise.")
        return {"experiment": args.experiment_name or self.settings.mlflow_experiment, "metric": args.metric,
                "mode": args.mode, "considered": len(runs), "scored": len(scored), "best": summary,
                "ranking": ranking, "skipped": skipped, "registry_versions": versions,
                "readiness_checks": checks, "caveats": caveats,
                "interpretation": "Best by the chosen metric among recent runs. A top metric alone does NOT make a model "
                                  "production-ready; review readiness_checks and caveats."}

    def _versions_for_run(self, run_id: str) -> list[dict[str, Any]]:
        try:
            mvs = self.client.search_model_versions(f"run_id='{run_id}'")
        except Exception:  # noqa: BLE001 - registry is optional
            return []
        return [{"name": mv.name, "version": mv.version, "aliases": list(getattr(mv, "aliases", []) or []),
                 "status": getattr(mv, "status", None)} for mv in mvs]

    def _readiness(self, run: Any, versions: list[dict]) -> tuple[dict[str, Any], list[str]]:
        m, tags = run.data.metrics, run.data.tags
        checks = {
            "run_finished": run.info.status == "FINISHED",
            "meets_min_f1": (m.get("f1") >= self.settings.min_f1) if "f1" in m else None,
            "meets_min_accuracy": (m.get("accuracy") >= self.settings.min_accuracy) if "accuracy" in m else None,
            "has_dataset_provenance": bool(tags.get("dvc.md5")),
            "registered_in_model_registry": bool(versions),
            "has_candidate_or_champion_alias": any(
                a in ("candidate", "champion") for v in versions for a in v["aliases"]),
        }
        caveats: list[str] = []
        if checks["meets_min_f1"] is False:
            caveats.append(f"f1 below configured threshold {self.settings.min_f1}")
        if checks["meets_min_accuracy"] is False:
            caveats.append(f"accuracy below configured threshold {self.settings.min_accuracy}")
        if not checks["has_dataset_provenance"]:
            caveats.append("No DVC provenance tags: training data cannot be traced")
        if not checks["registered_in_model_registry"]:
            caveats.append("Run is not registered in the model registry")
        caveats.append("Only offline test-set metrics are recorded; no online/shadow evaluation or drift data was checked.")
        return checks, caveats

    def compare_runs(self, args: MlCompareRunsInput) -> dict[str, Any]:
        runs = []
        for rid in args.run_ids:
            try:
                runs.append(self.client.get_run(rid))
            except Exception as exc:  # noqa: BLE001
                raise ToolInputError(f"Run '{rid}' not found or unreachable: {str(exc)[:120]}") from exc
        names = args.metrics or sorted({k for r in runs for k in r.data.metrics})
        rows = [{"run_id": r.info.run_id, "run_name": r.data.tags.get("mlflow.runName"), "status": r.info.status,
                 "metrics": {n: r.data.metrics.get(n) for n in names},
                 "dataset_md5": r.data.tags.get("dvc.md5")} for r in runs]
        keys = sorted({k for r in runs for k in r.data.params})
        differing = {k: [r.data.params.get(k) for r in runs] for k in keys
                     if len({r.data.params.get(k) for r in runs}) > 1}
        best = {n: max((row for row in rows if row["metrics"][n] is not None),
                       key=lambda row: row["metrics"][n], default=None) for n in names}
        return {"metrics_compared": names, "runs": rows, "differing_params": differing,
                "best_per_metric": {n: (b["run_id"] if b else None) for n, b in best.items()},
                "different_datasets": len({r["dataset_md5"] for r in rows if r["dataset_md5"]}) > 1}

    def compare_experiments(self, args: MlCompareExperimentsInput) -> dict[str, Any]:
        summary = []
        for name in args.experiment_names:
            eid = self._experiment_id(name)
            runs = self._call(self.client.search_runs, [eid], max_results=100)
            vals = [r.data.metrics[args.metric] for r in runs
                    if r.info.status == "FINISHED" and args.metric in r.data.metrics]
            summary.append({"experiment": name, "runs_with_metric": len(vals),
                            "best": (max(vals) if args.mode == "max" else min(vals)) if vals else None,
                            "mean": round(sum(vals) / len(vals), 4) if vals else None})
        return {"metric": args.metric, "mode": args.mode, "experiments": summary}

    def registry(self, args: MlRegistryInput) -> dict[str, Any]:
        try:
            models = ([self.client.get_registered_model(args.model_name)] if args.model_name
                      else self.client.search_registered_models())
        except Exception as exc:  # noqa: BLE001
            if args.model_name and "not found" in str(exc).lower():
                raise ToolInputError(f"Registered model '{args.model_name}' not found") from exc
            raise ServiceUnavailable(f"Registry unavailable: {str(exc)[:200]}") from exc
        out = []
        for m in models:
            versions = self.client.search_model_versions(f"name='{m.name}'")
            out.append({"name": m.name, "aliases": dict(getattr(m, "aliases", {}) or {}), "versions": [
                {"version": v.version, "run_id": v.run_id, "status": v.status,
                 "aliases": list(getattr(v, "aliases", []) or []), "created": _iso(v.creation_timestamp)}
                for v in sorted(versions, key=lambda v: int(v.version), reverse=True)]})
        return {"count": len(out), "models": out}

    def list_artifacts(self, args: MlRunInput) -> dict[str, Any]:
        arts = self._call(self.client.list_artifacts, args.run_id)
        return {"run_id": args.run_id, "artifacts": [{"path": a.path, "is_dir": a.is_dir, "size": a.file_size}
                                                      for a in arts]}


def build_tools(settings: Settings, client: Any = None) -> list[Tool]:
    s = MlflowService(settings, client)
    return [
        Tool("mlflow_list_experiments", "List MLflow experiments.", "mlflow", MlListExperimentsInput, s.list_experiments),
        Tool("mlflow_list_runs", "List the most recent training runs of an experiment with metrics, params and provenance tags.",
             "mlflow", MlListRunsInput, s.list_runs),
        Tool("mlflow_get_run_details", "Get metrics, parameters, tags and artifacts for one run.", "mlflow", MlRunInput, s.get_run),
        Tool("mlflow_find_best_run", "Find the best recent run by a metric (default f1) with readiness checks and caveats. "
             "Best metric != production-ready.", "mlflow", MlBestRunInput, s.best_run),
        Tool("mlflow_compare_runs", "Compare 2-10 runs: metrics, differing parameters, dataset hashes.", "mlflow",
             MlCompareRunsInput, s.compare_runs),
        Tool("mlflow_compare_experiments", "Compare experiments by best/mean value of a metric.", "mlflow",
             MlCompareExperimentsInput, s.compare_experiments),
        Tool("mlflow_get_model_registry", "Get registered models, versions and aliases (candidate/champion).", "mlflow",
             MlRegistryInput, s.registry),
        Tool("mlflow_list_run_artifacts", "List artifacts stored for a run.", "mlflow", MlRunInput, s.list_artifacts),
    ]

