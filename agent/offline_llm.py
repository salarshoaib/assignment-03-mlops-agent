"""Deterministic offline 'LLM' used when no API key is configured and in tests.

It plans tool calls from keywords, executes dependent steps (e.g. best run -> dataset for that run),
and writes an evidence-only report. It NEVER states an operational fact that is not in a tool result.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from agent.config import Settings
from agent.llm import LLMResponse

R = re.compile
_MODEL = R(r"\b(model|models|experiment\w*|mlflow|f1|accuracy|precision|recall|metric\w*|best|runs?)\b")
_DATASET = R(r"\b(dataset|dvc|data version|provenance|lineage)\b")
_REPRO = R(r"\breproduc\w*")
_CHANGED = R(r"\b(changed|change|modified|drift\w*)\b")
_K8S = R(r"\b(kubernetes|k8s|deploy\w*|pods?|replicas?|prediction api|inference|container|service|running)\b")
_DIAG = R(r"\b(why|unavailable|down|not working|broken|diagnos\w*|crash\w*|failing|investigate)\b")
_AIRFLOW = R(r"\b(airflow|dags?|pipeline|workflow|retrain\w*|training)\b")
_TRIGGER = R(r"\b(trigger|start|launch|kick ?off|execute)\b|\brun (the|a|my) (training|pipeline|dag)")
_FAIL = R(r"\b(fail\w*|error\w*|broken)\b")
_COMPARE = R(r"\b(compare|comparison)\b")
_RESTART = R(r"\brestart\b")


def _metric(q: str) -> str:
    for m in ("accuracy", "precision", "recall", "f1"):
        if m in q:
            return m
    return "f1"


class OfflineClient:
    def __init__(self, settings: Settings):
        self.s = settings

    # ------------------------------------------------------------------ protocol
    def complete(self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMResponse:
        idx = max(i for i, m in enumerate(messages) if m["role"] == "user" and isinstance(m["content"], str))
        question = messages[idx]["content"]
        names: dict[str, tuple[str, dict]] = {}
        results: dict[str, dict[str, Any]] = {}
        for m in messages[idx + 1:]:
            if not isinstance(m["content"], list):
                continue
            for b in m["content"]:
                if b["type"] == "tool_use":
                    names[b["id"]] = (b["name"], b["input"])
                elif b["type"] == "tool_result":
                    try:
                        results[names[b["tool_use_id"]][0]] = json.loads(b["content"])
                    except (KeyError, ValueError):
                        pass
        todo = [c for c in self._plan(question.lower(), results) if c[0] not in results]
        if todo:
            base = len(names)
            return LLMResponse([{"type": "tool_use", "id": f"offline_{base + i}", "name": n, "input": a}
                                for i, (n, a) in enumerate(todo)], "tool_use")
        return LLMResponse([{"type": "text", "text": build_report(question, results, self.s)}], "end_turn")

    # ------------------------------------------------------------------ planning
    def _plan(self, q: str, res: dict[str, dict]) -> list[tuple[str, dict]]:
        s = self.s
        trigger = bool(_TRIGGER.search(q)) and bool(_AIRFLOW.search(q))
        wants_model = bool(_MODEL.search(q)) and not (trigger and not re.search(r"best|compare|f1|metric|experiment", q))
        wants_data = bool(_DATASET.search(q))
        wants_k8s = bool(_K8S.search(q))
        wants_af = bool(_AIRFLOW.search(q))
        plan: list[tuple[str, dict]] = []
        if wants_model:
            if _COMPARE.search(q):
                plan.append(("mlflow_list_runs", {"max_results": 5}))
                plan.append(("mlflow_find_best_run", {"metric": _metric(q), "recent_n": 5}))
            else:
                plan.append(("mlflow_find_best_run", {"metric": _metric(q)}))
        if _RESTART.search(q):
            m = re.search(r"([a-z0-9][a-z0-9-]*)\s+namespace|namespace\s+([a-z0-9][a-z0-9-]*)", q)
            ns = next((g for g in (m.groups() if m else ()) if g and g != "the"), None)
            plan.append(("k8s_restart_deployment", {"namespace": ns} if ns else {}))
        if wants_data and not wants_model:
            plan += [("dvc_list_tracked_datasets", {}), ("dvc_get_dataset_version", {})]
        if wants_data and _CHANGED.search(q):
            plan.append(("dvc_detect_changes", {}))
        if wants_data and _REPRO.search(q):
            plan.append(("dvc_list_tracked_datasets", {}))
        if wants_k8s:
            plan += [("k8s_get_deployed_version", {}), ("k8s_find_unhealthy_pods", {})]
            if _DIAG.search(q):
                plan.append(("k8s_diagnose_deployment", {}))
        if wants_af:
            plan.append(("airflow_get_dag_runs", {"dag_id": s.approved_dags[0], "limit": 5}))
        if trigger:
            plan.append(("airflow_trigger_dag", {"dag_id": s.approved_dags[0], "conf": {}}))
        # dependent steps
        best = (res.get("mlflow_find_best_run") or {}).get("data")
        if wants_data and best:
            plan.append(("dvc_get_dataset_for_run", {"run_id": best["best"]["run_id"]}))
        if wants_data and _REPRO.search(q) and best:
            plan.append(("dvc_explain_reproduce", {"run_id": best["best"]["run_id"]}))
        runs = (res.get("mlflow_list_runs") or {}).get("data")
        if runs:
            ids = [r["run_id"] for r in runs["runs"] if r["status"] == "FINISHED"]
            if len(ids) >= 2:
                plan.append(("mlflow_compare_runs", {"run_ids": ids[:5]}))
        af = (res.get("airflow_get_dag_runs") or {}).get("data")
        if af and af["runs"] and (af["runs"][0]["state"] == "failed" or _FAIL.search(q)):
            plan.append(("airflow_get_failed_tasks", {"dag_id": s.approved_dags[0], "dag_run_id": af["runs"][0]["dag_run_id"]}))
        failed = (res.get("airflow_get_failed_tasks") or {}).get("data")
        if failed and failed["failed_tasks"]:
            t = failed["failed_tasks"][0]
            plan.append(("airflow_get_task_logs", {"dag_id": s.approved_dags[0], "dag_run_id": failed["dag_run_id"],
                                                    "task_id": t["task_id"], "try_number": max(1, int(t.get("try_number") or 1))}))
        return plan


# ====================================================================== reporting
def _d(res: dict, tool: str) -> dict | None:
    r = res.get(tool)
    return r["data"] if r and r.get("ok") else None


def assess_retraining(res: dict[str, dict], s: Settings) -> tuple[bool | None, list[str], list[str]]:
    """Return (needed, evidence, considerations). Retraining is recommended ONLY with explicit evidence."""
    evidence: list[str] = []
    notes: list[str] = []
    runs = _d(res, "airflow_get_dag_runs")
    if runs and runs["runs"] and runs["runs"][0]["state"] == "failed":
        ft = _d(res, "airflow_get_failed_tasks")
        where = ", ".join(t["task_id"] for t in ft["failed_tasks"]) if ft else "an unknown task"
        evidence.append(f"The latest training DAG run failed (failed task: {where}).")
    best = _d(res, "mlflow_find_best_run")
    if best and s.model_max_age_days is not None and best["best"].get("start_time"):
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(best["best"]["start_time"])).days
        if age > s.model_max_age_days:
            evidence.append(f"Freshness policy exceeded: model is {age} days old (limit {s.model_max_age_days}).")
    elif best and s.model_max_age_days is None:
        notes.append("No explicit freshness policy is configured (MODEL_MAX_AGE_DAYS), so age alone is not a trigger.")
    if best and best["readiness_checks"].get("meets_min_f1") is False:
        evidence.append("The best recorded model fails the configured f1 quality threshold.")
    prov = _d(res, "dvc_get_dataset_for_run")
    if prov and prov.get("current", {}).get("matches_recorded") is False:
        notes.append("The DVC dataset has changed since this model was trained; that is a reason to evaluate, "
                     "not by itself evidence of degradation.")
    notes.append("No model-degradation results (drift/online metrics) were supplied or checked.")
    if evidence:
        return True, evidence, notes
    if best is None and runs is None:
        return None, evidence, notes
    return False, evidence, notes


def build_report(question: str, res: dict[str, dict], s: Settings) -> str:
    if not res:
        return ("I can query DVC (dataset provenance), MLflow (runs/metrics/registry), Airflow (DAG runs) and "
                "Kubernetes (pods/deployments). Ask e.g. 'Find the best recent model, identify its dataset version, "
                "check the deployment and say whether retraining is necessary.'")
    facts: list[str] = []
    unknown: list[str] = []
    actions: list[str] = []
    pending = []

    for tool, r in res.items():
        if not r.get("ok"):
            et = r.get("error_type")
            if et == "confirmation_required":
                pending.append(r["data"])
            else:
                unknown.append(f"`{tool}` failed ({et}): {r.get('error')}. I cannot report anything from this source.")

    if (b := _d(res, "mlflow_find_best_run")):
        best = b["best"]
        m = best["metrics"]
        facts.append(f"MLflow: best of {b['scored']} recent finished runs by {b['metric']} is "
                     f"`{best['run_name'] or best['run_id'][:8]}` ({b['metric']}={m.get(b['metric'])}"
                     + (f", accuracy={m.get('accuracy')}" if "accuracy" in m and b["metric"] != "accuracy" else "") + ").")
        facts.append("Readiness checks: " + ", ".join(f"{k}={v}" for k, v in b["readiness_checks"].items()))
        unknown += b["caveats"]
    if (rl := _d(res, "mlflow_list_runs")):
        facts.append(f"MLflow: {rl['count']} most recent runs listed.")
    if (c := _d(res, "mlflow_compare_runs")):
        for n in c["metrics_compared"][:4]:
            facts.append(f"Best `{n}` among compared runs: {c['best_per_metric'][n][:8] if c['best_per_metric'][n] else 'n/a'}.")
        if c["differing_params"]:
            facts.append("Parameters that differ: " + ", ".join(c["differing_params"]))
        if c["different_datasets"]:
            unknown.append("Compared runs used different datasets, so metrics are not directly comparable.")
        actions.append("Do not pick solely on the top metric; check dataset, thresholds and registry state.")
    if (p := _d(res, "dvc_get_dataset_for_run")):
        rec = p["recorded"]
        if p.get("provenance_missing"):
            unknown.append(p["note"])
        else:
            rc = p.get("resolved_commit") or {}
            facts.append(f"DVC: that run recorded dataset `{rec['dataset_path']}` md5 `{str(rec['md5'])[:10]}` "
                         f"(version {rec.get('dataset_version') or ', '.join(rc.get('tags', [])) or 'untagged'}, "
                         f"git commit {rc.get('commit_short') or 'not found in history'}).")
            if p.get("current"):
                facts.append("The current tracked dataset " + ("matches" if p["current"]["matches_recorded"] else "DIFFERS from")
                             + " the one used for training.")
    if (ex := _d(res, "dvc_explain_reproduce")):
        facts.append(f"DVC reproduce: expected md5 `{str(ex['expected_md5'])[:10]}` at commit `{ex['commit'][:10]}`. Steps:")
        facts += [f"  `{s.split('#')[0].strip()}`" for s in ex["steps"]]
        facts.append(f"Non-destructive alternative: `{ex['non_destructive_alternative']}`")
    if (v := _d(res, "dvc_get_dataset_version")):
        facts.append(f"DVC: `{v['dataset_path']}` md5 `{str(v['md5'])[:10]}`, commit {v.get('commit_short')}, "
                     f"version tag {v.get('dataset_version')}.")
    if (dc := _d(res, "dvc_detect_changes")):
        facts.append(f"DVC: workspace data changed vs recorded version: {dc.get('changed')}.")
    if (dl := _d(res, "dvc_list_tracked_datasets")) and "dvc_get_dataset_for_run" not in res:
        facts.append(f"DVC: {dl['count']} tracked dataset(s): " + ", ".join(d["dataset_path"] for d in dl["datasets"]))
    if (k := _d(res, "k8s_get_deployed_version")):
        facts.append(f"Kubernetes: `{k['deployment']}` image `{k['image']}` (version label {k['app_version_label']}), "
                     f"{k['ready_replicas']}/{k['desired_replicas']} replicas ready.")
        if k.get("config"):
            facts.append("ConfigMap values: " + ", ".join(f"{a}={b}" for a, b in k["config"].items()))
    if (u := _d(res, "k8s_find_unhealthy_pods")):
        if u["all_healthy"]:
            facts.append(f"Kubernetes: all {u['total_pods']} pods healthy.")
        else:
            for pod in u["unhealthy"]:
                facts.append(f"Kubernetes: pod `{pod['name']}` unhealthy - " + "; ".join(pod["problems"]))
            actions.append("Resolve the deployment problem first; it is an infrastructure issue, not a modelling issue.")
    if (dg := _d(res, "k8s_diagnose_deployment")):
        for f in dg["findings"]:
            facts.append(f"Diagnosis evidence: {f}")
        for pod, lines in dg["error_log_lines"].items():
            for ln in lines[-3:]:
                facts.append(f"Log ({pod}): {ln}")
        if dg["healthy"]:
            facts.append("Diagnosis found no problems in pod status, events or logs.")
        else:
            actions.append("Fix the root cause shown above, then (with your confirmation) I can restart the Deployment.")
    if (a := _d(res, "airflow_get_dag_runs")):
        if not a["runs"]:
            unknown.append("Airflow has no runs recorded for the training DAG.")
        for r in a["runs"][:3]:
            facts.append(f"Airflow: run `{r['dag_run_id']}` is {r['state']} (start {r['start_date']}).")
    if (ft := _d(res, "airflow_get_failed_tasks")):
        for t in ft["failed_tasks"]:
            facts.append(f"Airflow: task `{t['task_id']}` failed (try {t['try_number']}/{t['max_tries']}).")
    if (lg := _d(res, "airflow_get_task_logs")):
        for ln in lg["error_lines"][-3:]:
            facts.append(f"Airflow log ({lg['task_id']}): {ln}")
    for pa in pending:
        actions.append(f"Awaiting your confirmation for `{pa['tool']}` with {pa['arguments']}. "
                       f"Confirm with id `{pa['pending_id']}` (`/confirm {pa['pending_id']}`).")

    out = ["## Known facts (from tool evidence)", *(f"- {x}" for x in facts or ["None - no tool returned usable data."])]
    if any(t in res for t in ("mlflow_find_best_run", "airflow_get_dag_runs", "k8s_get_deployed_version")) and \
            re.search(r"retrain", question.lower()):
        needed, evidence, notes = assess_retraining(res, s)
        out.append("\n## Retraining assessment")
        if needed is None:
            out.append("- Cannot assess: required evidence sources were unavailable.")
        elif needed:
            out += [f"- Evidence supports retraining: {e}" for e in evidence]
            actions.append("Retraining looks justified; I will only start it if you explicitly ask and confirm.")
        else:
            out.append("- No evidence that retraining is necessary (no failed quality gate/run, no policy breach).")
        unknown += notes
    out.append("\n## Uncertainties / not verified")
    out += [f"- {x}" for x in dict.fromkeys(unknown)] or ["- None identified."]
    out.append("\n## Recommended actions")
    out += [f"- {x}" for x in dict.fromkeys(actions)] or ["- None required."]
    return "\n".join(out)
