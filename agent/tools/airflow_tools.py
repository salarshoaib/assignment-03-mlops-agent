"""Task 3 - Apache Airflow 3 workflow tools (REST API v2 with JWT authentication)."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests

from agent.config import Settings
from agent.logging_utils import redact
from agent.schemas import (
    AfDagRunsInput,
    AfListDagsInput,
    AfRunRefInput,
    AfTaskLogsInput,
    AfTriggerInput,
    AfWaitInput,
)
from agent.tools.base import PermissionDenied, ServiceUnavailable, Tool, ToolInputError

TERMINAL = {"success", "failed"}


class AirflowClient:
    """Thin authenticated client for the Airflow 3 REST API (/api/v2)."""

    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.s = settings
        self.session = session or requests.Session()
        self._token: str | None = None

    def _authenticate(self) -> str:
        url = f"{self.s.airflow_base_url}/auth/token"
        try:
            resp = self.session.post(url, json={"username": self.s.airflow_username,
                                                "password": self.s.airflow_password}, timeout=10)
            if resp.status_code >= 400:  # SimpleAuthManager with ALL_ADMINS accepts GET without credentials
                resp = self.session.get(url, timeout=10)
        except requests.RequestException as exc:
            raise ServiceUnavailable(f"Airflow unreachable at {self.s.airflow_base_url}: {type(exc).__name__}") from exc
        if resp.status_code >= 400:
            raise ServiceUnavailable(f"Airflow authentication failed (HTTP {resp.status_code})")
        token = resp.json().get("access_token")
        if not token:
            raise ServiceUnavailable("Airflow auth response contained no access_token")
        self._token = token
        return token

    def request(self, method: str, path: str, *, params: dict | None = None, json: Any = None,
                accept: str = "application/json", timeout: float = 15) -> Any:
        url = f"{self.s.airflow_base_url}/api/v2{path}"
        for attempt in (0, 1):
            token = self._token or self._authenticate()
            try:
                resp = self.session.request(method, url, params=params, json=json, timeout=timeout,
                                            headers={"Authorization": f"Bearer {token}", "Accept": accept})
            except requests.RequestException as exc:
                raise ServiceUnavailable(f"Airflow unreachable: {type(exc).__name__}") from exc
            if resp.status_code == 401 and attempt == 0:
                self._token = None
                continue
            break
        if resp.status_code == 404:
            raise ToolInputError(f"Airflow resource not found: {path}")
        if resp.status_code in (401, 403):
            raise ServiceUnavailable(f"Airflow denied the request (HTTP {resp.status_code})")
        if resp.status_code == 422 or resp.status_code == 400:
            raise ToolInputError(f"Airflow rejected the request: {redact(resp.text[:300])}")
        if resp.status_code >= 500:
            raise ServiceUnavailable(f"Airflow server error (HTTP {resp.status_code})")
        return resp.json() if resp.content else {}


class AirflowService:
    def __init__(self, settings: Settings, client: AirflowClient | None = None):
        self.settings = settings
        self.client = client or AirflowClient(settings)

    # ------------------------------------------------------------------ helpers
    def _check_dag(self, dag_id: str) -> None:
        if dag_id not in self.settings.approved_dags:
            raise PermissionDenied(f"DAG '{dag_id}' is not in the approved list {list(self.settings.approved_dags)}")

    def _latest_run_id(self, dag_id: str) -> str:
        runs = self.client.request("GET", f"/dags/{dag_id}/dagRuns", params={"limit": 1, "order_by": "-run_after"})
        items = runs.get("dag_runs", [])
        if not items:
            raise ToolInputError(f"DAG '{dag_id}' has no runs")
        return items[0]["dag_run_id"]

    @staticmethod
    def _run(r: dict) -> dict[str, Any]:
        return {k: r.get(k) for k in ("dag_run_id", "state", "run_type", "logical_date", "start_date", "end_date", "conf")}

    @staticmethod
    def _ti(t: dict) -> dict[str, Any]:
        return {k: t.get(k) for k in ("task_id", "state", "try_number", "max_tries", "start_date", "end_date",
                                       "duration", "operator", "hostname")}

    # ------------------------------------------------------------------ tools
    def list_dags(self, args: AfListDagsInput) -> dict[str, Any]:
        data = self.client.request("GET", "/dags", params={"limit": args.limit})
        dags = [{"dag_id": d["dag_id"], "is_paused": d.get("is_paused"), "tags": [t.get("name") for t in d.get("tags", [])],
                 "approved_for_agent": d["dag_id"] in self.settings.approved_dags,
                 "next_run": d.get("next_dagrun_logical_date")} for d in data.get("dags", [])]
        return {"total": data.get("total_entries", len(dags)), "dags": dags}

    def dag_runs(self, args: AfDagRunsInput) -> dict[str, Any]:
        if args.dag_run_id:
            run = self.client.request("GET", f"/dags/{args.dag_id}/dagRuns/{args.dag_run_id}")
            return {"dag_id": args.dag_id, "runs": [self._run(run)]}
        data = self.client.request("GET", f"/dags/{args.dag_id}/dagRuns",
                                   params={"limit": args.limit, "order_by": "-run_after"})
        return {"dag_id": args.dag_id, "runs": [self._run(r) for r in data.get("dag_runs", [])]}

    def trigger(self, args: AfTriggerInput) -> dict[str, Any]:
        self._check_dag(args.dag_id)
        self._check_conf(args.conf)
        body = {"logical_date": datetime.now(timezone.utc).isoformat(), "conf": args.conf}
        run = self.client.request("POST", f"/dags/{args.dag_id}/dagRuns", json=body)
        return {"triggered": True, **self._run(run),
                "next_step": "Use airflow_wait_for_dag_run or airflow_get_dag_runs to follow progress."}

    def _check_conf(self, conf: dict) -> None:
        bad = [k for k in conf if k not in self.settings.approved_dag_conf_keys]
        if bad:
            raise PermissionDenied(f"conf keys not approved: {bad}. Allowed: {list(self.settings.approved_dag_conf_keys)}")

    def task_instances(self, args: AfRunRefInput) -> dict[str, Any]:
        run_id = args.dag_run_id or self._latest_run_id(args.dag_id)
        data = self.client.request("GET", f"/dags/{args.dag_id}/dagRuns/{run_id}/taskInstances")
        tis = [self._ti(t) for t in data.get("task_instances", [])]
        return {"dag_id": args.dag_id, "dag_run_id": run_id, "task_instances": tis}

    def failed_tasks(self, args: AfRunRefInput) -> dict[str, Any]:
        run_id = args.dag_run_id or self._latest_run_id(args.dag_id)
        info = self.task_instances(AfRunRefInput(dag_id=args.dag_id, dag_run_id=run_id))
        failed = [t for t in info["task_instances"] if t["state"] in ("failed", "upstream_failed")]
        root = [t for t in failed if t["state"] == "failed"]
        run = self.client.request("GET", f"/dags/{args.dag_id}/dagRuns/{run_id}")
        return {"dag_id": args.dag_id, "dag_run_id": run_id, "run_state": run.get("state"),
                "failed_tasks": root, "upstream_failed_tasks": [t for t in failed if t["state"] == "upstream_failed"],
                "skipped_tasks": [t["task_id"] for t in info["task_instances"] if t["state"] == "skipped"],
                "hint": "Use airflow_get_task_logs on a failed task for the failure reason." if root else None}

    def task_logs(self, args: AfTaskLogsInput) -> dict[str, Any]:
        data = self.client.request(
            "GET", f"/dags/{args.dag_id}/dagRuns/{args.dag_run_id}/taskInstances/{args.task_id}/logs/{args.try_number}",
            accept="application/json")
        content = data.get("content", data) if isinstance(data, dict) else data
        lines: list[str] = []
        if isinstance(content, str):
            lines = content.splitlines()
        else:
            for item in content or []:
                if isinstance(item, dict):
                    lines.append(f"{item.get('timestamp', '')} {item.get('event', item)}".strip())
                else:
                    lines.append(str(item))
        tail = [redact(line)[:400] for line in lines[-args.tail_lines:]]
        errors = [line for line in tail if any(w in line.lower() for w in ("error", "exception", "failed", "traceback"))]
        return {"dag_id": args.dag_id, "dag_run_id": args.dag_run_id, "task_id": args.task_id,
                "try_number": args.try_number, "total_lines": len(lines), "tail": tail, "error_lines": errors[-10:]}

    def wait(self, args: AfWaitInput) -> dict[str, Any]:
        self._check_dag(args.dag_id)
        deadline = time.monotonic() + args.timeout_s
        state = None
        while True:
            run = self.client.request("GET", f"/dags/{args.dag_id}/dagRuns/{args.dag_run_id}")
            state = run.get("state")
            if state in TERMINAL or time.monotonic() >= deadline:
                break
            time.sleep(args.poll_s)
        tis = self.task_instances(AfRunRefInput(dag_id=args.dag_id, dag_run_id=args.dag_run_id))["task_instances"]
        return {"dag_id": args.dag_id, "dag_run_id": args.dag_run_id, "state": state,
                "finished": state in TERMINAL, "tasks": [{"task_id": t["task_id"], "state": t["state"]} for t in tis]}


def build_tools(settings: Settings, client: AirflowClient | None = None) -> list[Tool]:
    s = AirflowService(settings, client)

    def dag_precheck(args: Any) -> None:
        s._check_dag(args.dag_id)

    def trigger_precheck(args: Any) -> None:
        s._check_dag(args.dag_id)
        s._check_conf(args.conf)

    return [
        Tool("airflow_list_dags", "List Airflow DAGs and whether each is approved for agent use.", "airflow",
             AfListDagsInput, s.list_dags),
        Tool("airflow_get_dag_runs", "Get recent DAG runs and their states (success/failed/running).", "airflow",
             AfDagRunsInput, s.dag_runs),
        Tool("airflow_trigger_dag", "Trigger an APPROVED training DAG. conf keys limited to: "
             + ", ".join(settings.approved_dag_conf_keys), "airflow", AfTriggerInput, s.trigger,
             mutating=settings.airflow_require_confirmation, precheck=trigger_precheck),
        Tool("airflow_get_task_instances", "Get per-task states of a DAG run (default: latest).", "airflow",
             AfRunRefInput, s.task_instances),
        Tool("airflow_get_failed_tasks", "Identify failed tasks (and upstream_failed) of a DAG run (default: latest).",
             "airflow", AfRunRefInput, s.failed_tasks),
        Tool("airflow_get_task_logs", "Fetch the tail of a task's logs for failure diagnostics.", "airflow",
             AfTaskLogsInput, s.task_logs),
        Tool("airflow_wait_for_dag_run", "Poll an approved DAG run until it succeeds/fails or the timeout elapses.",
             "airflow", AfWaitInput, s.wait, timeout_s=330, precheck=dag_precheck),
    ]
