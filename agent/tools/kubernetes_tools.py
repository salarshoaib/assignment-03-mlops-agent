"""Task 4 - Kubernetes operations tools. Every call is confined to approved namespaces."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent.config import Settings
from agent.logging_utils import redact
from agent.schemas import (
    K8sDeploymentInput,
    K8sEventsInput,
    K8sNamespaceInput,
    K8sPodLogsInput,
    K8sPodNameInput,
    K8sPodsInput,
    K8sScaleInput,
)
from agent.tools.base import PermissionDenied, ServiceUnavailable, Tool, ToolInputError


def _g(obj: Any, path: str, default: Any = None) -> Any:
    """Safe attribute-chain getter (works with kubernetes models and test fakes)."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, part, None)
    return default if cur is None else cur


def _iso(ts: Any) -> str | None:
    return ts.isoformat() if isinstance(ts, datetime) else (str(ts) if ts else None)


class KubernetesService:
    def __init__(self, settings: Settings, core_api: Any = None, apps_api: Any = None):
        self.settings = settings
        self._core, self._apps = core_api, apps_api

    # ------------------------------------------------------------------ clients + guards
    def _load(self) -> None:
        if self._core and self._apps:
            return
        try:
            from kubernetes import client, config
            try:
                config.load_incluster_config()
            except Exception:  # noqa: BLE001
                config.load_kube_config()
            self._core = self._core or client.CoreV1Api()
            self._apps = self._apps or client.AppsV1Api()
        except Exception as exc:  # noqa: BLE001
            raise ServiceUnavailable(f"Kubernetes client could not be configured: {type(exc).__name__}") from exc

    @property
    def core(self) -> Any:
        self._load()
        return self._core

    @property
    def apps(self) -> Any:
        self._load()
        return self._apps

    def ns(self, namespace: str | None) -> str:
        namespace = namespace or self.settings.k8s_default_namespace
        if namespace not in self.settings.k8s_allowed_namespaces:
            raise PermissionDenied(
                f"Namespace '{namespace}' is not approved. Allowed: {list(self.settings.k8s_allowed_namespaces)}")
        return namespace

    def _api(self, fn: Any, *a: Any, **kw: Any) -> Any:
        try:
            return fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001
            status = getattr(exc, "status", None)
            if status == 404:
                raise ToolInputError("Kubernetes resource not found") from exc
            if status in (401, 403):
                raise PermissionDenied(f"Kubernetes API denied access (HTTP {status})") from exc
            raise ServiceUnavailable(f"Kubernetes API error: {type(exc).__name__}: {str(exc)[:150]}") from exc

    # ------------------------------------------------------------------ summaries
    @staticmethod
    def _container_statuses(pod: Any) -> list[dict[str, Any]]:
        out = []
        for cs in _g(pod, "status.container_statuses", []) or []:
            waiting, terminated = _g(cs, "state.waiting"), _g(cs, "state.terminated")
            last_term = _g(cs, "last_state.terminated")
            out.append({
                "name": cs.name, "ready": bool(_g(cs, "ready", False)), "restart_count": int(_g(cs, "restart_count", 0)),
                "image": _g(cs, "image"),
                "waiting_reason": _g(waiting, "reason"), "waiting_message": redact(_g(waiting, "message", ""))[:200] or None,
                "terminated_reason": _g(terminated, "reason"), "exit_code": _g(terminated, "exit_code"),
                "last_terminated_reason": _g(last_term, "reason"), "last_exit_code": _g(last_term, "exit_code"),
            })
        return out

    def _pod(self, pod: Any) -> dict[str, Any]:
        statuses = self._container_statuses(pod)
        conds = {c.type: c.status for c in _g(pod, "status.conditions", []) or []}
        return {
            "name": pod.metadata.name, "phase": _g(pod, "status.phase"), "node": _g(pod, "spec.node_name"),
            "ready": conds.get("Ready") == "True", "conditions": conds,
            "restarts": sum(s["restart_count"] for s in statuses), "containers": statuses,
            "created": _iso(_g(pod, "metadata.creation_timestamp")),
            "image": (statuses[0]["image"] if statuses else None),
        }

    @staticmethod
    def _problems(p: dict[str, Any]) -> list[str]:
        issues = []
        if p["phase"] not in ("Running", "Succeeded"):
            issues.append(f"phase={p['phase']}")
        if p["phase"] == "Running" and not p["ready"]:
            issues.append("not Ready (readiness probe failing or containers not started)")
        for c in p["containers"]:
            if c["waiting_reason"]:
                issues.append(f"{c['name']}: waiting={c['waiting_reason']}")
            if c["restart_count"] > 0:
                reason = c["last_terminated_reason"] or c["terminated_reason"] or "unknown"
                issues.append(f"{c['name']}: restarted {c['restart_count']}x (last termination: {reason}, "
                              f"exit={c['last_exit_code']})")
        return issues

    def _deployment(self, d: Any) -> dict[str, Any]:
        containers = _g(d, "spec.template.spec.containers", []) or []
        labels = _g(d, "metadata.labels", {}) or {}
        return {
            "name": d.metadata.name, "desired_replicas": _g(d, "spec.replicas", 0),
            "ready_replicas": _g(d, "status.ready_replicas", 0), "available_replicas": _g(d, "status.available_replicas", 0),
            "updated_replicas": _g(d, "status.updated_replicas", 0), "images": [c.image for c in containers],
            "app_version_label": labels.get("app.kubernetes.io/version"),
            "conditions": [{"type": c.type, "status": c.status, "reason": _g(c, "reason"),
                            "message": redact(_g(c, "message", ""))[:200]} for c in _g(d, "status.conditions", []) or []],
        }

    # ------------------------------------------------------------------ tools
    def list_deployments(self, a: K8sNamespaceInput) -> dict[str, Any]:
        ns = self.ns(a.namespace)
        items = self._api(self.apps.list_namespaced_deployment, ns).items
        return {"namespace": ns, "deployments": [self._deployment(d) for d in items]}

    def list_pods(self, a: K8sPodsInput) -> dict[str, Any]:
        ns = self.ns(a.namespace)
        sel = a.label_selector or self.settings.k8s_label_selector
        items = self._api(self.core.list_namespaced_pod, ns, label_selector=sel).items
        return {"namespace": ns, "selector": sel, "pods": [self._pod(p) for p in items]}

    def pod_status(self, a: K8sPodNameInput) -> dict[str, Any]:
        ns = self.ns(a.namespace)
        pod = self._api(self.core.read_namespaced_pod, a.pod_name, ns)
        summary = self._pod(pod)
        probes = {}
        for c in _g(pod, "spec.containers", []) or []:
            for kind in ("readiness_probe", "liveness_probe", "startup_probe"):
                pr = _g(c, kind)
                if pr is not None:
                    probes[f"{c.name}.{kind}"] = {"path": _g(pr, "http_get.path"), "port": str(_g(pr, "http_get.port", ""))}
        return {"namespace": ns, **summary, "problems": self._problems(summary), "probes": probes}

    def pod_logs(self, a: K8sPodLogsInput) -> dict[str, Any]:
        ns = self.ns(a.namespace)
        kwargs: dict[str, Any] = {"tail_lines": a.tail_lines, "previous": a.previous}
        if a.container:
            kwargs["container"] = a.container
        text = self._api(self.core.read_namespaced_pod_log, a.pod_name, ns, **kwargs)
        lines = [redact(line)[:400] for line in str(text).splitlines()]
        errs = [ln for ln in lines if any(w in ln.lower() for w in ("error", "exception", "traceback", "critical", "failed"))]
        return {"namespace": ns, "pod": a.pod_name, "previous": a.previous, "line_count": len(lines),
                "logs": lines[-a.tail_lines:], "error_lines": errs[-10:]}

    def services(self, a: K8sNamespaceInput) -> dict[str, Any]:
        ns = self.ns(a.namespace)
        out = []
        for s in self._api(self.core.list_namespaced_service, ns).items:
            try:
                eps = self.core.read_namespaced_endpoints(s.metadata.name, ns)
                ready = sum(len(_g(sub, "addresses", []) or []) for sub in _g(eps, "subsets", []) or [])
                not_ready = sum(len(_g(sub, "not_ready_addresses", []) or []) for sub in _g(eps, "subsets", []) or [])
            except Exception:  # noqa: BLE001
                ready = not_ready = None
            out.append({"name": s.metadata.name, "type": _g(s, "spec.type"), "cluster_ip": _g(s, "spec.cluster_ip"),
                        "selector": _g(s, "spec.selector", {}),
                        "ports": [{"port": p.port, "target_port": str(p.target_port), "protocol": p.protocol}
                                  for p in _g(s, "spec.ports", []) or []],
                        "ready_endpoints": ready, "not_ready_endpoints": not_ready})
        return {"namespace": ns, "services": out}

    def events(self, a: K8sEventsInput) -> dict[str, Any]:
        ns = self.ns(a.namespace)
        items = self._api(self.core.list_namespaced_event, ns).items
        rows = []
        for e in items:
            name = _g(e, "involved_object.name", "")
            if a.object_name and a.object_name not in name:
                continue
            rows.append({"time": _iso(_g(e, "last_timestamp") or _g(e, "event_time") or _g(e, "metadata.creation_timestamp")),
                         "type": _g(e, "type"), "reason": _g(e, "reason"), "object": name,
                         "message": redact(_g(e, "message", ""))[:300], "count": _g(e, "count", 1)})
        rows.sort(key=lambda r: r["time"] or "", reverse=True)
        return {"namespace": ns, "events": rows[: a.limit],
                "warnings": [r for r in rows if r["type"] == "Warning"][:10]}

    def unhealthy(self, a: K8sPodsInput) -> dict[str, Any]:
        pods = self.list_pods(a)
        bad, restarts = [], []
        for p in pods["pods"]:
            problems = self._problems(p)
            if problems:
                bad.append({"name": p["name"], "phase": p["phase"], "ready": p["ready"], "problems": problems})
            if p["restarts"]:
                restarts.append({"name": p["name"], "restarts": p["restarts"]})
        return {"namespace": pods["namespace"], "total_pods": len(pods["pods"]), "unhealthy": bad,
                "restart_events": restarts, "all_healthy": not bad}

    def deployed_version(self, a: K8sDeploymentInput) -> dict[str, Any]:
        ns = self.ns(a.namespace)
        name = a.deployment_name or self.settings.k8s_deployment
        dep = self._api(self.apps.read_namespaced_deployment, name, ns)
        summary = self._deployment(dep)
        image = summary["images"][0] if summary["images"] else None
        tag = image.rsplit(":", 1)[1] if image and ":" in image.rsplit("/", 1)[-1] else None
        config_values: dict[str, Any] = {}
        try:
            for ref in _g(dep, "spec.template.spec.containers", [])[0].env_from or []:
                cm_name = _g(ref, "config_map_ref.name")
                if cm_name:
                    cm = self.core.read_namespaced_config_map(cm_name, ns)
                    data = cm.data or {}
                    config_values = {k: v for k, v in data.items() if k in ("MODEL_VERSION", "MODEL_PATH", "DATASET_VERSION", "APP_VERSION")}
        except Exception:  # noqa: BLE001
            config_values = {"note": "ConfigMap not readable"}
        return {"namespace": ns, "deployment": name, "image": image, "image_tag": tag,
                "app_version_label": summary["app_version_label"], "config": config_values,
                "desired_replicas": summary["desired_replicas"], "ready_replicas": summary["ready_replicas"],
                "available_replicas": summary["available_replicas"]}

    def diagnose(self, a: K8sDeploymentInput) -> dict[str, Any]:
        ns = self.ns(a.namespace)
        name = a.deployment_name or self.settings.k8s_deployment
        dep = self._api(self.apps.read_namespaced_deployment, name, ns)
        summary = self._deployment(dep)
        sel = ",".join(f"{k}={v}" for k, v in (_g(dep, "spec.selector.match_labels", {}) or {}).items())
        pods = [self._pod(p) for p in self._api(self.core.list_namespaced_pod, ns, label_selector=sel).items]
        events = self.events(K8sEventsInput(namespace=ns, limit=30))["events"]
        pod_events = [e for e in events if any(p["name"] in e["object"] for p in pods) or name in e["object"]]
        findings: list[str] = []
        if summary["ready_replicas"] < summary["desired_replicas"]:
            findings.append(f"Only {summary['ready_replicas']}/{summary['desired_replicas']} replicas are ready.")
        logs_by_pod: dict[str, Any] = {}
        for p in pods:
            probs = self._problems(p)
            findings += [f"{p['name']}: {x}" for x in probs]
            reasons = {c["waiting_reason"] for c in p["containers"]}
            if reasons & {"ImagePullBackOff", "ErrImagePull"}:
                findings.append(f"{p['name']}: image cannot be pulled - check image name/tag ({p['image']}) and registry access or `kind load`.")
            if any(c["last_terminated_reason"] == "OOMKilled" for c in p["containers"]):
                findings.append(f"{p['name']}: container was OOMKilled - memory limit too low.")
            if probs and p["phase"] == "Running":
                try:
                    logs_by_pod[p["name"]] = self.pod_logs(K8sPodLogsInput(pod_name=p["name"], namespace=ns, tail_lines=40))["error_lines"]
                except Exception as exc:  # noqa: BLE001
                    logs_by_pod[p["name"]] = [f"logs unavailable: {type(exc).__name__}"]
        for e in pod_events:
            msg = (e["message"] or "").lower()
            if "readiness probe failed" in msg:
                findings.append(f"Event: readiness probe failing ({e['message'][:120]}) - pods are removed from Service endpoints.")
            elif "liveness probe failed" in msg:
                findings.append(f"Event: liveness probe failing ({e['message'][:120]}) - container restarts.")
            elif e["reason"] in ("BackOff", "Failed", "FailedScheduling", "FailedMount"):
                findings.append(f"Event {e['reason']}: {e['message'][:140]}")
        return {"namespace": ns, "deployment": summary, "pods": pods, "events": pod_events[:15],
                "error_log_lines": logs_by_pod, "findings": sorted(set(findings)),
                "healthy": not findings, "note": "Findings are derived only from the collected pod status, events and logs."}

    def restart(self, a: K8sDeploymentInput) -> dict[str, Any]:
        ns = self.ns(a.namespace)
        name = a.deployment_name or self.settings.k8s_deployment
        body = {"spec": {"template": {"metadata": {"annotations": {
            "kubectl.kubernetes.io/restartedAt": datetime.now(timezone.utc).isoformat()}}}}}
        self._api(self.apps.patch_namespaced_deployment, name, ns, body)
        return {"restarted": True, "deployment": name, "namespace": ns, "method": "rolling restart annotation"}

    def scale(self, a: K8sScaleInput) -> dict[str, Any]:
        ns = self.ns(a.namespace)
        name = a.deployment_name or self.settings.k8s_deployment
        self._api(self.apps.patch_namespaced_deployment_scale, name, ns, {"spec": {"replicas": a.replicas}})
        return {"scaled": True, "deployment": name, "namespace": ns, "replicas": a.replicas}


def build_tools(settings: Settings, core_api: Any = None, apps_api: Any = None) -> list[Tool]:
    s = KubernetesService(settings, core_api, apps_api)

    def ns_precheck(args: Any) -> None:
        s.ns(getattr(args, "namespace", None))

    def t(name: str, desc: str, model: type, fn: Any, **kw: Any) -> Tool:
        return Tool(name, desc, "kubernetes", model, fn, precheck=ns_precheck, **kw)

    return [
        t("k8s_list_deployments", "List Deployments with replica counts and images.", K8sNamespaceInput, s.list_deployments),
        t("k8s_list_pods", "List application Pods with phase, readiness and restart counts.", K8sPodsInput, s.list_pods),
        t("k8s_get_pod_status", "Detailed status of one Pod: containers, probes, conditions, problems.", K8sPodNameInput, s.pod_status),
        t("k8s_get_pod_logs", "Tail logs of a Pod (secrets redacted); previous=true for the crashed container.", K8sPodLogsInput, s.pod_logs),
        t("k8s_list_services", "List Services with selectors, ports and ready endpoints.", K8sNamespaceInput, s.services),
        t("k8s_get_events", "Recent namespace events (warnings first-class evidence).", K8sEventsInput, s.events),
        t("k8s_find_unhealthy_pods", "Identify unhealthy Pods and restart events.", K8sPodsInput, s.unhealthy),
        t("k8s_get_deployed_version", "Report the deployed application version (image tag, version label, ConfigMap values).",
          K8sDeploymentInput, s.deployed_version),
        t("k8s_diagnose_deployment", "Collect pod status, probes, events and logs and derive evidence-based findings "
          "(use for 'why is my API unavailable?').", K8sDeploymentInput, s.diagnose),
        t("k8s_restart_deployment", "Rolling-restart a Deployment in an approved namespace.", K8sDeploymentInput, s.restart,
          mutating=True),
        t("k8s_scale_deployment", "Scale a Deployment (1-5 replicas) in an approved namespace.", K8sScaleInput, s.scale,
          mutating=True),
    ]
