from agent.tools import build_registry

from .conftest import FakeApps, FakeCore, make_deployment, make_event, make_pod


def reg(settings, pods=None, events=None, logs="INFO ok", deployment=None):
    apps = FakeApps(deployment)
    return build_registry(settings, core_api=FakeCore(pods or [], events or [], logs), apps_api=apps), apps


def test_unauthorized_namespace_modification_rejected(settings):
    r, apps = reg(settings)
    for confirmed in (False, True):
        res = r.execute("k8s_restart_deployment", {"namespace": "kube-system"}, confirmed=confirmed)
        assert res.error_type == "permission_denied"
    assert apps.patches == [] and r.pending_actions() == []
    assert r.execute("k8s_scale_deployment", {"namespace": "default", "replicas": 3}, confirmed=True).error_type == "permission_denied"
    assert apps.scales == []


def test_read_in_foreign_namespace_rejected(settings):
    r, _ = reg(settings)
    assert r.execute("k8s_list_pods", {"namespace": "kube-system"}).error_type == "permission_denied"


def test_restart_needs_confirmation_then_patches(settings):
    r, apps = reg(settings)
    first = r.execute("k8s_restart_deployment", {})
    assert first.error_type == "confirmation_required" and apps.patches == []
    done = r.confirm(first.data["pending_id"])
    assert done.ok and "restartedAt" in str(apps.patches[0][2])


def test_scale_limits_validated(settings):
    r, _ = reg(settings)
    assert r.execute("k8s_scale_deployment", {"replicas": 0}).error_type == "validation_error"
    assert r.execute("k8s_scale_deployment", {"replicas": 99}).error_type == "validation_error"


def test_list_and_unhealthy_detection(settings):
    pods = [make_pod("ok-1"), make_pod("bad-1", ready=False, restarts=4, waiting="CrashLoopBackOff", last_term="OOMKilled")]
    r, _ = reg(settings, pods)
    res = r.execute("k8s_find_unhealthy_pods", {})
    assert res.ok and not res.data["all_healthy"]
    assert [p["name"] for p in res.data["unhealthy"]] == ["bad-1"]
    assert res.data["restart_events"] == [{"name": "bad-1", "restarts": 4}]
    assert any("CrashLoopBackOff" in x for x in res.data["unhealthy"][0]["problems"])


def test_deployed_version_reports_image_label_config(settings):
    r, _ = reg(settings)
    d = r.execute("k8s_get_deployed_version", {}).data
    assert d["image_tag"] == "1.0.0" and d["app_version_label"] == "1.0.0"
    assert d["config"]["DATASET_VERSION"] == "data-v2" and "SECRET_KEY" not in d["config"]


def test_logs_are_redacted(settings):
    r, _ = reg(settings, [make_pod("p1")], logs="ERROR db login password=hunter2\nINFO fine")
    res = r.execute("k8s_get_pod_logs", {"pod_name": "p1"})
    assert res.ok and "hunter2" not in str(res.data) and res.data["error_lines"]


def test_services_report_endpoints(settings):
    r, _ = reg(settings, [make_pod("a"), make_pod("b", ready=False)])
    s = r.execute("k8s_list_services", {}).data["services"][0]
    assert s["type"] == "ClusterIP" and s["ready_endpoints"] == 1 and s["not_ready_endpoints"] == 1


def test_diagnose_readiness_failure_uses_events_and_logs(settings):
    pods = [make_pod("p1", ready=False), make_pod("p2", ready=False)]
    events = [make_event("p1", "Unhealthy", "Readiness probe failed: HTTP probe failed with statuscode: 503")]
    r, _ = reg(settings, pods, events, logs="ERROR MODEL LOAD FAILED: FileNotFoundError",
               deployment=make_deployment(desired=2, ready=0))
    res = r.execute("k8s_diagnose_deployment", {})
    assert res.ok and not res.data["healthy"]
    text = " ".join(res.data["findings"])
    assert "0/2 replicas" in text and "readiness probe" in text.lower()
    assert any("MODEL LOAD FAILED" in ln for lines in res.data["error_log_lines"].values() for ln in lines)


def test_diagnose_image_pull_failure(settings):
    pods = [make_pod("p1", phase="Pending", ready=False, waiting="ImagePullBackOff", image="mlops-inference:9.9.9")]
    r, _ = reg(settings, pods, deployment=make_deployment(ready=0, image="mlops-inference:9.9.9"))
    findings = " ".join(r.execute("k8s_diagnose_deployment", {}).data["findings"])
    assert "image cannot be pulled" in findings


def test_api_failure_becomes_service_unavailable(settings):
    class Boom(FakeCore):
        def list_namespaced_pod(self, ns, label_selector=None):
            raise ConnectionError("cluster down")
    r = build_registry(settings, core_api=Boom(), apps_api=FakeApps())
    assert r.execute("k8s_list_pods", {}).error_type == "service_unavailable"
