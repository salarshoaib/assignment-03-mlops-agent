from agent.tools import build_registry

from .conftest import DownMlflow


def reg(settings, client):
    return build_registry(settings, mlflow_client=client)


def test_best_run_ignores_failed_runs_and_flags_caveats(settings, mlflow_client):
    d = reg(settings, mlflow_client).execute("mlflow_find_best_run", {"metric": "f1"}).data
    assert d["best"]["run_name"] == "run-2"  # run-3 has higher f1 but FAILED
    assert d["scored"] == 3 and any(s["reason"] == "not FINISHED" for s in d["skipped"])
    assert d["readiness_checks"]["registered_in_model_registry"] is False
    assert any("not registered" in c for c in d["caveats"])
    assert "does NOT make a model production-ready" in d["interpretation"]


def test_best_run_min_mode(settings, mlflow_client):
    d = reg(settings, mlflow_client).execute("mlflow_find_best_run", {"metric": "f1", "mode": "min"}).data
    assert d["best"]["run_name"] == "run-1"


def test_close_scores_are_flagged_as_noise(settings, mlflow_client):
    d = reg(settings, mlflow_client).execute("mlflow_find_best_run", {"metric": "f1"}).data
    assert any("within noise" in c for c in d["caveats"])  # 0.91 vs 0.905


def test_threshold_caveat(settings, mlflow_client):
    d = reg(settings, mlflow_client).execute("mlflow_find_best_run", {"metric": "f1", "mode": "min"}).data
    assert d["readiness_checks"]["meets_min_f1"] is True or "below" in " ".join(d["caveats"])


def test_list_runs_and_details(settings, mlflow_client):
    r = reg(settings, mlflow_client)
    runs = r.execute("mlflow_list_runs", {"max_results": 3}).data
    assert runs["count"] == 3 and "metrics" in runs["runs"][0]
    detail = r.execute("mlflow_get_run_details", {"run_id": runs["runs"][0]["run_id"]})
    assert detail.ok and "params" in detail.data


def test_compare_runs_reports_param_differences(settings, mlflow_client):
    r = reg(settings, mlflow_client)
    ids = [x["run_id"] for x in r.execute("mlflow_list_runs", {}).data["runs"]][:3]
    d = r.execute("mlflow_compare_runs", {"run_ids": ids}).data
    assert "n_estimators" in d["differing_params"] and "max_depth" not in d["differing_params"]
    assert d["different_datasets"] is False


def test_compare_runs_requires_two(settings, mlflow_client):
    assert reg(settings, mlflow_client).execute("mlflow_compare_runs", {"run_ids": ["x"]}).error_type == "validation_error"


def test_compare_experiments_and_registry(settings, mlflow_client):
    mlflow_client.create_experiment("second")
    r = reg(settings, mlflow_client)
    d = r.execute("mlflow_compare_experiments", {"experiment_names": ["mlops-agent", "second"]}).data
    assert d["experiments"][0]["best"] == 0.91 and d["experiments"][1]["best"] is None
    assert r.execute("mlflow_get_model_registry", {}).data["count"] == 0
    assert r.execute("mlflow_list_experiments", {}).ok


def test_unknown_experiment_and_run(settings, mlflow_client):
    r = reg(settings, mlflow_client)
    assert r.execute("mlflow_find_best_run", {"experiment_name": "ghost"}).error_type == "validation_error"
    assert r.execute("mlflow_get_run_details", {"run_id": "doesnotexist"}).error_type == "validation_error"


def test_server_down_is_service_unavailable_and_redacted(settings):
    res = reg(settings, DownMlflow()).execute("mlflow_find_best_run", {})
    assert res.error_type == "service_unavailable" and "SUPERSECRET" not in res.error
