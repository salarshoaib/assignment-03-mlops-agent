import pytest

from agent.tools import build_registry
from agent.tools.base import PermissionDenied
from agent.tools.dvc_tools import CommandRunner

from .conftest import MD5_V1, MD5_V2, write_dvc


def reg(settings, repo, client=None):
    from agent.config import Settings
    s = Settings(repo_root=repo, tool_timeout_s=10)
    return build_registry(s, mlflow_client=client)


def test_list_tracked_datasets(git_repo, settings):
    res = reg(settings, git_repo).execute("dvc_list_tracked_datasets", {})
    assert res.ok and res.data["datasets"][0]["dataset_path"] == "data/raw/dataset.csv"


def test_get_version_commit_and_tag(git_repo, settings):
    d = reg(settings, git_repo).execute("dvc_get_dataset_version", {}).data
    assert d["md5"] == MD5_V2 and d["dataset_version"] == "data-v2" and len(d["commit"]) == 40


def test_compare_versions(git_repo, settings):
    d = reg(settings, git_repo).execute("dvc_compare_versions", {"rev_a": "data-v1", "rev_b": "data-v2"}).data
    assert d["changed"] is True and d["a"]["md5"] == MD5_V1 and d["b"]["md5"] == MD5_V2
    assert d["size_delta_bytes"] == 500


def test_detect_changes_via_md5(git_repo, settings):
    import hashlib
    data = git_repo / "data" / "raw" / "dataset.csv"
    data.write_text("a,b\n1,2\n")
    write_dvc(git_repo, hashlib.md5(data.read_bytes()).hexdigest(), 8)
    r = reg(settings, git_repo)
    assert r.execute("dvc_detect_changes", {}).data["changed"] is False
    data.write_text("a,b\n1,3\n")
    assert r.execute("dvc_detect_changes", {}).data["changed"] is True


def test_explain_reproduce_by_revision(git_repo, settings):
    d = reg(settings, git_repo).execute("dvc_explain_reproduce", {"rev": "data-v1"}).data
    assert d["expected_md5"] == MD5_V1 and any("dvc checkout" in s for s in d["steps"])


def test_dataset_for_run_resolves_commit(git_repo, settings, mlflow_client):
    run = mlflow_client.search_runs([mlflow_client.eid], filter_string="tags.mlflow.runName = 'run-2'")[0]
    d = reg(settings, git_repo, mlflow_client).execute("dvc_get_dataset_for_run", {"run_id": run.info.run_id}).data
    assert d["recorded"]["md5"] == MD5_V1
    assert "data-v1" in d["resolved_commit"]["tags"]
    assert d["current"]["matches_recorded"] is False  # data changed since v1


def test_dataset_for_run_name_and_missing_provenance(git_repo, settings, mlflow_client):
    r = reg(settings, git_repo, mlflow_client)
    assert r.execute("dvc_get_dataset_for_run", {"run_name": "run-1"}).ok
    mlflow_client.delete_tag(mlflow_client.search_runs([mlflow_client.eid], filter_string="tags.mlflow.runName = 'run-1'")[0].info.run_id, "dvc.md5")
    assert r.execute("dvc_get_dataset_for_run", {"run_name": "run-1"}).data["provenance_missing"] is True


def test_command_allowlist_blocks_unsafe_commands(tmp_path):
    runner = CommandRunner(tmp_path)
    for binary, args in [("git", ["push"]), ("git", ["checkout", "main"]), ("git", ["tag", "-d", "x"]),
                         ("dvc", ["push"]), ("dvc", ["gc"]), ("rm", ["-rf", "/"]), ("bash", ["-c", "id"])]:
        with pytest.raises(PermissionDenied):
            runner.run(binary, args)


def test_invalid_revision_and_path_traversal_rejected(git_repo, settings):
    r = reg(settings, git_repo)
    assert r.execute("dvc_compare_versions", {"rev_a": "--upload-pack=evil", "rev_b": "data-v1"}).error_type == "validation_error"
    assert r.execute("dvc_get_dataset_version", {"dataset_path": "../../etc/passwd"}).error_type == "permission_denied"
    assert r.execute("dvc_compare_versions", {"rev_a": "no-such-tag", "rev_b": "data-v1"}).error_type == "validation_error"
