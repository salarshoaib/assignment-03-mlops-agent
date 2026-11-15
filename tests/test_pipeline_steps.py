import pytest

from training import pipeline_steps as ps
from training.generate_data import generate


@pytest.fixture
def data(tmp_path):
    p = tmp_path / "d.csv"
    generate(600, 1, 1.0, 0.03).to_csv(p, index=False)
    return str(p)


def test_validation_passes_and_fails(data, tmp_path):
    assert ps.validate_dataset(data)["rows"] == 600
    with pytest.raises(ps.DataValidationError, match="too few rows"):
        ps.validate_dataset(data, min_rows=10**6)
    with pytest.raises(ps.DataValidationError, match="not found"):
        ps.validate_dataset(str(tmp_path / "nope.csv"))
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ps.DataValidationError, match="missing columns"):
        ps.validate_dataset(str(bad))


def test_preprocess_split(data, tmp_path):
    out = ps.preprocess(data, str(tmp_path / "w"))
    assert out["n_train"] + out["n_test"] == 600 and out["n_test"] == 120


def test_end_to_end_gate_blocks_and_promotes(data, tmp_path):
    from mlflow.tracking import MlflowClient
    uri = f"sqlite:///{tmp_path}/m.db"
    MlflowClient(uri).create_experiment("e", artifact_location=(tmp_path / "art").as_uri())
    prep = ps.preprocess(data, str(tmp_path / "w"))
    tr = ps.train_model(prep["train_path"], str(tmp_path / "model.joblib"), 20, 5)
    ev = ps.evaluate_and_log(prep["test_path"], tr["model_path"], tr["params"], data, uri, "e", "r1", repo_root=tmp_path)
    client = MlflowClient(uri)
    tags = client.get_run(ev["run_id"]).data.tags
    assert tags["dvc.md5"] != "unknown" and tags["dvc.dataset_path"]
    with pytest.raises(ps.QualityGateError):
        ps.quality_gate(ev["run_id"], uri, min_f1=0.9999, min_accuracy=0.5)
    assert client.get_run(ev["run_id"]).data.tags["quality_gate"] == "failed"
    assert client.search_registered_models() == []  # nothing promoted
    res = ps.quality_gate(ev["run_id"], uri, min_f1=0.5, min_accuracy=0.5)
    assert res["promoted"] and client.get_model_version_by_alias(ps.MODEL_NAME, "candidate").version == res["version"]
