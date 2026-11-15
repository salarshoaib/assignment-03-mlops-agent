import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from api.main import create_app


@pytest.fixture
def model_path(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 10))
    m = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, (X[:, 0] > 0).astype(int))
    p = tmp_path / "model.joblib"
    joblib.dump(m, p)
    return str(p)


def test_ready_and_predict(model_path):
    with TestClient(create_app(model_path)) as c:
        assert c.get("/health").status_code == 200 and c.get("/ready").status_code == 200
        r = c.post("/predict", json={"features": [0.5] * 10})
        assert r.status_code == 200 and r.json()["prediction"] in (0, 1)
        assert c.post("/predict", json={"features": [1.0, 2.0]}).status_code == 422


def test_missing_model_fails_readiness_but_not_liveness(tmp_path):
    with TestClient(create_app(str(tmp_path / "missing.joblib"))) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json={"features": [1.0] * 10}).status_code == 503
