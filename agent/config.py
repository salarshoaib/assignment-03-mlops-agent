"""Central configuration, read from environment variables (never hard-code secrets)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional convenience
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


def _csv(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(p.strip() for p in raw.split(",") if p.strip())


def _opt_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else None


@dataclass(frozen=True)
class Settings:
    repo_root: Path = field(default_factory=lambda: Path.cwd())
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "mlops-agent"
    airflow_base_url: str = "http://localhost:8080"
    airflow_username: str = "admin"
    airflow_password: str = field(default="", repr=False)
    approved_dags: tuple[str, ...] = ("training_pipeline",)
    approved_dag_conf_keys: tuple[str, ...] = (
        "inject_failure", "min_f1", "min_accuracy", "experiment_name", "n_estimators", "max_depth",
    )
    k8s_allowed_namespaces: tuple[str, ...] = ("mlops-agent",)
    k8s_default_namespace: str = "mlops-agent"
    k8s_deployment: str = "ml-inference"
    k8s_label_selector: str = "app=ml-inference"
    dvc_default_dataset: str = "data/raw/dataset.csv"
    anthropic_api_key: str = field(default="", repr=False)
    agent_model: str = "claude-sonnet-4-6"
    tool_timeout_s: float = 30.0
    min_f1: float = 0.80
    min_accuracy: float = 0.80
    model_max_age_days: int | None = None
    confirmation_ttl_s: int = 300
    agent_api_token: str = field(default="", repr=False)
    log_level: str = "INFO"
    log_file: str = ""
    airflow_require_confirmation: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        if load_dotenv:
            load_dotenv()
        ns = _csv("K8S_ALLOWED_NAMESPACES", "mlops-agent")
        return cls(
            repo_root=Path(os.getenv("REPO_ROOT", Path.cwd())).resolve(),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            mlflow_experiment=os.getenv("MLFLOW_EXPERIMENT", "mlops-agent"),
            airflow_base_url=os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080").rstrip("/"),
            airflow_username=os.getenv("AIRFLOW_USERNAME", "admin"),
            airflow_password=os.getenv("AIRFLOW_PASSWORD", ""),
            approved_dags=_csv("APPROVED_DAGS", "training_pipeline"),
            k8s_allowed_namespaces=ns,
            k8s_default_namespace=os.getenv("K8S_NAMESPACE", ns[0] if ns else "mlops-agent"),
            k8s_deployment=os.getenv("K8S_DEPLOYMENT", "ml-inference"),
            k8s_label_selector=os.getenv("K8S_LABEL_SELECTOR", "app=ml-inference"),
            dvc_default_dataset=os.getenv("DVC_DEFAULT_DATASET", "data/raw/dataset.csv"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            agent_model=os.getenv("AGENT_MODEL", "claude-sonnet-4-6"),
            tool_timeout_s=float(os.getenv("TOOL_TIMEOUT_S", "30")),
            min_f1=float(os.getenv("MIN_F1", "0.80")),
            min_accuracy=float(os.getenv("MIN_ACCURACY", "0.80")),
            model_max_age_days=_opt_int("MODEL_MAX_AGE_DAYS"),
            agent_api_token=os.getenv("AGENT_API_TOKEN", ""),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", ""),
        )
