"""Pydantic schemas: structured tool inputs/outputs and API payloads."""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_REV_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./@^~-]{0,120}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- results
class ToolResult(BaseModel):
    ok: bool
    tool: str
    data: dict[str, Any] | None = None
    error: str | None = None
    # validation_error | unknown_tool | permission_denied | confirmation_required |
    # timeout | service_unavailable | internal_error
    error_type: str | None = None
    duration_ms: float = 0.0
    request_id: str = ""


class ToolTrace(BaseModel):
    tool: str
    arguments: dict[str, Any]
    ok: bool
    error_type: str | None = None
    duration_ms: float = 0.0


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)


class ConfirmRequest(StrictModel):
    pending_id: str = Field(min_length=4, max_length=64)
    session_id: str | None = None


class AgentResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    tool_calls: list[ToolTrace] = []
    pending_actions: list[dict[str, Any]] = []


# --------------------------------------------------------------------------- DVC inputs
def _check_rev(v: str | None) -> str | None:
    if v is not None and (not _REV_RE.match(v) or ".." in v):
        raise ValueError("invalid git revision")
    return v


class DvcDatasetInput(StrictModel):
    dataset_path: str | None = Field(default=None, max_length=200)


class DvcCompareInput(StrictModel):
    dataset_path: str | None = Field(default=None, max_length=200)
    rev_a: str
    rev_b: str

    _v = field_validator("rev_a", "rev_b")(_check_rev)


class DvcReproduceInput(StrictModel):
    dataset_path: str | None = Field(default=None, max_length=200)
    rev: str | None = None
    run_id: str | None = Field(default=None, max_length=64)

    _v = field_validator("rev")(_check_rev)


class DvcRunInput(StrictModel):
    run_id: str | None = Field(default=None, max_length=64)
    run_name: str | None = Field(default=None, max_length=128)


class DvcRestoreInput(StrictModel):
    dataset_path: str | None = Field(default=None, max_length=200)
    rev: str

    _v = field_validator("rev")(_check_rev)


class EmptyInput(StrictModel):
    pass


# --------------------------------------------------------------------------- MLflow inputs
class MlListExperimentsInput(StrictModel):
    max_results: int = Field(default=20, ge=1, le=100)


class MlListRunsInput(StrictModel):
    experiment_name: str | None = None
    max_results: int = Field(default=10, ge=1, le=50)


class MlRunInput(StrictModel):
    run_id: str = Field(min_length=4, max_length=64)


class MlBestRunInput(StrictModel):
    experiment_name: str | None = None
    metric: str = Field(default="f1", pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    mode: Literal["max", "min"] = "max"
    recent_n: int = Field(default=10, ge=1, le=100)


class MlCompareRunsInput(StrictModel):
    run_ids: list[str] = Field(min_length=2, max_length=10)
    metrics: list[str] | None = None


class MlCompareExperimentsInput(StrictModel):
    experiment_names: list[str] = Field(min_length=2, max_length=5)
    metric: str = Field(default="f1", pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    mode: Literal["max", "min"] = "max"


class MlRegistryInput(StrictModel):
    model_name: str | None = None


# --------------------------------------------------------------------------- Airflow inputs
class AfListDagsInput(StrictModel):
    limit: int = Field(default=50, ge=1, le=200)


class AfDagRunsInput(StrictModel):
    dag_id: str = Field(pattern=_NAME_RE.pattern)
    limit: int = Field(default=5, ge=1, le=25)
    dag_run_id: str | None = None


class AfTriggerInput(StrictModel):
    dag_id: str = Field(pattern=_NAME_RE.pattern)
    conf: dict[str, Any] = Field(default_factory=dict)


class AfRunRefInput(StrictModel):
    dag_id: str = Field(pattern=_NAME_RE.pattern)
    dag_run_id: str | None = None


class AfTaskLogsInput(StrictModel):
    dag_id: str = Field(pattern=_NAME_RE.pattern)
    dag_run_id: str
    task_id: str = Field(pattern=_NAME_RE.pattern)
    try_number: int = Field(default=1, ge=1, le=20)
    tail_lines: int = Field(default=80, ge=5, le=300)


class AfWaitInput(StrictModel):
    dag_id: str = Field(pattern=_NAME_RE.pattern)
    dag_run_id: str
    timeout_s: int = Field(default=120, ge=5, le=300)
    poll_s: int = Field(default=5, ge=1, le=30)


# --------------------------------------------------------------------------- Kubernetes inputs
class K8sNamespaceInput(StrictModel):
    namespace: str | None = None


class K8sPodsInput(StrictModel):
    namespace: str | None = None
    label_selector: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_./=,!() -]{1,200}$")


class K8sPodNameInput(StrictModel):
    pod_name: str = Field(pattern=_NAME_RE.pattern)
    namespace: str | None = None


class K8sPodLogsInput(StrictModel):
    pod_name: str = Field(pattern=_NAME_RE.pattern)
    namespace: str | None = None
    tail_lines: int = Field(default=100, ge=1, le=500)
    previous: bool = False
    container: str | None = None


class K8sEventsInput(StrictModel):
    namespace: str | None = None
    object_name: str | None = None
    limit: int = Field(default=30, ge=1, le=100)


class K8sDeploymentInput(StrictModel):
    deployment_name: str | None = Field(default=None, pattern=_NAME_RE.pattern)
    namespace: str | None = None


class K8sScaleInput(StrictModel):
    deployment_name: str | None = Field(default=None, pattern=_NAME_RE.pattern)
    namespace: str | None = None
    replicas: int = Field(ge=1, le=5)
