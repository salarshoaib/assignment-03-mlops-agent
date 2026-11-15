"""Tool abstraction shared by all four integration modules."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


class ToolError(Exception):
    error_type = "internal_error"


class ToolInputError(ToolError):
    error_type = "validation_error"


class PermissionDenied(ToolError):
    error_type = "permission_denied"


class ServiceUnavailable(ToolError):
    error_type = "service_unavailable"


@dataclass
class Tool:
    name: str
    description: str
    module: str  # dvc | mlflow | airflow | kubernetes
    input_model: type[BaseModel]
    func: Callable[[Any], dict[str, Any]]
    mutating: bool = False  # True => requires explicit human confirmation
    timeout_s: float | None = None
    precheck: Callable[[Any], None] | None = None  # authorization checks run BEFORE confirmation

    def spec(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        suffix = " [MUTATING: requires explicit user confirmation]" if self.mutating else " [read-only]"
        return {"name": self.name, "description": self.description + suffix, "input_schema": schema}
