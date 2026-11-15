"""Tool package: build_registry() wires all four integrations into one registry."""
from __future__ import annotations

from agent.config import Settings
from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools
from agent.tools.registry import ToolRegistry


def build_registry(settings: Settings, *, mlflow_client=None, dvc_runner=None, airflow_client=None,
                   core_api=None, apps_api=None) -> ToolRegistry:
    tools = (
        dvc_tools.build_tools(settings, dvc_runner, mlflow_client)
        + mlflow_tools.build_tools(settings, mlflow_client)
        + airflow_tools.build_tools(settings, airflow_client)
        + kubernetes_tools.build_tools(settings, core_api, apps_api)
    )
    return ToolRegistry(settings, tools)
