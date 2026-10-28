# ruff: noqa: INP001
"""Fail loudly when the installed MindRoom runtime lacks the plugin's typed contract.

The plugin uses direct, typed access to MindRoom and agno APIs — no
defensive probing. These assertions turn runtime version skew into a red
test at vendor-bump time instead of silently degraded behavior at dispatch
time.
"""

from __future__ import annotations

from agno.tools import Toolkit as AgnoToolkit
from agno.tools.function import ToolResult
from agno.tools.serper import SerperTools

from mindroom.config.entity_view import ResolvedEntityView
from mindroom.config.main import Config
from mindroom.plugin_api import Toolkit, ToolRuntimeContext, ToolValue, toolkit_functions


def test_runtime_provides_typed_worker_target_api() -> None:
    """MindRoom: worker-target resolution and public scope accessor."""
    assert callable(ToolRuntimeContext.resolve_worker_target)
    assert callable(Config.resolve_entity)
    assert isinstance(ResolvedEntityView.execution_scope, property)
    assert callable(Config.get_agent)
    assert callable(Config.get_worker_grantable_credentials)


def test_toolkit_and_result_contracts() -> None:
    """MindRoom's toolkit_functions calls owned and Agno toolkits; ToolResult and ToolValue carry string content."""

    def search(query: str) -> str:
        return query

    assert toolkit_functions(Toolkit(name="contract-check", tools=[search]))["search"]("x") == "x"
    assert toolkit_functions(AgnoToolkit(name="contract-check", tools=[search]))["search"]("x") == "x"
    assert ToolResult(content="x").content == "x"
    assert ToolValue(content="x").content == "x"
    assert SerperTools(api_key="k").api_key == "k"
