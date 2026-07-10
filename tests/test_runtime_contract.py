# ruff: noqa: INP001
"""Fail loudly when the installed MindRoom runtime lacks the plugin's typed contract.

The plugin uses direct, typed access to MindRoom and agno APIs — no
defensive probing. These assertions turn runtime version skew into a red
test at vendor-bump time instead of silently degraded behavior at dispatch
time. Minimum runtime: MindRoom v2026.7.99.
"""

from __future__ import annotations

from agno.tools import Toolkit
from agno.tools.function import ToolResult
from agno.tools.serper import SerperTools

from mindroom.config.entity_view import ResolvedEntityView
from mindroom.config.main import Config
from mindroom.tool_system.runtime_context import ToolRuntimeContext


def test_runtime_provides_typed_worker_target_api() -> None:
    """MindRoom >= v2026.7.99: worker-target resolution and public scope accessor."""
    assert callable(ToolRuntimeContext.resolve_worker_target)
    assert callable(Config.resolve_entity)
    assert isinstance(ResolvedEntityView.execution_scope, property)
    assert callable(Config.get_agent)
    assert callable(Config.get_worker_grantable_credentials)


def test_toolkit_and_result_contracts() -> None:
    """agno: function registries are dicts, ToolResult carries string content."""
    toolkit = Toolkit(name="contract-check")
    assert isinstance(toolkit.functions, dict)
    assert isinstance(toolkit.async_functions, dict)
    assert ToolResult(content="x").content == "x"
    assert SerperTools(api_key="k").api_key == "k"
