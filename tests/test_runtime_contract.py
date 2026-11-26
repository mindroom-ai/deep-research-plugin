# ruff: noqa: INP001
"""Fail loudly when the installed MindRoom runtime lacks the plugin's typed contract.

The plugin uses direct, typed access to MindRoom APIs — no defensive probing.
These assertions turn runtime version skew into a red test at vendor-bump time
instead of silently degraded behavior at dispatch time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mindroom.config.main import Config
from mindroom.constants import resolve_primary_runtime_paths
from mindroom.plugin_api import (
    ModelScope,
    Toolkit,
    ToolRuntimeContext,
    ToolValue,
    config_plugins,
    get_tool_by_name,
    resolve_turn_model,
    tool,
    toolkit_functions,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_runtime_provides_typed_worker_target_api() -> None:
    """MindRoom: worker-target resolution, turn model resolution, and agent lookup."""
    assert callable(ToolRuntimeContext.resolve_worker_target)
    assert callable(resolve_turn_model)
    assert ModelScope(entity="code").entity == "code"
    assert callable(Config.get_agent)
    assert callable(Config.get_worker_grantable_credentials)


def test_toolkit_and_result_contracts(tmp_path: Path) -> None:
    """MindRoom's toolkit_functions calls owned and registered toolkits; ToolValue carries string content."""

    def search(query: str) -> str:
        return query

    class DecoratedSearch(Toolkit):
        name = "decorated-check"

        @tool
        def run_search(self, query: str) -> str:
            return query

    assert toolkit_functions(Toolkit(name="contract-check", tools=[search]))["search"]("x") == "x"
    assert toolkit_functions(DecoratedSearch())["run_search"]("x") == "x"
    assert ToolValue(content="x").content == "x"

    runtime_paths = resolve_primary_runtime_paths(config_path=tmp_path / "config.yaml")
    plugins = config_plugins(Config(), runtime_paths)
    serper = get_tool_by_name(
        "serper",
        runtime_paths,
        tools=plugins.tools,
        credential_overrides={"api_key": "k"},
        worker_target=None,
    )
    assert getattr(serper, "api_key", None) == "k"
    assert {"search_web", "search_news", "search_scholar"} <= set(toolkit_functions(serper))
