# ruff: noqa: INP001
# ruff: noqa: ANN401, ARG002, D103
"""Tests for the deep-research MindRoom tool wrapper."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
from importlib import util
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from agno.tools import Toolkit as AgnoToolkit
from agno.tools.function import ToolResult
from agno.tools.serper import SerperTools

from mindroom.config.agent import AgentConfig
from mindroom.config.main import Config, load_config
from mindroom.config.models import DefaultsConfig
from mindroom.constants import resolve_primary_runtime_paths
from mindroom.message_target import MessageTarget
from mindroom.plugin_api import HookBindings, Toolkit, tool, toolkit_functions
from mindroom.tool_system.declarations import SetupType, ToolStatus
from mindroom.tool_system.metadata import get_tool_by_name as resolve_tool_by_name
from mindroom.tool_system.plugins import PluginGeneration, config_plugins, prepare_plugins, publish_plugins
from mindroom.tool_system.registry import plugin_module_registrations
from mindroom.tool_system.runtime_context import ToolRuntimeContext, tool_runtime_context

if TYPE_CHECKING:
    from types import ModuleType

PACKAGE_NAME = f"mindroom_plugin_{Path(__file__).resolve().parents[1].name.replace('-', '_')}"


def _load_tools_module() -> ModuleType:
    tools_path = Path(__file__).resolve().parents[1] / "tools.py"
    module_name = f"{PACKAGE_NAME}.tools_test_{uuid4().hex}"
    sys.modules.pop(module_name, None)
    spec = util.spec_from_file_location(module_name, tools_path)
    assert spec is not None
    assert spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    # A plugin module registers its tools while it loads, as the plugin loader runs it.
    with plugin_module_registrations(module_name):
        spec.loader.exec_module(module)
    return module


def _load_loop_module() -> ModuleType:
    return importlib.import_module(f"{PACKAGE_NAME}.loop")


def _tool_context(
    *,
    sender: AsyncMock | None = None,
    active_model_name: str | None = "active",
) -> ToolRuntimeContext:
    agent = AgentConfig(display_name="Code")
    config = SimpleNamespace(
        models={"active": object(), "override": object(), "resolved": object()},
        debug=SimpleNamespace(log_llm_requests=False),
        agents={"code": agent},
        defaults=DefaultsConfig(),
        get_agent=lambda _name, _agent=agent: _agent,
        get_worker_grantable_credentials=lambda: frozenset(),
    )
    return ToolRuntimeContext(
        agent_name="code",
        target=MessageTarget(
            room_id="!room:localhost",
            source_thread_id="$thread-root",
            resolved_thread_id="$thread-root",
            reply_to_event_id=None,
            session_id="session-1",
        ),
        requester_id="@user:localhost",
        client=AsyncMock(),
        config=config,
        runtime_paths=SimpleNamespace(env_value=lambda _name: None),
        plugins=SimpleNamespace(tools=object()),
        conversation_reader=AsyncMock(),
        relations=AsyncMock(),
        active_model_name=active_model_name,
        hooks=HookBindings(message_sender=sender),
        correlation_id="corr-1",
    )


class _FakeSerper(AgnoToolkit):
    """A vendor toolkit, as Agno's Serper is; a subclass's overrides are its functions."""

    api_key = "key"

    def __init__(self) -> None:
        super().__init__(
            name="serper",
            tools=[self.search_web, self.search_news, self.search_scholar, self.scrape_webpage],
        )

    def search_web(self, _query: str, num_results: int | None = None) -> str:
        return json.dumps({"organic": []})

    def search_news(self, _query: str, num_results: int | None = None) -> str:
        return json.dumps({"news": []})

    def search_scholar(self, _query: str, num_results: int | None = None) -> str:
        return json.dumps({"organic": []})

    def scrape_webpage(self, _url: str) -> str:
        return json.dumps({"text": "fallback text"})


class _FakeWebsite(Toolkit):
    """An owned toolkit, as MindRoom's website reader is."""

    @tool
    def read_url(self, _url: str) -> str:
        return json.dumps([{"content": "website text long enough" * 40, "meta_data": {"title": "Title"}}])


def _tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
    if name == "serper":
        return _FakeSerper()
    if name == "website":
        return _FakeWebsite()
    raise AssertionError(name)


def _result(module: ModuleType, *, rounds_used: int = 1) -> Any:
    del module
    return _load_loop_module().LoopResult(
        question="What?",
        report="Answer [1]\n\n## Sources\n[1] T - https://example.com",
        sources=[{"id": 1, "url": "https://example.com", "title": "T", "snippet": "S"}],
        sources_considered=2,
        sources_used=1,
        confidence=0.82,
        rounds_used=rounds_used,
        stopped_reason="confident",
        elapsed_seconds=1.5,
        warnings=[],
        stats={"searches": 3, "reads": 1},
    )


async def _await(value: Any) -> Any:
    return value


def test_plugin_discovery_registers_tool_and_builtin_dependencies_resolve(tmp_path: Path) -> None:
    runtime_paths = resolve_primary_runtime_paths(config_path=tmp_path / "config.yaml")
    config = Config(plugins=[{"path": str(Path(__file__).resolve().parents[1])}])
    generation = prepare_plugins(config, runtime_paths)
    assert not generation.skipped
    # Publishing replaces every installed plugin module, the package conftest.py bootstraps included.
    bootstrapped = {name: module for name, module in sys.modules.items() if name.startswith(PACKAGE_NAME)}

    publish_plugins(generation)
    try:
        plugins = config_plugins(config, runtime_paths)
        with patch("mindroom.tool_system.metadata.ensure_tool_deps", return_value=False) as ensure_deps:
            deep_research = resolve_tool_by_name(
                "deep_research",
                runtime_paths,
                tools=plugins.tools,
                worker_target=None,
            )
            serper = resolve_tool_by_name("serper", runtime_paths, tools=plugins.tools, worker_target=None)
            website = resolve_tool_by_name("website", runtime_paths, tools=plugins.tools, worker_target=None)
        deep_research_metadata = plugins.tools.entries["deep_research"]
    finally:
        publish_plugins(PluginGeneration())
        sys.modules.update(bootstrapped)

    assert [plugin.name for plugin in plugins.loaded_plugins] == ["deep-research"]
    assert type(deep_research).__name__ == "DeepResearchTools"
    assert deep_research_metadata.status is ToolStatus.REQUIRES_CONFIG
    assert deep_research_metadata.setup_type is SetupType.API_KEY
    checked_dependency_sets = {tuple(call.args[0]) for call in ensure_deps.call_args_list}
    assert ("requests",) in checked_dependency_sets
    assert ("httpx", "beautifulsoup4") in checked_dependency_sets
    assert {"search_web", "search_news", "search_scholar"} <= set(toolkit_functions(serper))
    assert "read_url" in toolkit_functions(website)


@pytest.mark.asyncio
async def test_missing_runtime_context_returns_error_envelope() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    with tool_runtime_context(None):
        result = json.loads(await tools.deep_research("question"))

    assert result == {
        "message": "Deep research tool context is unavailable in this runtime path.",
        "status": "error",
        "tool": "deep_research",
        "warnings": [],
    }


@pytest.mark.asyncio
async def test_happy_path_payload_is_valid_sorted_json() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    sender = AsyncMock()

    with (
        tool_runtime_context(_tool_context(sender=sender)),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        raw = await tools.deep_research("What?")

    assert list(json.loads(raw)) == sorted(json.loads(raw))
    result = json.loads(raw)
    assert result["status"] == "ok"
    assert result["tool"] == "deep_research"
    for key in (
        "question",
        "report",
        "sources",
        "sources_considered",
        "sources_used",
        "confidence",
        "rounds_used",
        "stopped_reason",
        "elapsed_seconds",
        "warnings",
        "stats",
    ):
        assert key in result


@pytest.mark.asyncio
async def test_model_override_precedence_and_execution_identity_are_wired() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    context = _tool_context(sender=AsyncMock())

    with (
        tool_runtime_context(context),
        patch.object(module, "build_model_provider", return_value=object()) as build_provider,
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
        patch.object(module, "resolve_turn_model") as resolve_model,
    ):
        result = json.loads(await tools.deep_research("What?", model="override"))

    assert result["status"] == "ok"
    build_provider.assert_called_once_with(
        context.config,
        context.runtime_paths,
        "override",
        execution_identity=context.execution_identity,
    )
    resolve_model.assert_not_called()


@pytest.mark.asyncio
async def test_active_model_precedes_runtime_resolution() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    context = _tool_context(sender=AsyncMock(), active_model_name="active")

    with (
        tool_runtime_context(context),
        patch.object(module, "build_model_provider", return_value=object()) as build_provider,
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
        patch.object(module, "resolve_turn_model") as resolve_model,
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert build_provider.call_args.args[2] == "active"
    resolve_model.assert_not_called()


@pytest.mark.asyncio
async def test_runtime_model_resolution_fallback_when_no_active_model() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    context = _tool_context(sender=AsyncMock(), active_model_name=None)

    with (
        tool_runtime_context(context),
        patch.object(module, "build_model_provider", return_value=object()) as build_provider,
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
        patch.object(
            module,
            "resolve_turn_model",
            return_value=SimpleNamespace(model_name="resolved"),
        ) as resolve_model,
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert build_provider.call_args.args[2] == "resolved"
    resolve_model.assert_called_once_with(
        context.config,
        context.runtime_paths,
        module.ModelScope(entity="code", room_id="!room:localhost", thread_id="$thread-root"),
        None,
    )


@pytest.mark.asyncio
async def test_unknown_model_override_returns_error_envelope() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()) as build_provider,
        patch.object(module, "run_research_loop", AsyncMock()) as loop_mock,
    ):
        result = json.loads(await tools.deep_research("What?", model="bogus"))

    assert result["status"] == "error"
    assert "Unknown model override: bogus" in result["message"]
    build_provider.assert_not_called()
    loop_mock.assert_not_called()


@pytest.mark.asyncio
async def test_extract_model_routes_to_separate_model_instance() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", side_effect=[object(), object()]) as build_provider,
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        result = json.loads(await tools.deep_research("What?", extract_model="override"))

    assert result["status"] == "ok"
    assert [call.args[2] for call in build_provider.call_args_list] == ["active", "override"]


@pytest.mark.asyncio
async def test_unknown_extract_model_returns_error_envelope() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "run_research_loop", AsyncMock()) as loop_mock,
    ):
        result = json.loads(await tools.deep_research("What?", extract_model="bogus"))

    assert result["status"] == "error"
    assert "Unknown model override: bogus" in result["message"]
    loop_mock.assert_not_called()


@pytest.mark.asyncio
async def test_parallel_researchers_dispatches_heavy_mode() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock()) as single_mock,
        patch.object(module, "run_heavy_research_loop", AsyncMock(return_value=_result(module))) as heavy_mock,
    ):
        result = json.loads(await tools.deep_research("What?", parallel_researchers=3))

    assert result["status"] == "ok"
    single_mock.assert_not_called()
    assert heavy_mock.call_args.kwargs["researchers"] == 3


def test_round_progress_includes_researcher_prefix_in_heavy_mode() -> None:
    module = _load_tools_module()
    event = {"round": 2, "max_rounds": 9, "thought": "t", "confidence": 0.3, "researcher": 2}
    assert module._format_round_progress(event, verbose=False).startswith("researcher 2 · round 2/9")


@pytest.mark.asyncio
async def test_ground_fn_is_wired_into_loop_kwargs() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))) as loop_mock,
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert callable(loop_mock.call_args.kwargs["ground_fn"])


@pytest.mark.asyncio
async def test_verbosity_silent_sends_no_progress_messages() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    sender = AsyncMock()

    async def fake_loop(**kwargs: Any) -> Any:
        await kwargs["emit_fn"]({"round": 1, "max_rounds": 1, "thought": "x", "confidence": 0.1})
        return _result(module)

    with (
        tool_runtime_context(_tool_context(sender=sender)),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        result = json.loads(await tools.deep_research("What?", verbosity="silent"))

    assert result["status"] == "ok"
    sender.assert_not_awaited()


@pytest.mark.asyncio
async def test_verbosity_progress_sends_start_round_and_done() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    sender = AsyncMock()

    async def fake_loop(**kwargs: Any) -> Any:
        await kwargs["emit_fn"](
            {
                "round": 1,
                "max_rounds": 3,
                "thought": "short thought",
                "confidence": 0.5,
                "next_action": "finish",
                "search_queries": [],
                "read_urls": [],
            },
        )
        return _result(module)

    with (
        tool_runtime_context(_tool_context(sender=sender)),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        await tools.deep_research("What?", max_rounds=3)

    assert sender.await_count == 3
    sent_texts = [call.args[1] for call in sender.await_args_list]
    assert sent_texts[0].startswith("deep_research started")
    assert sent_texts[1].startswith("round 1/3")
    assert sent_texts[2].startswith("deep_research done")


@pytest.mark.asyncio
async def test_progress_message_failures_do_not_abort_research() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    sender = AsyncMock(side_effect=RuntimeError("matrix down"))

    with (
        tool_runtime_context(_tool_context(sender=sender)),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        result = json.loads(await tools.deep_research("What?", verbosity="progress"))

    assert result["status"] == "ok"
    assert sender.await_count == 2


@pytest.mark.asyncio
async def test_start_and_done_progress_emit_timeouts_do_not_hang_tool() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    async def never_send(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.Event().wait()

    sender = AsyncMock(side_effect=never_send)

    with (
        tool_runtime_context(_tool_context(sender=sender)),
        patch.object(module, "PROGRESS_EMIT_TIMEOUT_SECONDS", 0.01),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        raw = await asyncio.wait_for(tools.deep_research("What?", verbosity="progress"), timeout=0.2)

    result = json.loads(raw)
    assert result["status"] == "ok"
    assert sender.await_count == 2


@pytest.mark.asyncio
async def test_start_and_done_progress_emits_share_wall_clock_budget() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    async def never_send(*_args: Any, **_kwargs: Any) -> None:
        await asyncio.Event().wait()

    sender = AsyncMock(side_effect=never_send)

    with (
        tool_runtime_context(_tool_context(sender=sender)),
        patch.object(module, "clamp_int", side_effect=lambda value, minimum, maximum: value),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        raw = await asyncio.wait_for(
            tools.deep_research("What?", wall_clock_seconds=0.03, verbosity="progress"),
            timeout=0.2,
        )

    result = json.loads(raw)
    assert result["status"] == "ok"
    assert result["elapsed_seconds"] < 0.2
    assert sender.await_count <= 2


@pytest.mark.asyncio
async def test_arg_clamping_before_loop_call() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    captured: dict[str, Any] = {}

    async def fake_loop(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _result(module)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        await tools.deep_research("What?", max_rounds=999, wall_clock_seconds=99999)

    assert captured["max_rounds"] == 100
    assert captured["wall_clock_seconds"] == 9000


@pytest.mark.asyncio
async def test_depth_args_are_forwarded_to_loop_and_page_reader() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    captured: dict[str, Any] = {}

    async def fake_loop(**kwargs: Any) -> Any:
        captured.update(kwargs)
        page = await kwargs["read_fn"]("https://example.com")
        assert len(page.text) == 120_000
        return _result(module)

    class LongWebsite(Toolkit):
        @tool
        def read_url(self, _url: str) -> str:
            return json.dumps([{"content": "x" * 200_000, "meta_data": {"title": "Long"}}])

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "serper":
            return _FakeSerper()
        if name == "website":
            return LongWebsite()
        raise AssertionError(name)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        result = json.loads(
            await tools.deep_research(
                "What?",
                max_queries_per_round=8,
                results_per_query=10,
                max_reads_per_round=12,
                page_char_limit=120_000,
                report_token_cap=12_000,
            ),
        )

    assert result["status"] == "ok"
    assert captured["max_queries_per_round"] == 8
    assert captured["results_per_query"] == 10
    assert captured["max_reads_per_round"] == 12
    assert captured["report_char_cap"] == 48_000


@pytest.mark.asyncio
async def test_tool_resolution_keeps_sandbox_proxy_enabled() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    context = _tool_context(sender=AsyncMock())
    calls: list[dict[str, Any]] = []

    def tool_by_name(name: str, *_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return _tool_by_name(name)

    with (
        tool_runtime_context(context),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert calls
    assert all(call.get("runs_in_worker") is not False for call in calls)
    assert all(call["tools"] is context.plugins.tools for call in calls)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rounds": "many"},
        {"wall_clock_seconds": None},
    ],
)
@pytest.mark.asyncio
async def test_malformed_numeric_args_return_error_envelope(kwargs: dict[str, object]) -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()) as build_provider,
        patch.object(module, "run_research_loop", AsyncMock()) as loop_mock,
    ):
        result = json.loads(await tools.deep_research("What?", **kwargs))

    assert result["status"] == "error"
    build_provider.assert_not_called()
    loop_mock.assert_not_called()


@pytest.mark.asyncio
async def test_search_and_read_call_the_toolkits_functions_not_its_same_named_methods() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    class EntrypointSerper(Toolkit):
        @tool(name="search_web")
        def entrypoint_search_web(self, query: str, num_results: int | None = None) -> str:
            return json.dumps({"organic": [{"link": "https://example.com", "title": query, "snippet": str(num_results)}]})

        def search_web(self, _query: str, num_results: int | None = None) -> str:
            raise AssertionError("direct search method bypassed the toolkit's function")

    class EntrypointWebsite(Toolkit):
        # Async, as a function routed to a worker is.
        @tool(name="read_url")
        async def entrypoint_read_url(self, url: str) -> str:
            return json.dumps([{"content": f"entrypoint text for {url}", "meta_data": {"title": "Entrypoint"}}])

        def read_url(self, _url: str) -> str:
            raise AssertionError("direct read method bypassed the toolkit's function")

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "serper":
            return EntrypointSerper()
        if name == "website":
            return EntrypointWebsite()
        raise AssertionError(name)

    async def fake_loop(**kwargs: Any) -> Any:
        hits = await kwargs["search_fn"](_load_loop_module().SearchQuery(query="entry"), 4)
        page = await kwargs["read_fn"]("https://example.com")
        assert hits[0].title == "entry"
        assert page.title == "Entrypoint"
        assert "entrypoint text" in page.text
        return _result(module)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_serper_error_payload_returns_search_warning() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    class ErrorSerper(_FakeSerper):
        def search_web(self, _query: str, num_results: int | None = None) -> str:
            return json.dumps({"error": "quota exceeded"})

    async def ask_model(_model: Any, _prompt: str, *, schema: Any) -> Any:
        if schema is module.ResearchStep:
            return module.ResearchStep(
                thought="finish",
                updated_report="partial",
                open_questions=[],
                confidence=0.2,
                next_action="finish",
            )
        return "final report"

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "serper":
            return ErrorSerper()
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "_ask_model", ask_model),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
    ):
        result = json.loads(await tools.deep_research("What?", verbosity="silent"))

    assert result["status"] == "error"
    assert result["warnings"] == ["seed search failed: quota exceeded"]


@pytest.mark.asyncio
async def test_website_error_payload_returns_read_warning() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    class ErrorWebsite(Toolkit):
        @tool
        def read_url(self, _url: str) -> str:
            return json.dumps({"status": "error", "message": "fetch failed"})

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "serper":
            return _FakeSerper()
        if name == "website":
            return ErrorWebsite()
        raise AssertionError(name)

    async def fake_loop(**kwargs: Any) -> Any:
        loop = _load_loop_module()
        return await loop.run_research_loop(
            question="What?",
            max_rounds=1,
            wall_clock_seconds=60,
            reason_fn=lambda _prompt: _await(
                loop.ResearchStep(
                    thought="read",
                    updated_report="partial",
                    open_questions=[],
                    confidence=0.1,
                    next_action="read",
                    read_urls=["https://example.com/error"],
                ),
            ),
            extract_fn=lambda _prompt: _await(loop.Extraction(facts=[], relevant=False)),
            search_fn=lambda _query, _limit: _await([]),
            read_fn=kwargs["read_fn"],
            synthesize_fn=lambda _prompt: _await("fallback"),
        )

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        result = json.loads(await tools.deep_research("What?", verbosity="silent"))

    assert result["status"] == "ok"
    assert result["warnings"] == ["read failed for https://example.com/error: fetch failed"]


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://127.0.0.1/admin", "loopback IPv4"),
        ("http://localhost/admin", "localhost"),
        ("http://169.254.169.254/latest/meta-data", "metadata IP"),
        ("http://10.0.0.1/admin", "RFC1918 IPv4"),
        ("http://[::1]/admin", "loopback IPv6"),
        ("http://private.example.test/admin", "DNS private IP"),
        ("https://example.com/redirect-to-localhost", "redirect to internal host"),
    ],
)
@pytest.mark.asyncio
async def test_read_boundary_uses_native_website_hardening_without_serper_fallback(url: str, reason: str) -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    scrape_webpage = Mock(side_effect=AssertionError("serper scrape fallback must not be called"))

    class SearchOnlySerper(_FakeSerper):
        def scrape_webpage(self, scrape_url: str) -> str:
            return scrape_webpage(scrape_url)

    class HardenedWebsite(Toolkit):
        @tool
        def read_url(self, read_url: str) -> str:
            assert read_url == url
            raise RuntimeError(f"blocked by native website hardening: {reason}")

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "serper":
            return SearchOnlySerper()
        if name == "website":
            return HardenedWebsite()
        raise AssertionError(name)

    async def fake_loop(**kwargs: Any) -> Any:
        await kwargs["read_fn"](url)
        return _result(module)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "error"
    assert reason in result["message"]
    scrape_webpage.assert_not_called()


@pytest.mark.asyncio
async def test_serper_unconfigured_returns_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    from agno.tools.serper import SerperTools

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        return SerperTools(api_key=None) if name == "serper" else _FakeWebsite()

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock()) as loop_mock,
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "error"
    assert "Search tool 'serper' is not configured" in result["message"]
    loop_mock.assert_not_called()


def test_parse_search_results_raises_on_status_error_payload() -> None:
    module = _load_tools_module()

    with pytest.raises(RuntimeError, match="rate limited"):
        module._parse_search_results(json.dumps({"status": "error", "message": "rate limited"}))
    with pytest.raises(RuntimeError, match="RATE_LIMIT"):
        module._parse_search_results(json.dumps({"status": "error", "code": "RATE_LIMIT"}))


@pytest.mark.asyncio
async def test_custom_search_backend_serves_all_query_kinds() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools(search_tool="grounded_search", search_function="search_public_web")
    queries: list[str] = []

    class GroundedSearch(Toolkit):
        @tool
        def search_public_web(self, query: str) -> str:
            queries.append(query)
            return json.dumps(
                {
                    "status": "ok",
                    "answer": "grounded answer",
                    "sources": [{"uri": "https://example.com/a", "title": "A", "domain": "example.com"}],
                },
            )

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "grounded_search":
            return GroundedSearch()
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    async def fake_loop(**kwargs: Any) -> Any:
        loop = _load_loop_module()
        for kind in ("web", "news", "scholar"):
            hits = await kwargs["search_fn"](loop.SearchQuery(query=f"{kind} q", kind=kind), 5)
            assert hits[0].url == "https://example.com/a"
            assert hits[0].title == "A"
            assert hits[0].snippet == "example.com"
        return _result(module)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert queries == ["web q", "news q", "scholar q"]


@pytest.mark.asyncio
async def test_custom_async_search_function_is_awaited() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools(search_tool="grounded_search", search_function="search_public_web")

    class AsyncGroundedSearch(Toolkit):
        @tool
        async def search_public_web(self, query: str) -> str:
            return json.dumps({"sources": [{"uri": "https://example.com/async", "title": query}]})

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "grounded_search":
            return AsyncGroundedSearch()
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    async def fake_loop(**kwargs: Any) -> Any:
        hits = await kwargs["search_fn"](_load_loop_module().SearchQuery(query="async q"), 5)
        assert hits[0].url == "https://example.com/async"
        assert hits[0].title == "async q"
        return _result(module)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_search_tool_resolution_uses_calling_agents_authored_overrides() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools(search_tool="grounded_search", search_function="search_public_web")
    context = _tool_context(sender=AsyncMock())
    context.config.agents["code"].tools = [
        {"grounded_search": {"project_id": "demo-project"}},
    ]
    calls: list[tuple[str, Any]] = []

    class GroundedSearch(Toolkit):
        @tool
        def search_public_web(self, _query: str) -> str:
            return json.dumps({"sources": []})

    def tool_by_name(name: str, *_args: Any, **kwargs: Any) -> Any:
        calls.append((name, kwargs.get("tool_config_overrides")))
        if name == "grounded_search":
            return GroundedSearch()
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    with (
        tool_runtime_context(context),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert ("grounded_search", {"project_id": "demo-project"}) in calls


def test_authored_tool_overrides_reads_typed_agent_entries() -> None:
    module = _load_tools_module()
    agent = SimpleNamespace(
        tools=[
            SimpleNamespace(name="object_search", overrides={"project_id": "object-project"}),
            SimpleNamespace(name="other_tool", overrides={}),
        ],
    )
    context = SimpleNamespace(agent_name="code", config=SimpleNamespace(get_agent=lambda _name: agent))

    assert module._authored_tool_overrides(context, "object_search") == {"project_id": "object-project"}
    assert module._authored_tool_overrides(context, "unlisted_search") == {}


def test_authored_tool_overrides_leave_out_the_agents_function_toggles() -> None:
    module = _load_tools_module()
    overrides = {
        "all": False,
        "enable_search": False,
        "enable_search_news": False,
        "enable_search_scholar": True,
        "include_tools": ["search_scholar"],
        "exclude_tools": ["search_web"],
        "num_results": 10,
        "location": "nl",
    }
    agent = SimpleNamespace(tools=[SimpleNamespace(name="serper", overrides=overrides)])
    context = SimpleNamespace(agent_name="code", config=SimpleNamespace(get_agent=lambda _name: agent))

    assert module._authored_tool_overrides(context, "serper") == {
        "enable_search_scholar": True,
        "num_results": 10,
        "location": "nl",
    }


def test_authored_tool_overrides_keep_an_enable_for_a_function_that_defaults_off() -> None:
    module = _load_tools_module()
    overrides = {"enable_search": True, "enable_crawl": False, "all": True}
    agent = SimpleNamespace(tools=[SimpleNamespace(name="firecrawl", overrides=overrides)])
    context = SimpleNamespace(agent_name="code", config=SimpleNamespace(get_agent=lambda _name: agent))

    assert module._authored_tool_overrides(context, "firecrawl") == {"enable_search": True, "all": True}


def test_search_toolkit_keeps_the_searches_an_agent_hid_from_itself() -> None:
    module = _load_tools_module()
    hidden = {"enable_search": False, "enable_search_news": False, "enable_scrape_webpage": False, "num_results": 7}
    agent = SimpleNamespace(tools=[SimpleNamespace(name="serper", overrides=hidden)])
    context = SimpleNamespace(agent_name="code", config=SimpleNamespace(get_agent=lambda _name: agent))

    toolkit = SerperTools(api_key="key", **module._authored_tool_overrides(context, "serper"))

    assert {"search_web", "search_news", "search_scholar"} <= set(toolkit_functions(toolkit))
    assert toolkit.num_results == 7


def _serper_with(*names: str) -> Toolkit:
    serper = _FakeSerper()
    return Toolkit(name="serper", tools=[getattr(serper, name) for name in names])


@pytest.mark.asyncio
async def test_a_search_tool_that_cannot_search_the_web_fails_before_research() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    loop = AsyncMock(return_value=_result(module))

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "serper":
            return _serper_with("search_scholar")
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", loop),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "error"
    assert "search_web" in result["message"]
    loop.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_search_kind_the_tool_lacks_is_dropped_and_falls_back_to_web() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    loop = AsyncMock(return_value=_result(module))
    web_searches: list[str] = []

    def search_web(query: str, num_results: int | None = None) -> str:
        web_searches.append(query)
        return json.dumps({"organic": []})

    serper = Toolkit(name="serper", tools=[search_web, _FakeSerper().search_scholar])

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "serper":
            return serper
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", loop),
    ):
        result = json.loads(await tools.deep_research("What?"))
        search_fn = loop.await_args.kwargs["search_fn"]
        await search_fn(_load_loop_module().SearchQuery(query="latest", kind="news"), 3)

    assert result["status"] == "ok"
    assert [name for name, _description in loop.await_args.kwargs["search_channels"]] == ["web", "scholar"]
    assert any("news" in warning and "search_news" in warning for warning in result["warnings"])
    assert web_searches == ["latest"]


@pytest.mark.asyncio
async def test_a_run_whose_every_search_failed_and_found_no_sources_is_an_error() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    failed = _result(module)
    failed.sources = []
    failed.sources_used = 0
    failed.warnings = ["seed search failed: quota exceeded"]
    failed.stats = {"searches": 2, "search_attempts": 3, "search_failures": 2, "reads": 0}

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=failed)),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "error"
    assert result["message"] == "Every search failed, so deep_research found no sources."
    assert result["warnings"] == ["seed search failed: quota exceeded"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failures", "has_sources"),
    [(2, False), (3, True)],
    ids=["a search worked but found nothing", "every search failed but a read found a source"],
)
async def test_a_run_with_a_working_search_or_a_source_stays_ok(failures: int, has_sources: bool) -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    partial = _result(module)
    if not has_sources:
        partial.sources = []
        partial.sources_used = 0
    partial.stats = {"searches": 3, "search_attempts": 3, "search_failures": failures, "reads": 1}

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=partial)),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"


@pytest.mark.skipif(
    os.getenv("MINDROOM_DEEP_RESEARCH_INTEGRATION") != "1",
    reason="set MINDROOM_DEEP_RESEARCH_INTEGRATION=1 for live MindRoom integration smoke test",
)
@pytest.mark.asyncio
async def test_integration_smoke_live_research() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    config_path = os.getenv("MINDROOM_DEEP_RESEARCH_CONFIG_PATH")
    runtime_paths = resolve_primary_runtime_paths(config_path=Path(config_path) if config_path else None)
    config = load_config(runtime_paths, tolerate_plugin_load_errors=True)
    model_name = os.getenv("MINDROOM_DEEP_RESEARCH_MODEL") or next(iter(config.models), None)
    if model_name is None:
        pytest.skip("no MindRoom models configured")
    context = ToolRuntimeContext(
        agent_name=os.getenv("MINDROOM_DEEP_RESEARCH_AGENT", "code"),
        target=MessageTarget(
            room_id="!deep-research-integration:localhost",
            source_thread_id="$deep-research-integration",
            resolved_thread_id="$deep-research-integration",
            reply_to_event_id=None,
            session_id="deep-research-integration",
        ),
        requester_id="@deep-research-integration:localhost",
        client=AsyncMock(),
        config=config,
        runtime_paths=runtime_paths,
        plugins=config_plugins(config, runtime_paths),
        conversation_reader=AsyncMock(),
        relations=AsyncMock(),
        active_model_name=model_name,
        correlation_id="deep-research-integration",
    )

    with tool_runtime_context(context):
        result = json.loads(
            await tools.deep_research(
                "What is Serper's search API used for? Answer in two sentences.",
                max_rounds=2,
                wall_clock_seconds=120,
                verbosity="silent",
                model=model_name,
            ),
        )
    if result["status"] == "error" and "Serper" in result["message"]:
        pytest.skip(result["message"])
    assert result["status"] == "ok"
    assert result["tool"] == "deep_research"
    assert result["sources"]
    assert "[" in result["report"]


def test_parse_search_channels_accepts_dicts_and_compact_strings() -> None:
    module = _load_tools_module()

    channels = module._parse_search_channels(
        [
            {"name": " Wiki ", "tool": "wiki_tool", "function": "search_documents", "description": "Internal wiki"},
            "chat=chat_tool.search_messages|Team chat history",
            "web=shadow_tool.search",
            {"name": "wiki", "tool": "duplicate", "function": "search"},
            {"name": "broken", "tool": "", "function": "search"},
            42,
        ],
    )

    assert [channel.model_dump() for channel in channels] == [
        {
            "name": "wiki",
            "tool": "wiki_tool",
            "function": "search_documents",
            "description": "Internal wiki",
            "arguments": None,
        },
        {
            "name": "chat",
            "tool": "chat_tool",
            "function": "search_messages",
            "description": "Team chat history",
            "arguments": None,
        },
    ]


def test_parse_search_results_accepts_top_level_lists_and_context_snippets() -> None:
    module = _load_tools_module()

    hits = module._parse_search_results(
        json.dumps(
            [
                {"url": "https://wiki.example/doc/runbook", "title": "Runbook", "context": "restart the frobnicator"},
                {"permalink": "https://chat.example/msg/1", "text": "x" * 1000},
                {"title": "no url row"},
            ],
        ),
    )

    assert [hit.url for hit in hits] == ["https://wiki.example/doc/runbook", "https://chat.example/msg/1"]
    assert hits[0].snippet == "restart the frobnicator"
    assert len(hits[1].snippet) == module.HIT_SNIPPET_CHAR_LIMIT


@pytest.mark.asyncio
async def test_search_channels_route_by_kind_and_advertise_to_reasoner() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools(
        search_tool="grounded_search",
        search_function="search_public_web",
        search_channels=[
            {"name": "wiki", "tool": "wiki_tool", "function": "search_documents", "description": "Internal wiki"},
        ],
    )
    channel_queries: list[str] = []
    web_queries: list[str] = []

    class GroundedSearch(Toolkit):
        @tool
        def search_public_web(self, query: str) -> str:
            web_queries.append(query)
            return json.dumps({"sources": [{"uri": "https://web.example/a", "title": "W"}]})

    class WikiTool(Toolkit):
        @tool
        def search_documents(self, query: str) -> str:
            channel_queries.append(query)
            return json.dumps([{"url": "https://wiki.example/doc/1", "title": "Doc", "context": "wiki fact"}])

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "grounded_search":
            return GroundedSearch()
        if name == "wiki_tool":
            return WikiTool()
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    async def fake_loop(**kwargs: Any) -> Any:
        loop = _load_loop_module()
        assert ("wiki", "Internal wiki") in kwargs["search_channels"]
        assert kwargs["search_channels"][0][0] == "web"
        wiki_hits = await kwargs["search_fn"](loop.SearchQuery(query="wiki q", kind="wiki"), 5)
        assert wiki_hits[0].url == "https://wiki.example/doc/1"
        assert wiki_hits[0].snippet == "wiki fact"
        web_hits = await kwargs["search_fn"](loop.SearchQuery(query="web q", kind="web"), 5)
        assert web_hits[0].url == "https://web.example/a"
        unknown_hits = await kwargs["search_fn"](loop.SearchQuery(query="fallback q", kind="mystery"), 5)
        assert unknown_hits[0].url == "https://web.example/a"
        return _result(module)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert channel_queries == ["wiki q"]
    assert web_queries == ["web q", "fallback q"]


@pytest.mark.asyncio
async def test_unavailable_search_channel_degrades_with_warning() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools(
        search_tool="grounded_search",
        search_function="search_public_web",
        search_channels=["wiki=missing_tool.search|Internal wiki"],
    )

    class GroundedSearch(Toolkit):
        @tool
        def search_public_web(self, _query: str) -> str:
            return json.dumps({"sources": []})

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "grounded_search":
            return GroundedSearch()
        if name == "missing_tool":
            raise RuntimeError("not registered")
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    async def fake_loop(**kwargs: Any) -> Any:
        assert all(name != "wiki" for name, _description in kwargs["search_channels"])
        return _result(module)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert any("search channel 'wiki' unavailable" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_search_channel_tools_resolve_with_agent_authored_overrides() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools(
        search_tool="grounded_search",
        search_function="search_public_web",
        search_channels=[{"name": "wiki", "tool": "wiki_tool", "function": "search_documents"}],
    )
    context = _tool_context(sender=AsyncMock())
    context.config.agents["code"].tools = [
        {"wiki_tool": {"collection": "engineering"}},
    ]
    calls: list[tuple[str, Any]] = []

    class GroundedSearch(Toolkit):
        @tool
        def search_public_web(self, _query: str) -> str:
            return json.dumps({"sources": []})

    class WikiTool(Toolkit):
        @tool
        def search_documents(self, _query: str) -> str:
            return json.dumps([])

    def tool_by_name(name: str, *_args: Any, **kwargs: Any) -> Any:
        calls.append((name, kwargs.get("tool_config_overrides")))
        if name == "grounded_search":
            return GroundedSearch()
        if name == "wiki_tool":
            return WikiTool()
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    with (
        tool_runtime_context(context),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert ("wiki_tool", {"collection": "engineering"}) in calls


def test_parse_search_results_skips_non_string_values_in_key_chains() -> None:
    module = _load_tools_module()

    hits = module._parse_search_results(
        json.dumps(
            {
                "results": [
                    {
                        "link": {"metadata": "not a url"},
                        "url": "https://real.example/a",
                        "title": ["not", "a", "title"],
                        "name": "Real Title",
                        "snippet": 42,
                        "description": "real snippet",
                    },
                    {"link": {"only": "junk"}},
                ],
            },
        ),
    )

    assert len(hits) == 1
    assert hits[0].url == "https://real.example/a"
    assert hits[0].title == "Real Title"
    assert hits[0].snippet == "real snippet"


def test_substitute_placeholders_preserves_types_and_nests() -> None:
    module = _load_tools_module()

    substituted = module._substitute_placeholders(
        {
            "tool_name": "search_documents",
            "arguments": {"query": "{query}", "limit": "{num_results}", "note": "q={query} n={num_results}"},
            "tags": ["{query}", "static"],
            "flag": True,
        },
        query="ion traps",
        num_results=7,
    )

    assert substituted == {
        "tool_name": "search_documents",
        "arguments": {"query": "ion traps", "limit": 7, "note": "q=ion traps n=7"},
        "tags": ["ion traps", "static"],
        "flag": True,
    }


def test_parse_search_channels_rejects_non_dict_arguments() -> None:
    module = _load_tools_module()

    channels = module._parse_search_channels(
        [
            {
                "name": "wiki",
                "tool": "mcp_wiki",
                "function": "wiki_call_tool",
                "arguments": {"tool_name": "search_documents", "arguments": {"query": "{query}"}},
            },
            {"name": "bad", "tool": "t", "function": "f", "arguments": "not a dict"},
        ],
    )

    # Typed validation: an entry with a non-dict arguments template is
    # invalid and dropped, not silently nulled.
    assert [channel.name for channel in channels] == ["wiki"]
    assert channels[0].arguments == {"tool_name": "search_documents", "arguments": {"query": "{query}"}}


@pytest.mark.asyncio
async def test_mcp_bridge_channel_uses_arguments_template_awaits_a_vendor_async_function_and_unwraps_content() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools(
        search_tool="grounded_search",
        search_function="search_public_web",
        search_channels=[
            {
                "name": "wiki",
                "tool": "mcp_wiki",
                "function": "wiki_call_tool",
                "description": "Internal wiki",
                "arguments": {
                    "tool_name": "search_documents",
                    "arguments": {"query": "{query}", "limit": "{num_results}"},
                },
            },
        ],
    )
    bridge_calls: list[dict[str, Any]] = []

    async def wiki_call_tool(*, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        bridge_calls.append({"tool_name": tool_name, "arguments": arguments})
        return ToolResult(
            content=json.dumps(
                {"documents": [{"url": "https://wiki.example/doc/1", "title": "Doc", "context": "wiki fact"}]},
            ),
        )

    class GroundedSearch(Toolkit):
        @tool
        def search_public_web(self, _query: str) -> str:
            return json.dumps({"sources": []})

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "grounded_search":
            return GroundedSearch()
        if name == "mcp_wiki":
            return AgnoToolkit(name="mcp_wiki", tools=[wiki_call_tool])
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    async def fake_loop(**kwargs: Any) -> Any:
        hits = await kwargs["search_fn"](_load_loop_module().SearchQuery(query="frobnicator runbook", kind="wiki"), 5)
        assert hits[0].url == "https://wiki.example/doc/1"
        assert hits[0].snippet == "wiki fact"
        return _result(module)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert bridge_calls == [
        {"tool_name": "search_documents", "arguments": {"query": "frobnicator runbook", "limit": 5}},
    ]


def test_substitute_placeholders_does_not_reexpand_placeholder_text_in_query() -> None:
    module = _load_tools_module()

    substituted = module._substitute_placeholders(
        {"note": "q={query} limit={num_results}"},
        query="find {num_results} docs",
        num_results=5,
    )

    assert substituted == {"note": "q=find {num_results} docs limit=5"}


def test_result_text_accepts_str_and_tool_results_and_rejects_everything_else() -> None:
    module = _load_tools_module()

    assert module._result_text("already text") == "already text"
    assert module._result_text(ToolResult(content='{"results": []}')) == '{"results": []}'
    assert module._result_text(module.ToolValue(content='{"results": []}')) == '{"results": []}'
    with pytest.raises(TypeError, match="unsupported result type"):
        module._result_text(42)


def test_parse_search_channels_accepts_json_object_strings() -> None:
    module = _load_tools_module()

    channels = module._parse_search_channels(
        [
            '{"name": "wiki", "tool": "mcp_wiki", "function": "wiki_call_tool", "description": "Internal wiki", '
            '"arguments": {"tool_name": "list_documents", "arguments": {"query": "{query}", "limit": "{num_results}"}}}',
            '{"name": "chat", "tool": "chat_tool", "function": "search_messages"}',
        ],
    )

    assert [channel.name for channel in channels] == ["wiki", "chat"]
    assert channels[0].tool == "mcp_wiki"
    assert channels[0].function == "wiki_call_tool"
    assert channels[0].description == "Internal wiki"
    assert channels[0].arguments == {
        "tool_name": "list_documents",
        "arguments": {"query": "{query}", "limit": "{num_results}"},
    }
    assert channels[1].tool == "chat_tool"
    assert channels[1].function == "search_messages"
    assert channels[1].arguments is None


def test_parse_search_channels_reparses_comma_joined_runtime_string() -> None:
    module = _load_tools_module()

    # MindRoom's per-agent override path joins string[] values with ", "
    # before they reach the constructor; commas inside quoted JSON values
    # must not break the re-parse.
    joined = ", ".join(
        [
            '{"name": "wiki", "tool": "mcp_wiki", "function": "wiki_call_tool", "description": "Runbooks, docs, and more"}',
            '{"name": "chat", "tool": "chat_tool", "function": "search_messages"}',
        ],
    )

    channels = module._parse_search_channels(joined)

    assert [channel.name for channel in channels] == ["wiki", "chat"]
    assert channels[0].description == "Runbooks, docs, and more"


def test_parse_search_channels_drops_malformed_json_strings_without_crashing() -> None:
    module = _load_tools_module()

    channels = module._parse_search_channels(['{"name": "broken"', "chat=chat_tool.search|Chat"])

    assert [channel.name for channel in channels] == ["chat"]


def test_parse_search_channels_splits_comma_joined_compact_strings() -> None:
    module = _load_tools_module()

    joined = "wiki=wiki_tool.search_documents|Runbooks, docs, and more, chat=chat_tool.search_messages|Team chat"

    channels = module._parse_search_channels(joined)

    assert [channel.name for channel in channels] == ["wiki", "chat"]
    assert channels[0].description == "Runbooks, docs, and more"
    assert channels[1].description == "Team chat"


def test_parse_search_channels_drops_standalone_malformed_json_string() -> None:
    module = _load_tools_module()

    assert module._parse_search_channels('{"name": "broken"') == []


@pytest.mark.asyncio
async def test_ground_fn_is_wired_into_heavy_mode_loop_kwargs() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_heavy_research_loop", AsyncMock(return_value=_result(module))) as heavy_mock,
    ):
        result = json.loads(await tools.deep_research("What?", parallel_researchers=3))

    assert result["status"] == "ok"
    assert callable(heavy_mock.call_args.kwargs["ground_fn"])


@pytest.mark.asyncio
async def test_grounding_false_disables_the_gate() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))) as loop_mock,
    ):
        result = json.loads(await tools.deep_research("What?", grounding=False))

    assert result["status"] == "ok"
    assert loop_mock.call_args.kwargs["ground_fn"] is None


@pytest.mark.asyncio
async def test_ground_model_routes_to_separate_model_instance() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_model_provider", side_effect=[object(), object()]) as build_provider,
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        result = json.loads(await tools.deep_research("What?", ground_model="override"))

    assert result["status"] == "ok"
    assert [call.args[2] for call in build_provider.call_args_list] == ["active", "override"]


def test_round_progress_shows_open_question_count() -> None:
    module = _load_tools_module()
    event = {"round": 2, "max_rounds": 9, "thought": "t", "confidence": 0.3, "open_questions": 3}
    assert module._format_round_progress(event, verbose=False).startswith("round 2/9 · 0.30 · 3 open · t")
    event_clean = {"round": 2, "max_rounds": 9, "thought": "t", "confidence": 0.3, "open_questions": 0}
    assert module._format_round_progress(event_clean, verbose=False).startswith("round 2/9 · 0.30 · t")


@pytest.mark.asyncio
async def test_channel_toolkits_resolve_with_callers_worker_target_and_config() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools(
        search_tool="grounded_search",
        search_function="search_public_web",
        search_channels=[{"name": "wiki", "tool": "mcp_wiki", "function": "wiki_call_tool"}],
    )
    context = _tool_context(sender=AsyncMock())
    sentinel_target = SimpleNamespace(worker_scope="user_agent", worker_key="user_agent:bas:code")
    context.config.get_worker_grantable_credentials = lambda: frozenset({"google_vertex_adc"})
    calls: list[tuple[str, Any, Any, Any]] = []

    class GroundedSearch(Toolkit):
        @tool
        def search_public_web(self, _query: str) -> str:
            return json.dumps({"sources": []})

    class McpWiki(Toolkit):
        @tool
        async def wiki_call_tool(self, *, tool_name: str, arguments: dict[str, Any]) -> str:
            del tool_name, arguments
            return json.dumps({"documents": []})

    def tool_by_name(name: str, *_args: Any, **kwargs: Any) -> Any:
        calls.append(
            (name, kwargs.get("worker_target"), kwargs.get("allowed_shared_services"), kwargs.get("runtime_config")),
        )
        if name == "grounded_search":
            return GroundedSearch()
        if name == "mcp_wiki":
            return McpWiki()
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    with (
        tool_runtime_context(context),
        patch.object(type(context), "resolve_worker_target", lambda _self: sentinel_target),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    # The MCP channel is built with the caller's worker target, grantable
    # shared services, and config, which place it and reach its worker; the
    # main search backend and website reader keep their previous unscoped build.
    assert ("mcp_wiki", sentinel_target, frozenset({"google_vertex_adc"}), context.config) in calls
    assert ("grounded_search", None, None, None) in calls
    assert ("website", None, None, None) in calls


@pytest.mark.asyncio
async def test_worker_target_resolution_failure_is_loud() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools(
        search_tool="grounded_search",
        search_function="search_public_web",
        search_channels=[{"name": "wiki", "tool": "wiki_tool", "function": "search_documents"}],
    )
    context = _tool_context(sender=AsyncMock())

    def explode(_self: Any) -> Any:
        msg = "resolve_worker_target requires an agent dispatch"
        raise ValueError(msg)

    class GroundedSearch(Toolkit):
        @tool
        def search_public_web(self, _query: str) -> str:
            return json.dumps({"sources": []})

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "grounded_search":
            return GroundedSearch()
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    with (
        tool_runtime_context(context),
        patch.object(type(context), "resolve_worker_target", explode),
        patch.object(module, "build_model_provider", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
        pytest.raises(ValueError, match="requires an agent dispatch"),
    ):
        await tools.deep_research("What?")


@pytest.mark.asyncio
async def test_ask_model_sends_one_prompt_with_its_schema_through_the_owned_runtime() -> None:
    module = _load_tools_module()
    provider = object()
    step = module.ResearchStep(
        thought="finish",
        updated_report="partial",
        open_questions=[],
        confidence=0.2,
        next_action="finish",
    )
    complete = AsyncMock(return_value=SimpleNamespace(content=step))
    with patch.object(module, "complete_prompt", complete):
        result = await module._ask_model(provider, "Plan.", schema=module.ResearchStep)
    assert result is step
    complete.assert_awaited_once_with(provider, "Plan.", schema=module.ResearchStep)
