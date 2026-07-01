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

from mindroom.config.main import Config, load_config
from mindroom.constants import resolve_primary_runtime_paths
from mindroom.tool_system.metadata import get_tool_by_name as resolve_tool_by_name
from mindroom.tool_system.plugins import load_plugins
from mindroom.tool_system.registry_state import capture_tool_registry_snapshot, restore_tool_registry_snapshot
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
    spec.loader.exec_module(module)
    return module


def _load_loop_module() -> ModuleType:
    return importlib.import_module(f"{PACKAGE_NAME}.loop")


def _tool_context(
    *,
    sender: AsyncMock | None = None,
    active_model_name: str | None = "active",
    resolve_runtime_model: Mock | None = None,
) -> ToolRuntimeContext:
    resolver = resolve_runtime_model or Mock(return_value=SimpleNamespace(model_name="resolved"))
    config = SimpleNamespace(
        models={"active": object(), "override": object(), "resolved": object()},
        resolve_runtime_model=resolver,
        debug=SimpleNamespace(log_llm_requests=False),
    )
    return ToolRuntimeContext(
        agent_name="code",
        room_id="!room:localhost",
        thread_id="$thread-root",
        resolved_thread_id="$thread-root",
        requester_id="@user:localhost",
        client=AsyncMock(),
        config=config,
        runtime_paths=SimpleNamespace(),
        event_cache=AsyncMock(),
        conversation_cache=AsyncMock(),
        active_model_name=active_model_name,
        session_id="session-1",
        hook_message_sender=sender,
        correlation_id="corr-1",
    )


class _FakeSerper:
    api_key = "key"

    def search_web(self, _query: str, num_results: int | None = None) -> str:
        return json.dumps({"organic": []})

    def search_news(self, _query: str, num_results: int | None = None) -> str:
        return json.dumps({"news": []})

    def search_scholar(self, _query: str, num_results: int | None = None) -> str:
        return json.dumps({"organic": []})

    def scrape_webpage(self, _url: str) -> str:
        return json.dumps({"text": "fallback text"})


class _FakeWebsite:
    def read_url(self, _url: str) -> str:
        return json.dumps([{"content": "website text long enough" * 40, "meta_data": {"title": "Title"}}])


def _tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
    if name == "serper":
        return _FakeSerper()
    if name == "website":
        return _FakeWebsite()
    raise AssertionError(name)


def _entrypoint(entrypoint: Any) -> SimpleNamespace:
    return SimpleNamespace(entrypoint=entrypoint)


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
    )


async def _await(value: Any) -> Any:
    return value


def test_plugin_discovery_registers_tool_and_builtin_dependencies_resolve(tmp_path: Path) -> None:
    snapshot = capture_tool_registry_snapshot()
    runtime_paths = resolve_primary_runtime_paths(config_path=tmp_path / "config.yaml")
    config = Config(plugins=[{"path": str(Path(__file__).resolve().parents[1])}])

    try:
        with patch("mindroom.tool_system.metadata.ensure_tool_deps", return_value=False) as ensure_deps:
            plugins = load_plugins(config, runtime_paths, set_skill_roots=False, skip_broken_plugins=False)
            deep_research = resolve_tool_by_name(
                "deep_research",
                runtime_paths,
                worker_target=None,
            )
            resolve_tool_by_name("serper", runtime_paths, worker_target=None)
            resolve_tool_by_name("website", runtime_paths, worker_target=None)
    finally:
        restore_tool_registry_snapshot(snapshot)

    assert [plugin.name for plugin in plugins] == ["deep-research"]
    assert type(deep_research).__name__ == "DeepResearchTools"
    checked_dependency_sets = {tuple(call.args[0]) for call in ensure_deps.call_args_list}
    assert ("requests",) in checked_dependency_sets
    assert ("httpx", "beautifulsoup4") in checked_dependency_sets


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
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
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
    ):
        assert key in result


@pytest.mark.asyncio
async def test_model_override_precedence_and_execution_identity_are_wired() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    context = _tool_context(sender=AsyncMock())
    execution_identity = object()

    with (
        tool_runtime_context(context),
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=execution_identity),
        patch.object(module, "get_model_instance", return_value=object()) as get_model,
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        result = json.loads(await tools.deep_research("What?", model="override"))

    assert result["status"] == "ok"
    get_model.assert_called_once_with(
        context.config,
        context.runtime_paths,
        "override",
        execution_identity=execution_identity,
    )
    context.config.resolve_runtime_model.assert_not_called()


@pytest.mark.asyncio
async def test_active_model_precedes_runtime_resolution() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    context = _tool_context(sender=AsyncMock(), active_model_name="active")

    with (
        tool_runtime_context(context),
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()) as get_model,
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert get_model.call_args.args[2] == "active"
    context.config.resolve_runtime_model.assert_not_called()


@pytest.mark.asyncio
async def test_runtime_model_resolution_fallback_when_no_active_model() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    resolver = Mock(return_value=SimpleNamespace(model_name="resolved"))
    context = _tool_context(sender=AsyncMock(), active_model_name=None, resolve_runtime_model=resolver)

    with (
        tool_runtime_context(context),
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()) as get_model,
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert get_model.call_args.args[2] == "resolved"
    resolver.assert_called_once_with(
        entity_name="code",
        room_id="!room:localhost",
        thread_id="$thread-root",
        runtime_paths=context.runtime_paths,
    )


@pytest.mark.asyncio
async def test_unknown_model_override_returns_error_envelope() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "get_model_instance", return_value=object()) as get_model,
        patch.object(module, "run_research_loop", AsyncMock()) as loop_mock,
    ):
        result = json.loads(await tools.deep_research("What?", model="bogus"))

    assert result["status"] == "error"
    assert "Unknown model override: bogus" in result["message"]
    get_model.assert_not_called()
    loop_mock.assert_not_called()


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
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
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
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
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
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
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
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
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
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
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
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=_tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        await tools.deep_research("What?", max_rounds=999, wall_clock_seconds=99999)

    assert captured["max_rounds"] == 40
    assert captured["wall_clock_seconds"] == 900


@pytest.mark.asyncio
async def test_tool_resolution_keeps_sandbox_proxy_enabled() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()
    calls: list[dict[str, Any]] = []

    def tool_by_name(name: str, *_args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return _tool_by_name(name)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock(return_value=_result(module))),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "ok"
    assert calls
    assert all(call.get("disable_sandbox_proxy") is not True for call in calls)


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
        patch.object(module, "get_model_instance", return_value=object()) as get_model,
        patch.object(module, "run_research_loop", AsyncMock()) as loop_mock,
    ):
        result = json.loads(await tools.deep_research("What?", **kwargs))

    assert result["status"] == "error"
    get_model.assert_not_called()
    loop_mock.assert_not_called()


@pytest.mark.asyncio
async def test_tool_function_entrypoints_are_used_for_search_and_read() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    class EntrypointSerper(_FakeSerper):
        def __init__(self) -> None:
            self.functions = {"search_web": _entrypoint(self.entrypoint_search_web)}

        def entrypoint_search_web(self, query: str, num_results: int | None = None) -> str:
            return json.dumps({"organic": [{"link": "https://example.com", "title": query, "snippet": str(num_results)}]})

        def search_web(self, _query: str, num_results: int | None = None) -> str:
            raise AssertionError("direct search method bypassed toolkit function entrypoint")

    class EntrypointWebsite(_FakeWebsite):
        def __init__(self) -> None:
            self.functions = {"read_url": _entrypoint(self.entrypoint_read_url)}

        def entrypoint_read_url(self, url: str) -> str:
            return json.dumps([{"content": f"entrypoint text for {url}", "meta_data": {"title": "Entrypoint"}}])

        def read_url(self, _url: str) -> str:
            raise AssertionError("direct read method bypassed toolkit function entrypoint")

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
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
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

    class FakeAgent:
        def __init__(self, *_args: Any, output_schema: Any = None, **_kwargs: Any) -> None:
            self.output_schema = output_schema

        async def arun(self, _prompt: str, session_id: str | None = None) -> Any:
            if self.output_schema is module.ResearchStep:
                return SimpleNamespace(
                    content=module.ResearchStep(
                        thought="finish",
                        updated_report="partial",
                        open_questions=[],
                        confidence=0.2,
                        next_action="finish",
                    ),
                )
            return SimpleNamespace(content="final report")

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "serper":
            return ErrorSerper()
        if name == "website":
            return _FakeWebsite()
        raise AssertionError(name)

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "Agent", FakeAgent),
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
    ):
        result = json.loads(await tools.deep_research("What?", verbosity="silent"))

    assert result["status"] == "ok"
    assert result["warnings"] == ["seed search failed: quota exceeded"]


@pytest.mark.asyncio
async def test_website_error_payload_returns_read_warning() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    class ErrorWebsite:
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
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
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

    class HardenedWebsite:
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
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", side_effect=fake_loop),
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "error"
    assert reason in result["message"]
    scrape_webpage.assert_not_called()


@pytest.mark.asyncio
async def test_serper_unconfigured_returns_clear_error() -> None:
    module = _load_tools_module()
    tools = module.DeepResearchTools()

    class NoKeySerper(_FakeSerper):
        api_key = None

    def tool_by_name(name: str, *_args: Any, **_kwargs: Any) -> Any:
        return NoKeySerper() if name == "serper" else _FakeWebsite()

    with (
        tool_runtime_context(_tool_context(sender=AsyncMock())),
        patch.object(module, "build_execution_identity_from_runtime_context", return_value=object()),
        patch.object(module, "get_model_instance", return_value=object()),
        patch.object(module, "get_tool_by_name", side_effect=tool_by_name),
        patch.object(module, "run_research_loop", AsyncMock()) as loop_mock,
    ):
        result = json.loads(await tools.deep_research("What?"))

    assert result["status"] == "error"
    assert "Serper is not configured" in result["message"]
    loop_mock.assert_not_called()


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
        room_id="!deep-research-integration:localhost",
        thread_id="$deep-research-integration",
        resolved_thread_id="$deep-research-integration",
        requester_id="@deep-research-integration:localhost",
        client=AsyncMock(),
        config=config,
        runtime_paths=runtime_paths,
        event_cache=AsyncMock(),
        conversation_cache=AsyncMock(),
        active_model_name=model_name,
        session_id="deep-research-integration",
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
