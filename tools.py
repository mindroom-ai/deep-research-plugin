# ruff: noqa: INP001
"""Agent-facing tools for the MindRoom deep-research plugin."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agno.agent import Agent
from agno.tools import Toolkit
from agno.tools.function import ToolResult
from agno.tools.serper import SerperTools
from pydantic import BaseModel, ValidationError, model_validator

from mindroom.logging_config import get_logger
from mindroom.model_loading import get_model_instance
from mindroom.tool_system.metadata import (
    ConfigField,
    SetupType,
    ToolCategory,
    ToolStatus,
    get_tool_by_name,
    register_tool_with_metadata,
)
from mindroom.tool_system.runtime_context import (
    build_execution_identity_from_runtime_context,
    get_tool_runtime_context,
    resolve_tool_runtime_hook_bindings,
)

if TYPE_CHECKING:
    from mindroom.tool_system.runtime_context import ToolRuntimeContext
    from mindroom.tool_system.worker_routing import ResolvedWorkerTarget

from .prompts import BASE_CHANNEL_DESCRIPTIONS
from .loop import (
    MAX_ROUNDS_CAP,
    MAX_QUERIES_PER_ROUND,
    MAX_QUERIES_PER_ROUND_CAP,
    MAX_READS_PER_ROUND,
    MAX_READS_PER_ROUND_CAP,
    PARALLEL_RESEARCHERS_CAP,
    REPORT_TOKEN_CAP,
    RESULTS_PER_QUERY,
    RESULTS_PER_QUERY_CAP,
    WALL_CLOCK_SECONDS_CAP,
    Extraction,
    GroundingCheck,
    Page,
    ResearchStep,
    SearchHit,
    SearchQuery,
    clamp_int,
    run_heavy_research_loop,
    run_research_loop,
)

LOGGER = get_logger(__name__)
TOOL_NAME = "deep_research"
DEFAULT_SEARCH_TOOL = "serper"
SERPER_SEARCH_FUNCTIONS = {"web": "search_web", "news": "search_news", "scholar": "search_scholar"}
HIT_SNIPPET_CHAR_LIMIT = 500
_PLACEHOLDER_RE = re.compile(r"\{query\}|\{num_results\}")


def _coerce_channel_entries(raw: object) -> list[object]:
    """Expand raw channel config into a list of entry objects.

    MindRoom validates per-agent string[] overrides as lists of strings and
    hands them to the constructor comma-joined into ONE string, so channels
    with structured fields (arguments templates) must be authored as JSON
    object strings. Wrapping the joined form in brackets re-parses it as a
    JSON array regardless of commas inside quoted values; individual JSON
    object strings inside lists are decoded the same way.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("{"):
            try:
                # A successful parse of "[...]" is always a list.
                return json.loads(f"[{text}]")
            except json.JSONDecodeError:
                pass
        # Comma-joined compact entries: split only where the next segment
        # looks like a compact channel ("name="), so commas inside
        # descriptions do not split.
        return re.split(r", (?=[A-Za-z0-9_-]+=)", text)
    if not isinstance(raw, list):
        return []
    entries: list[object] = []
    for entry in raw:
        if isinstance(entry, str) and entry.lstrip().startswith("{"):
            try:
                entries.append(json.loads(entry))
                continue
            except json.JSONDecodeError:
                pass
        entries.append(entry)
    return entries


class _ChannelConfig(BaseModel):
    """One authored search channel: a named tool+function evidence backend."""

    name: str
    tool: str
    function: str
    description: str = ""
    arguments: dict[str, object] | None = None

    @model_validator(mode="after")
    def _normalize(self) -> _ChannelConfig:
        object.__setattr__(self, "name", self.name.strip().lower())
        object.__setattr__(self, "tool", self.tool.strip())
        object.__setattr__(self, "function", self.function.strip())
        description = self.description.strip() or f"search via the {self.tool} tool"
        object.__setattr__(self, "description", description)
        if not self.name or not self.tool or not self.function:
            msg = "channel entries need name, tool, and function"
            raise ValueError(msg)
        return self


def _coerce_channel_entries(raw: object) -> list[object]:
    """Expand raw channel config into a list of entry objects.

    MindRoom validates per-agent string[] overrides as lists of strings and
    hands them to the constructor comma-joined into ONE string, so channels
    with structured fields (arguments templates) must be authored as JSON
    object strings. Wrapping the joined form in brackets re-parses it as a
    JSON array regardless of commas inside quoted values; individual JSON
    object strings inside lists are decoded the same way.
    """
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        if text.startswith("{"):
            try:
                # A successful parse of "[...]" is always a list.
                return json.loads(f"[{text}]")
            except json.JSONDecodeError:
                pass
        # Comma-joined compact entries: split only where the next segment
        # looks like a compact channel ("name="), so commas inside
        # descriptions do not split.
        return re.split(r", (?=[A-Za-z0-9_-]+=)", text)
    if not isinstance(raw, list):
        return []
    entries: list[object] = []
    for entry in raw:
        if isinstance(entry, str) and entry.lstrip().startswith("{"):
            try:
                entries.append(json.loads(entry))
                continue
            except json.JSONDecodeError:
                pass
        entries.append(entry)
    return entries


def _parse_search_channels(raw: object) -> list[_ChannelConfig]:
    """Validate authored search_channels config into typed channel configs.

    Accepts structured entries ({name, tool, function, description}, plus an
    optional arguments template with {query}/{num_results} placeholders for
    functions that take structured kwargs, e.g. MCP tools), JSON object
    strings of the same shape (required for MindRoom per-agent overrides,
    whose string[] fields must be lists of strings), and the compact string
    form "name=tool.function|description" (no arguments template). Invalid
    entries and names that shadow the built-in web/news/scholar channels are
    dropped with a warning.
    """
    channels: list[_ChannelConfig] = []
    seen_names: set[str] = set()
    for entry in _coerce_channel_entries(raw):
        if isinstance(entry, str):
            head, _, description = entry.partition("|")
            name, _, target = head.partition("=")
            tool, _, function = target.strip().partition(".")
            candidate: object = {
                "name": name,
                "tool": tool,
                "function": function,
                "description": description.strip(),
            }
        else:
            candidate = entry
        try:
            channel = _ChannelConfig.model_validate(candidate)
        except ValidationError:
            LOGGER.warning("deep_research_search_channel_invalid", entry=repr(entry))
            continue
        if channel.name in BASE_CHANNEL_DESCRIPTIONS or channel.name in seen_names:
            LOGGER.warning("deep_research_search_channel_name_conflict", channel=channel.name)
            continue
        seen_names.add(channel.name)
        channels.append(channel)
    return channels


@dataclass(frozen=True)
class _ResolvedChannel:
    """One search channel bound to its resolved toolkit for a single run."""

    toolkit: Toolkit
    function: str
    description: str
    arguments: dict[str, object] | None


DEFAULT_MAX_ROUNDS = MAX_ROUNDS_CAPDEFAULT_MAX_ROUNDS = MAX_ROUNDS_CAP
DEFAULT_WALL_CLOCK_SECONDS = WALL_CLOCK_SECONDS_CAP
DEFAULT_MAX_QUERIES_PER_ROUND = MAX_QUERIES_PER_ROUND
DEFAULT_RESULTS_PER_QUERY = RESULTS_PER_QUERY
DEFAULT_MAX_READS_PER_ROUND = MAX_READS_PER_ROUND
DEFAULT_PAGE_CHAR_LIMIT = 150_000
DEFAULT_REPORT_TOKEN_CAP = REPORT_TOKEN_CAP
MIN_PAGE_CHAR_LIMIT = 10_000
MAX_PAGE_CHAR_LIMIT = 600_000
MIN_REPORT_TOKEN_CAP = 2_000
MAX_REPORT_TOKEN_CAP = 64_000
PROGRESS_EMIT_TIMEOUT_SECONDS = 2.0


def _payload(status: str, tool: str = TOOL_NAME, **kwargs: object) -> str:
    payload: dict[str, object] = {"status": status, "tool": tool}
    payload.update(kwargs)
    return json.dumps(payload, sort_keys=True)


def _error(message: str, *, warnings: list[str] | None = None) -> str:
    return _payload("error", message=message, warnings=warnings or [])


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def _resolve_model_name(context: ToolRuntimeContext, model: str | None) -> str:
    if model:
        if model not in context.config.models:
            available = ", ".join(sorted(context.config.models))
            msg = f"Unknown model override: {model}. Available models: {available}"
            raise ValueError(msg)
        return model
    if context.active_model_name:
        return context.active_model_name
    resolved = context.config.resolve_runtime_model(
        entity_name=context.agent_name,
        room_id=context.room_id,
        thread_id=context.resolved_thread_id,
        runtime_paths=context.runtime_paths,
    )
    return resolved.model_name


def _session_id(context: ToolRuntimeContext, role: str, count: int) -> str:
    base = context.session_id or context.correlation_id or context.resolved_thread_id or context.room_id
    return f"deep-research:{base}:{role}:{count}"


def _extract_text_from_website_payload(payload: str) -> tuple[str, str]:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return "", payload
    if isinstance(parsed, dict):
        error = parsed.get("error")
        status = str(parsed.get("status") or "").lower()
        message = parsed.get("message")
        if error or status == "error":
            raise RuntimeError(str(error or message or "website read failed"))
    docs = parsed if isinstance(parsed, list) else [parsed]
    title = ""
    chunks: list[str] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        meta = doc.get("meta_data") or doc.get("metadata") or {}
        if isinstance(meta, dict) and not title:
            title = str(meta.get("title") or meta.get("name") or "")
        if not title:
            title = str(doc.get("title") or doc.get("name") or "")
        for key in ("content", "text", "page_content", "description"):
            value = doc.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
                break
    return title, "\n\n".join(chunks)


_URL_KEYS = ("link", "url", "uri", "permalink")
_TITLE_KEYS = ("title", "name")
_SNIPPET_KEYS = ("snippet", "description", "summary", "context", "text", "domain")


class _SearchResultRow(BaseModel):
    """One search row normalized from any supported backend shape.

    Backends use different key names; each field takes the first key in its
    priority tuple that carries a non-empty string, skipping non-string
    values (some APIs put metadata objects under keys like "link").
    """

    url: str
    title: str = ""
    snippet: str = ""

    @model_validator(mode="before")
    @classmethod
    def _from_backend_row(cls, data: object) -> dict[str, str]:
        if not isinstance(data, dict):
            msg = "search row must be a JSON object"
            raise ValueError(msg)

        def first_string(keys: tuple[str, ...]) -> str:
            for key in keys:
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return ""

        url = first_string(_URL_KEYS)
        if not url:
            msg = "search row carries no URL"
            raise ValueError(msg)
        return {
            "url": url,
            "title": first_string(_TITLE_KEYS),
            "snippet": first_string(_SNIPPET_KEYS)[:HIT_SNIPPET_CHAR_LIMIT],
        }


def _parse_search_results(raw: str) -> list[SearchHit]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    rows: list[object] = []
    if isinstance(parsed, list):
        rows = list(parsed)
    elif isinstance(parsed, dict):
        error = parsed.get("error")
        if error or str(parsed.get("status") or "").lower() == "error":
            detail = error or parsed.get("message") or parsed.get("description") or parsed.get("code")
            raise RuntimeError(str(detail or "search failed"))
        for key in ("organic", "news", "articles", "scholar", "results", "sources", "items", "documents"):
            value = parsed.get(key)
            if isinstance(value, list):
                rows.extend(value)
    else:
        return []
    hits: list[SearchHit] = []
    for row in rows:
        try:
            normalized = _SearchResultRow.model_validate(row)
        except ValidationError:
            continue
        hits.append(
            SearchHit(url=normalized.url, title=normalized.title or normalized.url, snippet=normalized.snippet),
        )
    return hits


def _tool_function_entrypoint(toolkit: Toolkit, function_name: str) -> Callable[..., object]:
    # MCP-backed toolkits register their functions in async_functions.
    function = toolkit.functions.get(function_name) or toolkit.async_functions.get(function_name)
    if function is None or function.entrypoint is None:
        msg = f"Tool function is unavailable: {function_name}"
        raise RuntimeError(msg)
    return function.entrypoint


async def _call_tool_function(toolkit: Toolkit, function_name: str, *args: object, **kwargs: object) -> str:
    raw = await asyncio.to_thread(_tool_function_entrypoint(toolkit, function_name), *args, **kwargs)
    return _result_text(raw)


def _accepts_keyword(function: Callable[..., object], name: str) -> bool:
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return True
    if name in parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def _substitute_placeholders(value: object, *, query: str, num_results: int) -> object:
    """Fill {query}/{num_results} placeholders in an arguments template.

    A string that is exactly one placeholder keeps the substituted value's
    type (so "{num_results}" becomes an int); placeholders embedded in longer
    strings are replaced textually.
    """
    if isinstance(value, str):
        if value == "{query}":
            return query
        if value == "{num_results}":
            return num_results
        # Single-pass substitution: placeholder-like text inside the query
        # itself must not be expanded again.
        return _PLACEHOLDER_RE.sub(
            lambda match: query if match.group() == "{query}" else str(num_results),
            value,
        )
    if isinstance(value, dict):
        return {key: _substitute_placeholders(item, query=query, num_results=num_results) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_placeholders(item, query=query, num_results=num_results) for item in value]
    return value


async def _call_search_function(
    toolkit: object,
    function_name: str,
    query: str,
    *,
    num_results: int,
    arguments: dict[str, object] | None = None,
) -> str:
    entrypoint = _tool_function_entrypoint(toolkit, function_name)
    if arguments is not None:
        # Keyword-shaped call for functions that take structured kwargs
        # (e.g. MCP tools); the template carries the query placement.
        args: tuple[object, ...] = ()
        substituted = _substitute_placeholders(arguments, query=query, num_results=num_results)
        kwargs = substituted if isinstance(substituted, dict) else {}
    else:
        args = (query,)
        kwargs = {"num_results": num_results} if _accepts_keyword(entrypoint, "num_results") else {}
    if inspect.iscoroutinefunction(entrypoint):
        raw = await entrypoint(*args, **kwargs)
    else:
        raw = await asyncio.to_thread(entrypoint, *args, **kwargs)
        if inspect.isawaitable(raw):
            raw = await raw
    return _result_text(raw)


def _result_text(raw: object) -> str:
    """Normalize a search call's return value to text.

    Search callables return plain text; MindRoom's MCP toolkits return an
    agno ToolResult whose content is already normalized to a string.
    Anything else is a contract violation and fails loudly.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, ToolResult):
        return raw.content
    msg = f"Search function returned an unsupported result type: {type(raw).__name__}"
    raise TypeError(msg)


def _authored_tool_overrides(context: ToolRuntimeContext, tool_name: str) -> dict[str, object]:
    """Return the calling agent's authored config overrides for one tool."""
    for entry in context.config.get_agent(context.agent_name).tools:
        if entry.name == tool_name:
            return dict(entry.overrides)
    return {}


async def _emit_message(context: ToolRuntimeContext, text: str, *, timeout_seconds: float = PROGRESS_EMIT_TIMEOUT_SECONDS) -> None:
    bindings = resolve_tool_runtime_hook_bindings(context)
    if bindings.message_sender is None:
        return
    if timeout_seconds <= 0:
        LOGGER.warning("deep_research_progress_emit_skipped_budget_exhausted")
        return
    try:
        async with asyncio.timeout(min(PROGRESS_EMIT_TIMEOUT_SECONDS, timeout_seconds)):
            await bindings.message_sender(
                context.room_id,
                text,
                context.resolved_thread_id or context.thread_id,
                "deep-research:progress",
                None,
                trigger_dispatch=False,
            )
    except TimeoutError:
        LOGGER.warning("deep_research_progress_emit_timeout")
    except Exception as exc:
        LOGGER.warning("deep_research_progress_emit_failed", error=str(exc))


def _format_round_progress(event: dict[str, object], *, verbose: bool) -> str:
    thought = str(event.get("thought") or "").replace("\n", " ")
    if len(thought) > 80:
        thought = thought[:77].rstrip() + "..."
    confidence = float(event.get("confidence") or 0)
    researcher = event.get("researcher")
    prefix = f"researcher {researcher} · " if researcher else ""
    open_questions = int(event.get("open_questions") or 0)
    open_note = f" · {open_questions} open" if open_questions else ""
    line = f"{prefix}round {event['round']}/{event['max_rounds']} · {confidence:.2f}{open_note} · {thought}"
    if not verbose:
        return line
    queries = event.get("search_queries")
    urls = event.get("read_urls")
    details: list[str] = []
    if queries:
        details.append(f"queries={queries}")
    if urls:
        details.append(f"urls={urls}")
    return line if not details else f"{line}\n" + "\n".join(details)


class DeepResearchTools(Toolkit):
    """Toolkit exposing bounded web research as one MindRoom tool."""

    def __init__(
        self,
        *,
        search_tool: str | None = None,
        search_function: str | None = None,
        search_channels: object = None,
    ) -> None:
        self.search_tool = (search_tool or "").strip() if isinstance(search_tool, str) else ""
        self.search_tool = self.search_tool or DEFAULT_SEARCH_TOOL
        self.search_function = (search_function or "").strip() if isinstance(search_function, str) else ""
        self.search_channels = _parse_search_channels(search_channels)
        super().__init__(
            name=TOOL_NAME,
            instructions=(
                "Use deep_research for bounded, cited research over the web. "
                "It returns a JSON envelope whose report field contains the final cited Markdown answer."
            ),
            tools=[self.deep_research],
        )

    async def deep_research(  # noqa: C901, PLR0911, PLR0915
        self,
        question: str,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        wall_clock_seconds: int = DEFAULT_WALL_CLOCK_SECONDS,
        model: str | None = None,
        verbosity: str = "progress",
        max_queries_per_round: int = DEFAULT_MAX_QUERIES_PER_ROUND,
        results_per_query: int = DEFAULT_RESULTS_PER_QUERY,
        max_reads_per_round: int = DEFAULT_MAX_READS_PER_ROUND,
        page_char_limit: int = DEFAULT_PAGE_CHAR_LIMIT,
        report_token_cap: int = DEFAULT_REPORT_TOKEN_CAP,
        parallel_researchers: int = 1,
        extract_model: str | None = None,
        grounding: bool = True,
        ground_model: str | None = None,
    ) -> str:
        """Run a bounded deep research loop for one question.

        Set parallel_researchers > 1 for heavy mode: N independent research
        loops explore the question from different angles concurrently and a
        final synthesis pass integrates their cited reports (roughly N times
        the token cost). Set extract_model to route the high-volume page
        extraction role to a cheaper model. Set grounding=False to skip the
        final grounding gate (saves one or two LLM calls on quick bounded
        runs), and ground_model to run the grounding check on a different
        configured model than the writer.
        """
        normalized_question = question.strip() if isinstance(question, str) else ""
        if not normalized_question:
            return _error("question must be a non-empty string")

        context = get_tool_runtime_context()
        if context is None:
            return _error("Deep research tool context is unavailable in this runtime path.")

        try:
            max_rounds = clamp_int(max_rounds, minimum=1, maximum=MAX_ROUNDS_CAP)
            wall_clock_seconds = clamp_int(wall_clock_seconds, minimum=60, maximum=WALL_CLOCK_SECONDS_CAP)
            max_queries_per_round = clamp_int(
                max_queries_per_round,
                minimum=1,
                maximum=MAX_QUERIES_PER_ROUND_CAP,
            )
            results_per_query = clamp_int(results_per_query, minimum=1, maximum=RESULTS_PER_QUERY_CAP)
            max_reads_per_round = clamp_int(max_reads_per_round, minimum=1, maximum=MAX_READS_PER_ROUND_CAP)
            page_char_limit = clamp_int(
                page_char_limit,
                minimum=MIN_PAGE_CHAR_LIMIT,
                maximum=MAX_PAGE_CHAR_LIMIT,
            )
            report_token_cap = clamp_int(
                report_token_cap,
                minimum=MIN_REPORT_TOKEN_CAP,
                maximum=MAX_REPORT_TOKEN_CAP,
            )
            parallel_researchers = clamp_int(
                parallel_researchers,
                minimum=1,
                maximum=PARALLEL_RESEARCHERS_CAP,
            )
            verbosity = (
                verbosity
                if isinstance(verbosity, str) and verbosity in {"silent", "progress", "verbose"}
                else "progress"
            )
            grounding = grounding if isinstance(grounding, bool) else True
            model_name = _resolve_model_name(context, model)
            extract_model_name = _resolve_model_name(context, extract_model) if extract_model else model_name
            ground_model_name = _resolve_model_name(context, ground_model) if ground_model else model_name
            execution_identity = build_execution_identity_from_runtime_context(context)
            live_model = get_model_instance(
                context.config,
                context.runtime_paths,
                model_name,
                execution_identity=execution_identity,
            )
            extract_live_model = (
                live_model
                if extract_model_name == model_name
                else get_model_instance(
                    context.config,
                    context.runtime_paths,
                    extract_model_name,
                    execution_identity=execution_identity,
                )
            )
            ground_live_model = (
                live_model
                if ground_model_name == model_name
                else get_model_instance(
                    context.config,
                    context.runtime_paths,
                    ground_model_name,
                    execution_identity=execution_identity,
                )
            )
        except Exception as exc:
            LOGGER.warning("deep_research_model_resolution_failed", error=str(exc))
            return _error(str(exc))

        search_tool_name = self.search_tool
        try:
            search_overrides = _authored_tool_overrides(context, search_tool_name)
            search_toolkit = get_tool_by_name(
                search_tool_name,
                context.runtime_paths,
                tool_config_overrides=search_overrides or None,
                worker_target=None,
            )
        except Exception as exc:
            LOGGER.warning("deep_research_search_tool_unavailable", search_tool=search_tool_name, error=str(exc))
            return _error(
                f"Search tool '{search_tool_name}' is not configured or unavailable. "
                "Configure it to use deep_research.",
            )
        if isinstance(search_toolkit, SerperTools) and not search_toolkit.api_key:
            return _error(
                f"Search tool '{search_tool_name}' is not configured. "
                "Configure its API key to use deep_research.",
            )

        # Extra search channels degrade gracefully: an unavailable channel is
        # dropped (and reported in warnings) instead of failing the run, and
        # the reasoner is only told about channels that actually resolved.
        channels: dict[str, _ResolvedChannel] = {}
        channel_warnings: list[str] = []
        channel_worker_target: ResolvedWorkerTarget | None = None
        channel_shared_services: frozenset[str] | None = None
        if self.search_channels:
            # Requires MindRoom >= v2026.7.38 (ToolRuntimeContext.resolve_worker_target);
            # older runtimes fail loudly here rather than degrading to an
            # unscoped session that can never be signed in.
            channel_worker_target = context.resolve_worker_target()
            if channel_worker_target.worker_scope is not None:
                channel_shared_services = context.config.get_worker_grantable_credentials()
        for channel_config in self.search_channels:
            try:
                channel_toolkit = get_tool_by_name(
                    channel_config.tool,
                    context.runtime_paths,
                    tool_config_overrides=_authored_tool_overrides(context, channel_config.tool) or None,
                    allowed_shared_services=channel_shared_services,
                    worker_target=channel_worker_target,
                )
                _tool_function_entrypoint(channel_toolkit, channel_config.function)
            except Exception as exc:
                LOGGER.warning(
                    "deep_research_search_channel_unavailable",
                    channel=channel_config.name,
                    tool=channel_config.tool,
                    error=str(exc),
                )
                channel_warnings.append(f"search channel '{channel_config.name}' unavailable: {exc}")
                continue
            channels[channel_config.name] = _ResolvedChannel(
                toolkit=channel_toolkit,
                function=channel_config.function,
                description=channel_config.description,
                arguments=channel_config.arguments,
            )

        if self.search_function:
            prompt_channels = [("web", BASE_CHANNEL_DESCRIPTIONS["web"])]
        else:
            prompt_channels = [(name, BASE_CHANNEL_DESCRIPTIONS[name]) for name in ("web", "news", "scholar")]
        prompt_channels.extend((name, channel.description) for name, channel in channels.items())

        try:
            website = get_tool_by_name(
                "website",
                context.runtime_paths,
                worker_target=None,
            )
        except Exception as exc:
            LOGGER.warning("deep_research_website_unavailable", error=str(exc))
            return _error(f"Website reader is unavailable: {exc}")

        counters = {"reason": 0, "extract": 0, "synthesize": 0, "ground": 0}
        wrapper_start = time.monotonic()

        def remaining_wall_clock_seconds() -> float:
            return max(0.0, wall_clock_seconds - (time.monotonic() - wrapper_start))

        async def reason_fn(prompt: str) -> object:
            counters["reason"] += 1
            agent = Agent(
                model=live_model,
                output_schema=ResearchStep,
                telemetry=False,
                markdown=False,
            )
            response = await agent.arun(prompt, session_id=_session_id(context, "reason", counters["reason"]))
            return response.content

        async def extract_fn(prompt: str) -> object:
            counters["extract"] += 1
            agent = Agent(
                model=extract_live_model,
                output_schema=Extraction,
                telemetry=False,
                markdown=False,
            )
            response = await agent.arun(prompt, session_id=_session_id(context, "extract", counters["extract"]))
            return response.content

        async def synthesize_fn(prompt: str) -> str:
            counters["synthesize"] += 1
            agent = Agent(model=live_model, telemetry=False, markdown=False)
            response = await agent.arun(prompt, session_id=_session_id(context, "synthesize", counters["synthesize"]))
            content = response.content
            return content if isinstance(content, str) else str(content)

        async def ground_fn(prompt: str) -> object:
            counters["ground"] += 1
            agent = Agent(
                model=ground_live_model,
                output_schema=GroundingCheck,
                telemetry=False,
                markdown=False,
            )
            response = await agent.arun(prompt, session_id=_session_id(context, "ground", counters["ground"]))
            return response.content

        async def search_fn(query: SearchQuery, limit: int) -> list[SearchHit]:
            channel = channels.get(query.kind)
            if channel is not None:
                raw = await _call_search_function(
                    channel.toolkit,
                    channel.function,
                    query.query,
                    num_results=limit,
                    arguments=channel.arguments,
                )
            else:
                search_method = (
                    self.search_function
                    or SERPER_SEARCH_FUNCTIONS.get(query.kind)
                    or SERPER_SEARCH_FUNCTIONS["web"]
                )
                raw = await _call_search_function(search_toolkit, search_method, query.query, num_results=limit)
            return _parse_search_results(raw)

        async def read_fn(url: str) -> Page:
            title, text = _extract_text_from_website_payload(await _call_tool_function(website, "read_url", url))
            return Page(url=url, title=title or url, text=_truncate(text, page_char_limit))

        async def emit_fn(event: dict[str, object]) -> None:
            if verbosity == "silent":
                return
            await _emit_message(
                context,
                _format_round_progress(event, verbose=verbosity == "verbose"),
                timeout_seconds=remaining_wall_clock_seconds(),
            )

        try:
            if verbosity != "silent":
                researchers_note = f" · {parallel_researchers} researchers" if parallel_researchers > 1 else ""
                await _emit_message(
                    context,
                    f"deep_research started · {max_rounds} rounds · {model_name}{researchers_note}",
                    timeout_seconds=remaining_wall_clock_seconds(),
                )
            loop_kwargs: dict[str, object] = {
                "question": normalized_question,
                "max_rounds": max_rounds,
                "wall_clock_seconds": wall_clock_seconds,
                "reason_fn": reason_fn,
                "extract_fn": extract_fn,
                "search_fn": search_fn,
                "read_fn": read_fn,
                "synthesize_fn": synthesize_fn,
                "emit_fn": emit_fn,
                "budget_start": wrapper_start,
                "max_queries_per_round": max_queries_per_round,
                "results_per_query": results_per_query,
                "max_reads_per_round": max_reads_per_round,
                "report_char_cap": report_token_cap * 4,
                "search_channels": prompt_channels,
                "ground_fn": ground_fn if grounding else None,
            }
            if parallel_researchers > 1:
                result = await run_heavy_research_loop(researchers=parallel_researchers, **loop_kwargs)
            else:
                result = await run_research_loop(**loop_kwargs)
            if verbosity != "silent":
                await _emit_message(
                    context,
                    f"deep_research done · {result.stopped_reason} · {result.rounds_used} rounds",
                    timeout_seconds=remaining_wall_clock_seconds(),
                )
            result.elapsed_seconds = time.monotonic() - wrapper_start
            result.warnings = channel_warnings + result.warnings
        except Exception as exc:
            LOGGER.warning("deep_research_loop_failed", error=str(exc))
            return _error(f"deep_research failed: {exc}", warnings=channel_warnings)

        return _payload(
            "ok",
            question=result.question,
            report=result.report,
            sources=result.sources,
            sources_considered=result.sources_considered,
            sources_used=result.sources_used,
            confidence=result.confidence,
            rounds_used=result.rounds_used,
            stopped_reason=result.stopped_reason,
            elapsed_seconds=result.elapsed_seconds,
            warnings=result.warnings,
            stats=result.stats,
        )


_CONFIG_FIELDS = [
    ConfigField(
        name="search_tool",
        label="Search Tool",
        type="text",
        required=False,
        default=DEFAULT_SEARCH_TOOL,
        description="Registered MindRoom tool used for web search. Defaults to the built-in Serper tool.",
    ),
    ConfigField(
        name="search_function",
        label="Search Function",
        type="text",
        required=False,
        default="",
        description=(
            "Search-tool function used for every query kind. Leave empty for Serper's "
            "search_web / search_news / search_scholar routing."
        ),
    ),
    ConfigField(
        name="search_channels",
        label="Extra Search Channels",
        type="string[]",
        required=False,
        default=[],
        description=(
            "Additional search backends the reasoner can query as search kinds. Each entry is "
            "either a mapping with name, tool, function, and description keys, or the compact "
            'string form "name=tool.function|description".'
        ),
    ),
]


@register_tool_with_metadata(
    name=TOOL_NAME,
    display_name="Deep Research",
    description=(
        "Run a bounded cited web-research loop using the caller's active MindRoom model. "
        "Requires a configured web-search tool (the built-in Serper tool by default)."
    ),
    category=ToolCategory.RESEARCH,
    status=ToolStatus.REQUIRES_CONFIG,
    setup_type=SetupType.API_KEY,
    icon="FaSearchengin",
    icon_color="text-blue-600",
    config_fields=_CONFIG_FIELDS,
    agent_override_fields=_CONFIG_FIELDS,
)
def deep_research_factory() -> type[DeepResearchTools]:
    """Factory function for the deep-research toolkit."""
    return DeepResearchTools


__all__ = ["DeepResearchTools", "deep_research_factory"]
