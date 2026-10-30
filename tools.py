# ruff: noqa: INP001
"""Agent-facing tools for the MindRoom deep-research plugin."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agno.tools import Toolkit
from agno.tools.function import ToolResult
from agno.tools.serper import SerperTools
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from mindroom.plugin_api import (
    ConfigField,
    SetupType,
    ToolCategory,
    ToolFileAccess,
    ToolStatus,
    ToolValue,
    build_model_provider,
    complete_prompt,
    get_logger,
    get_tool_by_name,
    get_tool_runtime_context,
    register_tool_with_metadata,
    toolkit_functions,
)
from mindroom.turn_model import ModelScope, resolve_turn_model

if TYPE_CHECKING:
    from mindroom.llm.provider import ModelProvider
    from mindroom.plugin_api import ResolvedWorkerTarget, ToolRuntimeContext

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


class _ChannelConfig(BaseModel):
    """One authored search channel: a named tool+function evidence backend."""

    name: str
    tool: str
    function: str
    description: str = ""
    arguments: dict[str, object] | None = None

    @field_validator("name")
    @classmethod
    def _normalize_name(cls, value: str) -> str:
        name = value.strip().lower()
        if not name:
            msg = "channel name must not be empty"
            raise ValueError(msg)
        return name

    @field_validator("tool", "function")
    @classmethod
    def _require_identifier(cls, value: str) -> str:
        identifier = value.strip()
        if not identifier:
            msg = "channel entries need name, tool, and function"
            raise ValueError(msg)
        return identifier

    @model_validator(mode="after")
    def _default_description(self) -> _ChannelConfig:
        self.description = self.description.strip() or f"search via the {self.tool} tool"
        return self


def _coerce_channel_entries(raw: object) -> list[object]:
    """Expand raw channel config into a list of entry objects.

    MindRoom validates string[] settings as lists of strings, so channels
    with structured fields (arguments templates) must be authored as JSON
    object strings; each JSON object string in the list is decoded. A single
    string, such as a comma-joined list, is also accepted: wrapping it in
    brackets re-parses it as a JSON array regardless of commas inside
    quoted values.
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
    """One search channel bound to its resolved toolkit function for a single run."""

    search: Callable[..., object]
    description: str
    arguments: dict[str, object] | None


DEFAULT_MAX_ROUNDS = MAX_ROUNDS_CAP
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
    scope = ModelScope(entity=context.agent_name, room_id=context.room_id, thread_id=context.resolved_thread_id)
    return resolve_turn_model(context.config, context.runtime_paths, scope, None).model_name


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


def _tool_function(functions: Mapping[str, Callable[..., object]], function_name: str) -> Callable[..., object]:
    function = functions.get(function_name)
    if function is None:
        msg = f"Tool function is unavailable: {function_name}"
        raise RuntimeError(msg)
    return function


async def _call_tool_function(function: Callable[..., object], *args: object, **kwargs: object) -> str:
    if inspect.iscoroutinefunction(function):
        raw = await function(*args, **kwargs)
    else:
        raw = await asyncio.to_thread(function, *args, **kwargs)
        if inspect.isawaitable(raw):
            raw = await raw
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
    search: Callable[..., object],
    query: str,
    *,
    num_results: int,
    arguments: dict[str, object] | None = None,
) -> str:
    if arguments is not None:
        # Keyword-shaped call for functions that take structured kwargs
        # (e.g. MCP tools); the template carries the query placement.
        substituted = _substitute_placeholders(arguments, query=query, num_results=num_results)
        return await _call_tool_function(search, **(substituted if isinstance(substituted, dict) else {}))
    kwargs = {"num_results": num_results} if _accepts_keyword(search, "num_results") else {}
    return await _call_tool_function(search, query, **kwargs)


def _result_text(raw: object) -> str:
    """Normalize a search call's return value to text.

    Search callables return plain text; MindRoom's MCP toolkits return an
    agno ToolResult or a MindRoom ToolValue whose content is already
    normalized to a string.
    Anything else is a contract violation and fails loudly.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (ToolResult, ToolValue)):
        return raw.content
    msg = f"Search function returned an unsupported result type: {type(raw).__name__}"
    raise TypeError(msg)


def _hides_functions(key: str, value: object) -> bool:
    # Agno toolkits name their function switches `enable_<function>` and `all`; an enabled one is kept,
    # since some functions, such as Firecrawl's search, are off unless enabled.
    return key in {"include_tools", "exclude_tools"} or ((key == "all" or key.startswith("enable_")) and not value)


def _authored_tool_overrides(context: ToolRuntimeContext, tool_name: str) -> dict[str, object]:
    """Return the calling agent's settings for one tool, without the toggles that hide functions from the agent.

    An agent may hide a function from itself, for example a second `search_web`, while this tool still calls it.
    """
    for entry in context.config.get_agent(context.agent_name).tools:
        if entry.name == tool_name:
            return {key: value for key, value in entry.overrides.items() if not _hides_functions(key, value)}
    return {}


async def _emit_message(context: ToolRuntimeContext, text: str, *, timeout_seconds: float = PROGRESS_EMIT_TIMEOUT_SECONDS) -> None:
    message_sender = context.hooks.message_sender
    if message_sender is None:
        return
    if timeout_seconds <= 0:
        LOGGER.warning("deep_research_progress_emit_skipped_budget_exhausted")
        return
    try:
        async with asyncio.timeout(min(PROGRESS_EMIT_TIMEOUT_SECONDS, timeout_seconds)):
            await message_sender(
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
            execution_identity = context.execution_identity
            live_model = build_model_provider(
                context.config,
                context.runtime_paths,
                model_name,
                execution_identity=execution_identity,
            )
            extract_live_model = (
                live_model
                if extract_model_name == model_name
                else build_model_provider(
                    context.config,
                    context.runtime_paths,
                    extract_model_name,
                    execution_identity=execution_identity,
                )
            )
            ground_live_model = (
                live_model
                if ground_model_name == model_name
                else build_model_provider(
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
                tools=context.plugins.tools,
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

        # The search tool must answer web queries; a missing news or scholar
        # search is dropped with a warning, like an extra channel below.
        channel_warnings: list[str] = []
        wanted_functions = {"web": self.search_function} if self.search_function else SERPER_SEARCH_FUNCTIONS
        search_toolkit_functions = toolkit_functions(search_toolkit)
        search_functions: dict[str, Callable[..., object]] = {}
        for kind, function_name in wanted_functions.items():
            try:
                search_functions[kind] = _tool_function(search_toolkit_functions, function_name)
            except RuntimeError as exc:
                if kind == "web":
                    return _error(f"Search tool '{search_tool_name}' cannot search the web: {exc}")
                LOGGER.warning("deep_research_search_kind_unavailable", kind=kind, error=str(exc))
                channel_warnings.append(f"search kind '{kind}' unavailable: {exc}")

        # Extra search channels degrade gracefully: an unavailable channel is
        # dropped (and reported in warnings) instead of failing the run, and
        # the reasoner is only told about channels that actually resolved.
        channels: dict[str, _ResolvedChannel] = {}
        channel_worker_target: ResolvedWorkerTarget | None = None
        channel_shared_services: frozenset[str] | None = None
        if self.search_channels:
            # A resolution failure propagates rather than degrading to an
            # unscoped session that can never be signed in.
            channel_worker_target = context.resolve_worker_target()
            if channel_worker_target.worker_scope is not None:
                channel_shared_services = context.config.get_worker_grantable_credentials()
        for channel_config in self.search_channels:
            try:
                channel_toolkit = get_tool_by_name(
                    channel_config.tool,
                    context.runtime_paths,
                    tools=context.plugins.tools,
                    tool_config_overrides=_authored_tool_overrides(context, channel_config.tool) or None,
                    allowed_shared_services=channel_shared_services,
                    worker_target=channel_worker_target,
                    runtime_config=context.config,
                )
                channel_search = _tool_function(toolkit_functions(channel_toolkit), channel_config.function)
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
                search=channel_search,
                description=channel_config.description,
                arguments=channel_config.arguments,
            )

        prompt_channels = [(kind, BASE_CHANNEL_DESCRIPTIONS[kind]) for kind in search_functions]
        prompt_channels.extend((name, channel.description) for name, channel in channels.items())

        try:
            website = get_tool_by_name(
                "website",
                context.runtime_paths,
                tools=context.plugins.tools,
                worker_target=None,
            )
            read_url = _tool_function(toolkit_functions(website), "read_url")
        except Exception as exc:
            LOGGER.warning("deep_research_website_unavailable", error=str(exc))
            return _error(f"Website reader is unavailable: {exc}")

        wrapper_start = time.monotonic()

        def remaining_wall_clock_seconds() -> float:
            return max(0.0, wall_clock_seconds - (time.monotonic() - wrapper_start))

        async def reason_fn(prompt: str) -> object:
            return await _ask_model(live_model, prompt, schema=ResearchStep)

        async def extract_fn(prompt: str) -> object:
            return await _ask_model(extract_live_model, prompt, schema=Extraction)

        async def synthesize_fn(prompt: str) -> str:
            content = await _ask_model(live_model, prompt, schema=None)
            return content if isinstance(content, str) else str(content)

        async def ground_fn(prompt: str) -> object:
            return await _ask_model(ground_live_model, prompt, schema=GroundingCheck)

        async def search_fn(query: SearchQuery, limit: int) -> list[SearchHit]:
            channel = channels.get(query.kind)
            if channel is not None:
                raw = await _call_search_function(
                    channel.search,
                    query.query,
                    num_results=limit,
                    arguments=channel.arguments,
                )
            else:
                search = search_functions.get(query.kind, search_functions["web"])
                raw = await _call_search_function(search, query.query, num_results=limit)
            return _parse_search_results(raw)

        async def read_fn(url: str) -> Page:
            title, text = _extract_text_from_website_payload(await _call_tool_function(read_url, url))
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

        searches = result.stats.get("searches", 0)
        if not result.sources and searches and result.stats.get("search_failures", 0) >= searches:
            return _error("Every search failed, so deep_research found no sources.", warnings=result.warnings)

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


async def _ask_model(model: ModelProvider, prompt: str, *, schema: type[BaseModel] | None) -> object:
    """Send one prompt to ``model`` and return its content, parsed into ``schema`` when it parses."""
    response = await complete_prompt(model, prompt, schema=schema)
    return response.content


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
    file_access=ToolFileAccess.NONE,
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
