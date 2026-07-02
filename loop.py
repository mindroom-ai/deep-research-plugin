# ruff: noqa: INP001
"""Pure IterResearch control flow for the deep-research plugin."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError

from .prompts import extractor_prompt, heavy_synthesize_prompt, reasoner_prompt, synthesize_prompt

CONFIDENCE_STOP = 0.8
MAX_ROUNDS_CAP = 100
WALL_CLOCK_SECONDS_CAP = 150 * 60
MAX_QUERIES_PER_ROUND = 5
MAX_QUERIES_PER_ROUND_CAP = 10
RESULTS_PER_QUERY = 10
RESULTS_PER_QUERY_CAP = 30
MAX_READS_PER_ROUND = 10
MAX_READS_PER_ROUND_CAP = 20
SNIPPET_SOURCES_PER_ROUND_CAP = 5
REPORT_TOKEN_CAP = 16_000
REPORT_CHAR_CAP = REPORT_TOKEN_CAP * 4
NO_PROGRESS_LIMIT = 3
OP_TIMEOUT_SECONDS = 120.0
RETRY_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 0.5
MAX_CONCURRENCY = 5
FACTS_PER_SOURCE_CAP = 20
EVIDENCE_DIGEST_CHAR_CAP = 20_000
RECENT_QUERIES_PROMPT_CAP = 15
FETCHED_URLS_PROMPT_CAP = 20
UNVETTED_EXCERPT_CHARS = 400
PARALLEL_RESEARCHERS_CAP = 4
SYNTHESIS_RESERVE_SECONDS = 180
LOOP_SYNTHESIS_RESERVE_SECONDS = 90
RESEARCH_ANGLES = (
    "comprehensive overview: map the topic broadly and establish the key facts",
    "primary sources: official data, original documents, and concrete numbers",
    "skeptical verification: counterevidence, criticisms, and known failure modes",
    "recency: the latest developments, announcements, and current state",
)

StructuredT = TypeVar("StructuredT", bound=BaseModel)
AwaitedT = TypeVar("AwaitedT")


class SearchQuery(BaseModel):
    """One search request planned by the reasoner."""

    query: str
    kind: Literal["web", "scholar", "news"] = "web"


class ResearchStep(BaseModel):
    """Structured reasoner output for one research round."""

    thought: str
    updated_report: str
    open_questions: list[str]
    confidence: float = Field(ge=0, le=1)
    next_action: Literal["search", "read", "finish"]
    search_queries: list[SearchQuery] = []
    read_urls: list[str] = []
    cite_snippet_urls: list[str] = []


class Extraction(BaseModel):
    """Structured extractor output for one fetched page."""

    facts: list[str]
    relevant: bool


class SearchHit(BaseModel):
    """Normalized search result."""

    url: str
    title: str = ""
    snippet: str = ""


class Page(BaseModel):
    """Normalized fetched page."""

    url: str
    title: str = ""
    text: str


class SourceRecord(BaseModel):
    """Stable source registry entry."""

    id: int
    url: str
    title: str
    snippet: str = ""


StoppedReason = Literal[
    "confident",
    "model_finished",
    "max_rounds",
    "wall_clock",
    "no_progress",
    "synthesis_truncated",
]


class LoopResult(BaseModel):
    """Final pure-loop result before tool envelope serialization."""

    question: str
    report: str
    sources: list[dict[str, object]]
    sources_considered: int
    sources_used: int
    confidence: float
    rounds_used: int
    stopped_reason: StoppedReason
    elapsed_seconds: float
    warnings: list[str]
    stats: dict[str, int] = {}


ReasonFn = Callable[[str], Awaitable[object]]
ExtractFn = Callable[[str], Awaitable[object]]
SearchFn = Callable[[SearchQuery, int], Awaitable[Sequence[SearchHit | dict[str, object]]]]
ReadFn = Callable[[str], Awaitable[Page | dict[str, object] | str]]
SynthesizeFn = Callable[[str], Awaitable[str]]
EmitFn = Callable[[dict[str, object]], Awaitable[None]]
ClockFn = Callable[[], float]


class _WallClockExpired(TimeoutError):
    """Raised when the remaining research wall-clock budget expires."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


class _OpTimeoutError(TimeoutError):
    """Raised when one network operation exceeds its per-operation timeout."""

    def __init__(self, label: str) -> None:
        super().__init__(label)
        self.label = label


@dataclass(frozen=True)
class _WallClockBudget:
    start: float
    seconds: int
    clock: ClockFn

    def elapsed(self) -> float:
        return self.clock() - self.start

    def remaining(self) -> float:
        return self.seconds - self.elapsed()

    def expired(self) -> bool:
        return self.remaining() <= 0

    async def wait_for(self, label: str, fn: Callable[[], Awaitable[AwaitedT]]) -> AwaitedT:
        return await self.wait_for_op(label, fn, float("inf"))

    async def wait_for_op(
        self,
        label: str,
        fn: Callable[[], Awaitable[AwaitedT]],
        op_timeout_seconds: float,
    ) -> AwaitedT:
        """Like wait_for, but additionally bounded by a per-operation timeout."""
        remaining = self.remaining()
        if remaining <= 0:
            raise _WallClockExpired(label)
        budget_limited = remaining <= op_timeout_seconds
        timeout = asyncio.timeout(min(remaining, op_timeout_seconds))
        try:
            async with timeout:
                return await fn()
        except TimeoutError as exc:
            if timeout.expired():
                if budget_limited:
                    raise _WallClockExpired(label) from exc
                raise _OpTimeoutError(label) from exc
            raise


async def _attempt_with_retries(
    label: str,
    fn: Callable[[], Awaitable[AwaitedT]],
    *,
    budget: _WallClockBudget,
    op_timeout_seconds: float,
    backoff_seconds: float,
    attempts: int = RETRY_ATTEMPTS,
) -> AwaitedT:
    """Run one bounded network operation with transient-failure retries."""
    for attempt in range(attempts):
        try:
            return await budget.wait_for_op(label, fn, op_timeout_seconds)
        except _WallClockExpired:
            raise
        except Exception:
            if attempt + 1 >= attempts or budget.remaining() <= backoff_seconds:
                raise
            await asyncio.sleep(backoff_seconds)
    msg = f"retry loop exhausted for {label}"  # pragma: no cover
    raise RuntimeError(msg)  # pragma: no cover


def clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    """Clamp an integer value to a closed range."""
    return max(minimum, min(maximum, int(value)))


def truncate_report(text: str, max_chars: int = REPORT_CHAR_CAP) -> str:
    """Truncate the rolling report to its budget, preferring a paragraph boundary."""
    if len(text) <= max_chars:
        return text
    cut = text[: max(0, max_chars - 28)]
    boundary = cut.rfind("\n\n")
    if boundary > max_chars // 2:
        cut = cut[:boundary]
    return (cut.rstrip() + "\n\n[truncated to budget]").strip()


_THINK_BLOCK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.IGNORECASE | re.DOTALL)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _json_candidates(text: str) -> list[str]:
    cleaned = _THINK_BLOCK_RE.sub("", text).strip()
    candidates = [match.group(1).strip() for match in _FENCED_JSON_RE.finditer(cleaned)]
    candidates.append(cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start : end + 1])
    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _validate_structured(value: object, schema: type[StructuredT]) -> StructuredT:
    if isinstance(value, schema):
        return value
    if isinstance(value, str):
        last_error: Exception | None = None
        for candidate in _json_candidates(value):
            try:
                return schema.model_validate_json(candidate)
            except ValidationError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        msg = "structured response was empty"
        raise ValueError(msg)
    return schema.model_validate(value)


async def _call_structured_with_retry(
    fn: Callable[[str], Awaitable[object]],
    prompt: str,
    schema: type[StructuredT],
    fallback: Callable[[], StructuredT],
    warnings: list[str],
    label: str,
    budget: _WallClockBudget,
) -> StructuredT:
    last_error: Exception | None = None
    for attempt in range(2):
        attempt_prompt = prompt
        if attempt == 1:
            attempt_prompt = (
                f"{prompt}\n\nYour previous response was not valid JSON for the schema. "
                "Return ONLY the JSON object."
            )
        try:
            return _validate_structured(await budget.wait_for(label, lambda: fn(attempt_prompt)), schema)
        except Exception as exc:
            if isinstance(exc, _WallClockExpired):
                raise
            last_error = exc
            if attempt == 0:
                continue
    warnings.append(f"{label} structured output failed after retry: {last_error}")
    return fallback()


def _source_lines(sources_by_url: dict[str, SourceRecord]) -> list[str]:
    return [
        f"[{source.id}] {source.title or source.url} - {source.url}"
        for source in sorted(sources_by_url.values(), key=lambda item: item.id)
    ]


def _register_source(
    sources_by_url: dict[str, SourceRecord],
    *,
    url: str,
    title: str = "",
    snippet: str = "",
) -> tuple[SourceRecord, bool]:
    normalized_url = url.strip()
    existing = sources_by_url.get(normalized_url)
    if existing is not None:
        if (not existing.title and title) or (not existing.snippet and snippet):
            sources_by_url[normalized_url] = existing.model_copy(
                update={
                    "title": existing.title or title,
                    "snippet": existing.snippet or snippet,
                },
            )
            existing = sources_by_url[normalized_url]
        return existing, False

    record = SourceRecord(
        id=len(sources_by_url) + 1,
        url=normalized_url,
        title=title.strip() or normalized_url,
        snippet=snippet.strip(),
    )
    sources_by_url[normalized_url] = record
    return record, True


def _coerce_hits(raw_hits: Sequence[SearchHit | dict[str, object]]) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for raw_hit in raw_hits:
        try:
            hit = raw_hit if isinstance(raw_hit, SearchHit) else SearchHit.model_validate(raw_hit)
        except ValidationError:
            continue
        if hit.url:
            hits.append(hit)
    return hits


def _coerce_page(raw_page: Page | dict[str, object] | str, url: str) -> Page:
    if isinstance(raw_page, Page):
        return raw_page
    if isinstance(raw_page, str):
        return Page(url=url, title=url, text=raw_page)
    page_data = dict(raw_page)
    page_data.setdefault("url", url)
    if "content" in page_data and "text" not in page_data:
        page_data["text"] = page_data["content"]
    return Page.model_validate(page_data)


def _sources_json(sources_by_url: dict[str, SourceRecord]) -> list[dict[str, object]]:
    return [
        source.model_dump()
        for source in sorted(sources_by_url.values(), key=lambda item: item.id)
    ]


_SOURCE_SECTION_HEADING_RE = re.compile(r"(?im)^#{1,6}\s*(?:sources?|references|bibliography)\s*:?\s*$")
_NUMERIC_CITATION_GROUP_RE = re.compile(r"\[((?:\d+\s*,\s*)*\d+)\]")


def _strip_source_like_tail(report: str) -> str:
    match = _SOURCE_SECTION_HEADING_RE.search(report)
    return report[: match.start()] if match else report


def _report_body(report: str) -> str:
    return _strip_source_like_tail(report)


def _valid_source_ids(sources_by_url: dict[str, SourceRecord]) -> set[int]:
    return {source.id for source in sources_by_url.values()}


def _cited_source_ids(report: str, sources_by_url: dict[str, SourceRecord]) -> set[int]:
    valid_ids = _valid_source_ids(sources_by_url)
    body = _report_body(report)
    cited_ids: set[int] = set()
    for match in _NUMERIC_CITATION_GROUP_RE.finditer(body):
        cited_ids.update(
            cited_id
            for cited_id in (int(part.strip()) for part in match.group(1).split(","))
            if cited_id in valid_ids
        )
    return cited_ids


def _cited_source_count(report: str, sources_by_url: dict[str, SourceRecord]) -> int:
    return len(_cited_source_ids(report, sources_by_url))


def _ensure_sources_section(
    report: str,
    sources: list[dict[str, object]],
    sources_by_url: dict[str, SourceRecord],
) -> str:
    valid_ids = _valid_source_ids(sources_by_url)
    body = _report_body(report).rstrip()

    def keep_valid_citation_group(match: re.Match[str]) -> str:
        kept_ids: list[int] = []
        for cited_id in (int(part.strip()) for part in match.group(1).split(",")):
            if cited_id in valid_ids and cited_id not in kept_ids:
                kept_ids.append(cited_id)
        if not kept_ids:
            return ""
        return "[" + ", ".join(str(cited_id) for cited_id in kept_ids) + "]"

    repaired_body = _NUMERIC_CITATION_GROUP_RE.sub(keep_valid_citation_group, body).rstrip()
    cited_ids = _cited_source_ids(repaired_body, sources_by_url)
    source_lines = [
        f"[{source['id']}] {source.get('title') or source.get('url')} - {source.get('url')}"
        for source in sources
        if source["id"] in cited_ids
    ]
    return f"{repaired_body}\n\n## Sources\n" + ("\n".join(source_lines) if source_lines else "(none)")


def _candidate_line(hit: SearchHit) -> str:
    title = hit.title.strip() or hit.url
    snippet = hit.snippet.strip()
    if snippet:
        return f"Candidate URL: {title} - {hit.url}: {snippet}"
    return f"Candidate URL: {title} - {hit.url}"


def _final_workspace(report: str, pending_evidence: Sequence[str]) -> str:
    useful_evidence = [item for item in pending_evidence if item != "(no new evidence)"]
    if not useful_evidence:
        return report
    evidence_text = "\n".join(f"- {item}" for item in useful_evidence)
    return f"{report.rstrip()}\n\nUnsynthesized evidence from the final round:\n{evidence_text}".strip()


def _evidence_digest(
    facts_by_source: dict[int, list[str]],
    max_chars: int = EVIDENCE_DIGEST_CHAR_CAP,
) -> list[str]:
    """Flatten the per-source fact bank into bounded [n]-prefixed evidence lines."""
    lines: list[str] = []
    total = 0
    for source_id in sorted(facts_by_source):
        for fact in facts_by_source[source_id]:
            line = f"[{source_id}] {fact}"
            total += len(line) + 1
            if total > max_chars:
                lines.append("(evidence digest truncated to budget)")
                return lines
            lines.append(line)
    return lines


def _has_valid_citations(report: str, sources_by_url: dict[str, SourceRecord]) -> bool:
    return _cited_source_count(report, sources_by_url) > 0


def _needs_citation_retry(report: str, sources_by_url: dict[str, SourceRecord]) -> bool:
    return bool(sources_by_url) and not _has_valid_citations(report, sources_by_url)


def _citation_retry_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\nYour previous final answer used no valid source citations. "
        "Rewrite the final answer using at least one valid inline [n] citation from the source registry. "
        "Do not cite candidate URLs or unverified search snippets."
    )


def _wall_clock_warning(label: str) -> str:
    return f"wall clock expired during {label}"


def _search_phase_seconds(wall_clock_seconds: int) -> int:
    """Search-phase deadline that reserves wall clock for the loop's own final synthesis."""
    reserve = min(LOOP_SYNTHESIS_RESERVE_SECONDS, max(0, wall_clock_seconds - 60) // 2)
    return wall_clock_seconds - reserve


async def _finalize_report(
    *,
    final_prompt: str,
    fallback_report: str,
    sources: list[dict[str, object]],
    sources_by_url: dict[str, SourceRecord],
    synthesize_fn: SynthesizeFn,
    budget: _WallClockBudget,
    warnings: list[str],
) -> tuple[str, bool]:
    """Run final synthesis with citation repair; return (report, wall_clock_expired)."""
    try:
        final_report = _ensure_sources_section(
            await budget.wait_for("final synthesis", lambda: synthesize_fn(final_prompt)),
            sources,
            sources_by_url,
        )
        if _needs_citation_retry(final_report, sources_by_url):
            retried_report = _ensure_sources_section(
                await budget.wait_for(
                    "final synthesis citation retry",
                    lambda: synthesize_fn(_citation_retry_prompt(final_prompt)),
                ),
                sources,
                sources_by_url,
            )
            if _has_valid_citations(retried_report, sources_by_url):
                final_report = retried_report
            else:
                warnings.append("final synthesis produced no valid citations after retry")
                fallback_with_sources = _ensure_sources_section(fallback_report, sources, sources_by_url)
                final_report = (
                    fallback_with_sources
                    if _has_valid_citations(fallback_with_sources, sources_by_url)
                    else retried_report
                )
    except _WallClockExpired as exc:
        warnings.append(_wall_clock_warning(exc.label))
        return _ensure_sources_section(fallback_report, sources, sources_by_url), True
    except Exception as exc:
        warnings.append(f"final synthesis failed: {exc}")
        return _ensure_sources_section(fallback_report, sources, sources_by_url), False
    return final_report, False


async def _run_worker_batch(  # noqa: C901
    workers: Sequence[tuple[list[str], Awaitable[None]]],
    *,
    budget: _WallClockBudget,
    warnings: list[str],
    max_concurrency: int,
) -> bool:
    """Run (phase-label, worker) pairs concurrently; return True if the wall clock expired."""
    if not workers:
        return False
    semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def guarded(worker: Awaitable[None]) -> None:
        async with semaphore:
            await worker

    tasks = [asyncio.create_task(guarded(worker)) for _, worker in workers]
    remaining = budget.remaining()
    if remaining <= 0:
        pending = set(tasks)
    else:
        _, pending = await asyncio.wait(tasks, timeout=remaining)
    expired = bool(pending)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    for task, (phase, _) in zip(tasks, workers, strict=True):
        if task in pending:
            warnings.append(_wall_clock_warning(phase[0]))
            continue
        if task.cancelled():
            warnings.append(f"{phase[0]} was cancelled")
            continue
        exc = task.exception()
        if isinstance(exc, _WallClockExpired):
            expired = True
            warnings.append(_wall_clock_warning(exc.label))
        elif exc is not None:
            warnings.append(f"{phase[0]} failed unexpectedly: {exc}")
    return expired


async def run_research_loop(  # noqa: C901, PLR0912, PLR0915
    *,
    question: str,
    max_rounds: int,
    wall_clock_seconds: int,
    reason_fn: ReasonFn,
    extract_fn: ExtractFn,
    search_fn: SearchFn,
    read_fn: ReadFn,
    synthesize_fn: SynthesizeFn,
    emit_fn: EmitFn | None = None,
    clock: ClockFn = time.monotonic,
    budget_start: float | None = None,
    max_queries_per_round: int = MAX_QUERIES_PER_ROUND,
    results_per_query: int = RESULTS_PER_QUERY,
    max_reads_per_round: int = MAX_READS_PER_ROUND,
    report_char_cap: int = REPORT_CHAR_CAP,
    op_timeout_seconds: float = OP_TIMEOUT_SECONDS,
    retry_backoff_seconds: float = RETRY_BACKOFF_SECONDS,
    max_concurrency: int = MAX_CONCURRENCY,
    angle: str = "",
) -> LoopResult:
    """Run the bounded research loop using injected LLM and network callables."""
    max_rounds = clamp_int(max_rounds, minimum=1, maximum=MAX_ROUNDS_CAP)
    wall_clock_seconds = clamp_int(wall_clock_seconds, minimum=60, maximum=WALL_CLOCK_SECONDS_CAP)
    max_queries_per_round = clamp_int(
        max_queries_per_round,
        minimum=1,
        maximum=MAX_QUERIES_PER_ROUND_CAP,
    )
    results_per_query = clamp_int(results_per_query, minimum=1, maximum=RESULTS_PER_QUERY_CAP)
    max_reads_per_round = clamp_int(max_reads_per_round, minimum=1, maximum=MAX_READS_PER_ROUND_CAP)

    warnings: list[str] = []
    stats = {
        "searches": 0,
        "search_attempts": 0,
        "search_failures": 0,
        "reads": 0,
        "read_attempts": 0,
        "read_failures": 0,
        "extractions": 0,
        "duplicate_queries_skipped": 0,
        "duplicate_reads_skipped": 0,
        "read_snippet_fallbacks": 0,
        "snippet_sources_registered": 0,
    }
    start = clock() if budget_start is None else budget_start
    total_budget = _WallClockBudget(start=start, seconds=wall_clock_seconds, clock=clock)
    # Search/read rounds run against a shorter deadline so final synthesis
    # always keeps a reserved slice of the wall clock instead of starting
    # with whatever the last round happened to leave over.
    budget = _WallClockBudget(start=start, seconds=_search_phase_seconds(wall_clock_seconds), clock=clock)
    report = ""
    confidence = 0.0
    rounds_used = 0
    stopped_reason: StoppedReason = "max_rounds"
    sources_by_url: dict[str, SourceRecord] = {}
    facts_by_source: dict[int, list[str]] = {}
    considered_urls: set[str] = set()
    executed_query_keys: set[str] = set()
    executed_queries: list[str] = []
    attempted_read_urls: set[str] = set()
    attempted_read_order: list[str] = []
    candidate_meta: dict[str, tuple[str, str]] = {}
    no_progress_counter = 0
    seen_evidence: set[str] = set()

    def _query_key(query: SearchQuery) -> str:
        return f"{query.kind}:{' '.join(query.query.lower().split())}"

    async def _search_once(query: SearchQuery, label: str) -> list[SearchHit]:
        stats["searches"] += 1

        async def attempt() -> Sequence[SearchHit | dict[str, object]]:
            stats["search_attempts"] += 1
            return await search_fn(query, results_per_query)

        raw = await _attempt_with_retries(
            label,
            attempt,
            budget=budget,
            op_timeout_seconds=op_timeout_seconds,
            backoff_seconds=retry_backoff_seconds,
        )
        return _coerce_hits(raw)

    def _note_candidate(hit: SearchHit) -> str:
        url = hit.url.strip()
        considered_urls.add(url)
        candidate_meta.setdefault(url, (hit.title, hit.snippet))
        return _candidate_line(hit)

    seed_query = SearchQuery(query=question)
    executed_query_keys.add(_query_key(seed_query))
    executed_queries.append(question)
    try:
        seed_hits = await _search_once(seed_query, "seed search")
    except _WallClockExpired as exc:
        warnings.append(_wall_clock_warning(exc.label))
        seed_hits = []
        stopped_reason = "wall_clock"
    except _OpTimeoutError as exc:
        stats["search_failures"] += 1
        warnings.append(f"timed out during {exc.label}")
        seed_hits = []
    except Exception as exc:
        stats["search_failures"] += 1
        warnings.append(f"seed search failed: {exc}")
        seed_hits = []
    pending_evidence: list[str] = []
    for hit in seed_hits[:results_per_query]:
        evidence_line = _note_candidate(hit)
        if evidence_line not in seen_evidence:
            pending_evidence.append(evidence_line)
        seen_evidence.add(evidence_line)
    if not pending_evidence:
        pending_evidence = ["(no new evidence)"]

    while rounds_used < max_rounds and stopped_reason != "wall_clock":
        if budget.expired():
            stopped_reason = "wall_clock"
            warnings.append(_wall_clock_warning("round start"))
            break

        rounds_used += 1
        prompt = reasoner_prompt(
            question=question,
            report=report,
            pending_evidence=pending_evidence,
            sources=_source_lines(sources_by_url),
            budget_left=f"{max_rounds - rounds_used} rounds; {max(0, int(budget.remaining()))} seconds",
            max_queries=max_queries_per_round,
            max_reads=max_reads_per_round,
            recent_queries=executed_queries[-RECENT_QUERIES_PROMPT_CAP:],
            fetched_urls=attempted_read_order[-FETCHED_URLS_PROMPT_CAP:],
            angle=angle,
        )
        fallback_step = ResearchStep(
            thought="Structured reasoner output failed; skipping this round.",
            updated_report=report,
            open_questions=[],
            confidence=confidence,
            next_action="search",
        )
        reasoner_failed = False

        def fallback_reasoner_step(step: ResearchStep = fallback_step) -> ResearchStep:
            nonlocal reasoner_failed
            reasoner_failed = True
            return step

        try:
            step = await _call_structured_with_retry(
                reason_fn,
                prompt,
                ResearchStep,
                fallback_reasoner_step,
                warnings,
                "reasoner",
                budget,
            )
        except _WallClockExpired as exc:
            stopped_reason = "wall_clock"
            warnings.append(_wall_clock_warning(exc.label))
            rounds_used -= 1
            break
        report = truncate_report(step.updated_report, report_char_cap)
        confidence = step.confidence

        if emit_fn is not None:
            try:
                await budget.wait_for(
                    "progress emit",
                    lambda: emit_fn(
                        {
                            "round": rounds_used,
                            "max_rounds": max_rounds,
                            "thought": step.thought,
                            "confidence": confidence,
                            "next_action": step.next_action,
                            "search_queries": [query.model_dump() for query in step.search_queries],
                            "read_urls": step.read_urls,
                        },
                    ),
                )
            except _WallClockExpired as exc:
                stopped_reason = "wall_clock"
                warnings.append(_wall_clock_warning(exc.label))
                break

        # Reasoner-nominated candidates become citable unvetted sources from
        # their search snippet alone, without spending a page read. Only URLs
        # that actually appeared as search hits (candidate_meta) qualify, so
        # a hallucinated URL cannot enter the registry. Runs before the stop
        # checks so a final "cite these and finish" step still lands.
        snippet_evidence: list[str] = []
        registered_snippet_sources = 0
        for raw_url in step.cite_snippet_urls:
            if registered_snippet_sources >= SNIPPET_SOURCES_PER_ROUND_CAP:
                break
            url = raw_url.strip()
            if not url or url in sources_by_url:
                continue
            title, snippet = candidate_meta.get(url, ("", ""))
            snippet = snippet.strip()
            if not snippet:
                continue
            registered_snippet_sources += 1
            stats["snippet_sources_registered"] += 1
            considered_urls.add(url)
            source, _ = _register_source(sources_by_url, url=url, title=title, snippet=snippet)
            fact = f"Unvetted search snippet: {snippet}"
            source_facts = facts_by_source.setdefault(source.id, [])
            if fact not in source_facts and len(source_facts) < FACTS_PER_SOURCE_CAP:
                source_facts.append(fact)
            snippet_evidence.append(f"[{source.id}] {fact}")

        if confidence >= CONFIDENCE_STOP and sources_by_url:
            stopped_reason = "confident"
            break
        if step.next_action == "finish":
            stopped_reason = "model_finished"
            break

        planned_queries: list[SearchQuery] = []
        for query in step.search_queries:
            if len(planned_queries) >= max_queries_per_round:
                break
            if not query.query.strip():
                continue
            key = _query_key(query)
            if key in executed_query_keys:
                stats["duplicate_queries_skipped"] += 1
                continue
            executed_query_keys.add(key)
            executed_queries.append(query.query)
            planned_queries.append(query)

        planned_urls: list[str] = []
        for raw_url in step.read_urls:
            if len(planned_urls) >= max_reads_per_round:
                break
            url = raw_url.strip()
            if not url:
                continue
            if url in attempted_read_urls:
                stats["duplicate_reads_skipped"] += 1
                continue
            attempted_read_urls.add(url)
            attempted_read_order.append(url)
            # A directly requested URL counts as considered even if the read
            # later fails; search hits are counted in _note_candidate.
            considered_urls.add(url)
            planned_urls.append(url)

        search_results: dict[int, list[SearchHit]] = {}
        read_results: dict[int, tuple[str, Page, Extraction]] = {}

        async def search_worker(index: int, query: SearchQuery, phase: list[str]) -> None:
            try:
                search_results[index] = await _search_once(query, phase[0])
            except _WallClockExpired:
                raise
            except _OpTimeoutError:
                stats["search_failures"] += 1
                warnings.append(f"timed out during {phase[0]}")
            except Exception as exc:
                stats["search_failures"] += 1
                warnings.append(f"search failed for {query.query}: {exc}")

        async def read_worker(index: int, url: str, phase: list[str]) -> None:
            stats["reads"] += 1

            async def attempt() -> Page | dict[str, object] | str:
                stats["read_attempts"] += 1
                return await read_fn(url)

            def snippet_fallback() -> None:
                # A blocked or failing page (e.g. a 403 anti-bot response)
                # degrades to its search snippet instead of vanishing from
                # the evidence pool; the fact is labeled unvetted. Writing
                # into this worker's own read_results slot keeps the fallback
                # on the same merge path as a successful read — if the batch
                # structure ever changes, this write must move with it.
                _, snippet = candidate_meta.get(url, ("", ""))
                snippet = snippet.strip()
                if not snippet:
                    return
                stats["read_snippet_fallbacks"] += 1
                read_results[index] = (
                    url,
                    Page(url=url, title="", text=""),
                    Extraction(facts=[f"Unvetted search snippet: {snippet}"], relevant=True),
                )

            try:
                page = _coerce_page(
                    await _attempt_with_retries(
                        phase[0],
                        attempt,
                        budget=budget,
                        op_timeout_seconds=op_timeout_seconds,
                        backoff_seconds=retry_backoff_seconds,
                    ),
                    url,
                )
            except _WallClockExpired:
                raise
            except _OpTimeoutError:
                stats["read_failures"] += 1
                warnings.append(f"timed out during {phase[0]}")
                snippet_fallback()
                return
            except Exception as exc:
                stats["read_failures"] += 1
                warnings.append(f"read failed for {url}: {exc}")
                snippet_fallback()
                return
            phase[0] = f"extractor for {url}"
            stats["extractions"] += 1
            excerpt = " ".join(page.text.split())[:UNVETTED_EXCERPT_CHARS]

            def fallback_extraction() -> Extraction:
                if excerpt:
                    return Extraction(facts=[f"Unvetted page excerpt: {excerpt}"], relevant=True)
                return Extraction(facts=[], relevant=False)

            extraction = await _call_structured_with_retry(
                extract_fn,
                extractor_prompt(question=question, url=url, page_text=page.text),
                Extraction,
                fallback_extraction,
                warnings,
                phase[0],
                budget,
            )
            read_results[index] = (url, page, extraction)

        # Each worker carries a single-element mutable phase label that it
        # updates as it advances (read -> extractor), so batch-expiry warnings
        # name the sub-operation that was actually in flight.
        workers: list[tuple[list[str], Awaitable[None]]] = []
        for index, query in enumerate(planned_queries):
            phase = [f"search for {query.query}"]
            workers.append((phase, search_worker(index, query, phase)))
        for index, url in enumerate(planned_urls):
            phase = [f"read for {url}"]
            workers.append((phase, read_worker(index, url, phase)))

        if await _run_worker_batch(workers, budget=budget, warnings=warnings, max_concurrency=max_concurrency):
            stopped_reason = "wall_clock"

        evidence: list[str] = []
        made_progress = False
        for evidence_line in snippet_evidence:
            if evidence_line not in seen_evidence:
                made_progress = True
                evidence.append(evidence_line)
            seen_evidence.add(evidence_line)
        for index in sorted(search_results):
            for hit in search_results[index][:results_per_query]:
                evidence_line = _note_candidate(hit)
                if evidence_line not in seen_evidence:
                    made_progress = True
                    evidence.append(evidence_line)
                seen_evidence.add(evidence_line)

        for index in sorted(read_results):
            url, page, extraction = read_results[index]
            considered_urls.add(url)
            if not extraction.relevant:
                continue
            candidate_title, candidate_snippet = candidate_meta.get(url, ("", ""))
            title = page.title.strip()
            if not title or title == page.url:
                title = candidate_title
            source, is_new = _register_source(
                sources_by_url,
                url=page.url,
                title=title,
                snippet=candidate_snippet,
            )
            source_facts = facts_by_source.setdefault(source.id, [])
            for fact in extraction.facts:
                fact = fact.strip()
                if not fact:
                    continue
                if fact not in source_facts and len(source_facts) < FACTS_PER_SOURCE_CAP:
                    source_facts.append(fact)
                evidence_line = f"[{source.id}] {fact}"
                if is_new or evidence_line not in seen_evidence:
                    made_progress = True
                    evidence.append(evidence_line)
                seen_evidence.add(evidence_line)

        if stopped_reason == "wall_clock":
            pending_evidence = evidence or ["(no new evidence)"]
            break

        if not made_progress:
            no_progress_counter += 1
        else:
            no_progress_counter = 0
        if no_progress_counter >= NO_PROGRESS_LIMIT:
            stopped_reason = "no_progress"
            pending_evidence = evidence or pending_evidence
            break

        # A skipped (failed-reasoner) round never compressed the pending
        # evidence, so keep it visible for the retry round.
        if not reasoner_failed:
            pending_evidence = evidence or ["(no new evidence)"]

        if budget.expired():
            stopped_reason = "wall_clock"
            warnings.append(_wall_clock_warning("round end"))
            break

    sources = _sources_json(sources_by_url)
    final_prompt = synthesize_prompt(
        question=question,
        report=_final_workspace(report, pending_evidence),
        sources=sources,
        evidence=_evidence_digest(facts_by_source),
    )
    fallback_report = _final_workspace(report, pending_evidence).strip()
    if not fallback_report:
        fallback_report = "Research stopped because the wall-clock budget expired before a final report was produced."
    final_report, synthesis_expired = await _finalize_report(
        final_prompt=final_prompt,
        fallback_report=fallback_report,
        sources=sources,
        sources_by_url=sources_by_url,
        synthesize_fn=synthesize_fn,
        budget=total_budget,
        warnings=warnings,
    )
    if synthesis_expired:
        stopped_reason = "synthesis_truncated"
    elapsed = total_budget.elapsed()
    sources_used = _cited_source_count(final_report, sources_by_url)

    return LoopResult(
        question=question,
        report=final_report,
        sources=sources,
        sources_considered=len(considered_urls),
        sources_used=sources_used,
        confidence=confidence,
        rounds_used=rounds_used,
        stopped_reason=stopped_reason,
        elapsed_seconds=elapsed,
        warnings=warnings,
        stats=stats,
    )


def _remap_citation_ids(report_body: str, id_map: dict[int, int]) -> str:
    """Rewrite [n] citation groups through id_map, dropping ids that do not map."""

    def remap_group(match: re.Match[str]) -> str:
        kept_ids: list[int] = []
        for cited_id in (int(part.strip()) for part in match.group(1).split(",")):
            mapped = id_map.get(cited_id)
            if mapped is not None and mapped not in kept_ids:
                kept_ids.append(mapped)
        if not kept_ids:
            return ""
        return "[" + ", ".join(str(cited_id) for cited_id in kept_ids) + "]"

    return _NUMERIC_CITATION_GROUP_RE.sub(remap_group, report_body)


def _merge_researcher_results(
    labeled_results: Sequence[tuple[int, LoopResult]],
) -> tuple[dict[str, SourceRecord], list[tuple[int, str]]]:
    """Merge researcher source registries and remap each report body to global ids.

    Results carry their original 1-based researcher number so a failed
    researcher does not shift the attribution of the survivors.
    """
    merged: dict[str, SourceRecord] = {}
    remapped_reports: list[tuple[int, str]] = []
    for label, result in labeled_results:
        id_map: dict[int, int] = {}
        for source in result.sources:
            url = str(source.get("url") or "").strip()
            if not url:
                continue
            record, _ = _register_source(
                merged,
                url=url,
                title=str(source.get("title") or ""),
                snippet=str(source.get("snippet") or ""),
            )
            id_map[int(source["id"])] = record.id
        remapped_reports.append((label, _remap_citation_ids(_report_body(result.report).rstrip(), id_map)))
    return merged, remapped_reports


def _researcher_wall_clock(wall_clock_seconds: int) -> int:
    """Researcher budget that leaves a synthesis reserve, scaled down for small budgets."""
    reserve = min(SYNTHESIS_RESERVE_SECONDS, max(0, wall_clock_seconds - 60) // 2)
    return max(60, wall_clock_seconds - reserve)


class _SharedCallCache(Generic[AwaitedT]):
    """Share in-flight and completed single-argument calls across researchers.

    Heavy-mode researchers explore the same question concurrently, so they
    frequently pick the same URLs to read (and therefore build identical
    extraction prompts). Routing those calls through one shared task per key
    means each page is fetched and extracted at most once across the fleet.
    Failed or cancelled calls are not cached: the next caller re-issues them,
    which keeps the per-researcher retry semantics intact.

    A cache instance lives for exactly one heavy run and never evicts; its
    size is bounded by that run's read/extraction budget. Do not reuse an
    instance across runs.
    """

    def __init__(self, fn: Callable[[str], Awaitable[AwaitedT]]) -> None:
        self._fn = fn
        self._tasks: dict[str, asyncio.Task[AwaitedT]] = {}
        self.hits = 0

    async def __call__(self, key: str) -> AwaitedT:
        task = self._tasks.get(key)
        if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
            task = asyncio.ensure_future(self._fn(key))
            # Retrieve exceptions eagerly so a task whose awaiters were all
            # cancelled does not log "exception was never retrieved".
            task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
            self._tasks[key] = task
        else:
            self.hits += 1
        # Awaiting a shared task does not propagate this caller's
        # cancellation to it, so other researchers can still use the result.
        return await task

    def cancel_pending(self) -> None:
        """Cancel calls still in flight once no researcher can use them."""
        for task in self._tasks.values():
            if not task.done():
                task.cancel()


_STOP_REASON_PRIORITY: tuple[str, ...] = (
    "confident",
    "model_finished",
    "no_progress",
    "max_rounds",
    "wall_clock",
    "synthesis_truncated",
)


def _combined_stopped_reason(
    results: Sequence[LoopResult],
) -> StoppedReason:
    reasons = {result.stopped_reason for result in results}
    for reason in _STOP_REASON_PRIORITY:
        if reason in reasons:
            return reason  # type: ignore[return-value]
    return "max_rounds"  # pragma: no cover


async def run_heavy_research_loop(
    *,
    question: str,
    researchers: int,
    max_rounds: int,
    wall_clock_seconds: int,
    reason_fn: ReasonFn,
    extract_fn: ExtractFn,
    search_fn: SearchFn,
    read_fn: ReadFn,
    synthesize_fn: SynthesizeFn,
    emit_fn: EmitFn | None = None,
    clock: ClockFn = time.monotonic,
    budget_start: float | None = None,
    max_queries_per_round: int = MAX_QUERIES_PER_ROUND,
    results_per_query: int = RESULTS_PER_QUERY,
    max_reads_per_round: int = MAX_READS_PER_ROUND,
    report_char_cap: int = REPORT_CHAR_CAP,
    op_timeout_seconds: float = OP_TIMEOUT_SECONDS,
    retry_backoff_seconds: float = RETRY_BACKOFF_SECONDS,
    max_concurrency: int = MAX_CONCURRENCY,
) -> LoopResult:
    """Run N research loops on different angles concurrently and synthesize one report.

    This is the IterResearch "Heavy mode" (Research-Synthesis) pattern: each
    researcher explores the question independently, and a final synthesis pass
    integrates their cited reports over a merged source registry.
    """
    researchers = clamp_int(researchers, minimum=1, maximum=PARALLEL_RESEARCHERS_CAP)
    wall_clock_seconds = clamp_int(wall_clock_seconds, minimum=60, maximum=WALL_CLOCK_SECONDS_CAP)

    def researcher_kwargs(index: int) -> dict[str, object]:
        researcher_emit = emit_fn
        if emit_fn is not None and researchers > 1:
            async def tagged_emit(event: dict[str, object], researcher: int = index + 1) -> None:
                await emit_fn({**event, "researcher": researcher})
            researcher_emit = tagged_emit
        return {
            "question": question,
            "max_rounds": max_rounds,
            "reason_fn": reason_fn,
            "extract_fn": extract_fn,
            "search_fn": search_fn,
            "read_fn": read_fn,
            "synthesize_fn": synthesize_fn,
            "emit_fn": researcher_emit,
            "clock": clock,
            "max_queries_per_round": max_queries_per_round,
            "results_per_query": results_per_query,
            "max_reads_per_round": max_reads_per_round,
            "report_char_cap": report_char_cap,
            "op_timeout_seconds": op_timeout_seconds,
            "retry_backoff_seconds": retry_backoff_seconds,
            "max_concurrency": max_concurrency,
        }

    if researchers == 1:
        return await run_research_loop(
            wall_clock_seconds=wall_clock_seconds,
            budget_start=budget_start,
            **researcher_kwargs(0),  # type: ignore[arg-type]
        )

    start = clock() if budget_start is None else budget_start
    budget = _WallClockBudget(start=start, seconds=wall_clock_seconds, clock=clock)
    researcher_wall_clock = _researcher_wall_clock(wall_clock_seconds)

    # Researchers often converge on the same URLs (they all start from the
    # same question), so page reads and extractions are shared across the
    # fleet; researcher_kwargs picks up the rebound callables.
    shared_read_fn: _SharedCallCache[Page | dict[str, object] | str] = _SharedCallCache(read_fn)
    shared_extract_fn: _SharedCallCache[object] = _SharedCallCache(extract_fn)
    read_fn = shared_read_fn
    extract_fn = shared_extract_fn

    raw_results = await asyncio.gather(
        *(
            run_research_loop(
                wall_clock_seconds=researcher_wall_clock,
                budget_start=start,
                angle=RESEARCH_ANGLES[index % len(RESEARCH_ANGLES)],
                **researcher_kwargs(index),  # type: ignore[arg-type]
            )
            for index in range(researchers)
        ),
        return_exceptions=True,
    )
    shared_read_fn.cancel_pending()
    shared_extract_fn.cancel_pending()

    warnings: list[str] = []
    labeled_results: list[tuple[int, LoopResult]] = []
    for index, raw in enumerate(raw_results):
        if isinstance(raw, LoopResult):
            labeled_results.append((index + 1, raw))
            warnings.extend(f"researcher {index + 1}: {warning}" for warning in raw.warnings)
        else:
            warnings.append(f"researcher {index + 1} failed: {raw}")
    if not labeled_results:
        msg = "all researchers failed"
        raise RuntimeError(msg)
    results = [result for _, result in labeled_results]

    merged_sources_by_url, remapped_reports = _merge_researcher_results(labeled_results)
    sources = _sources_json(merged_sources_by_url)
    fallback_report = "\n\n".join(
        f"## Researcher {label} findings\n{body}"
        for label, body in remapped_reports
        if body.strip()
    ).strip() or "All researchers stopped before producing findings."

    final_report, synthesis_expired = await _finalize_report(
        final_prompt=heavy_synthesize_prompt(question=question, reports=remapped_reports, sources=sources),
        fallback_report=fallback_report,
        sources=sources,
        sources_by_url=merged_sources_by_url,
        synthesize_fn=synthesize_fn,
        budget=budget,
        warnings=warnings,
    )

    stats: dict[str, int] = {}
    for result in results:
        for key, value in result.stats.items():
            stats[key] = stats.get(key, 0) + value
    stats["cross_researcher_reads_shared"] = shared_read_fn.hits
    stats["cross_researcher_extracts_shared"] = shared_extract_fn.hits

    return LoopResult(
        question=question,
        report=final_report,
        sources=sources,
        sources_considered=sum(result.sources_considered for result in results),
        sources_used=_cited_source_count(final_report, merged_sources_by_url),
        confidence=max(result.confidence for result in results),
        rounds_used=sum(result.rounds_used for result in results),
        stopped_reason="synthesis_truncated" if synthesis_expired else _combined_stopped_reason(results),
        elapsed_seconds=budget.elapsed(),
        warnings=warnings,
        stats=stats,
    )
