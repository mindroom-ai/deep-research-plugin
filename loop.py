# ruff: noqa: INP001
"""Pure IterResearch control flow for the deep-research plugin."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError

from .prompts import extractor_prompt, reasoner_prompt, synthesize_prompt

CONFIDENCE_STOP = 0.8
MAX_ROUNDS_CAP = 100
WALL_CLOCK_SECONDS_CAP = 150 * 60
MAX_QUERIES_PER_ROUND = 5
MAX_QUERIES_PER_ROUND_CAP = 10
RESULTS_PER_QUERY = 10
RESULTS_PER_QUERY_CAP = 10
MAX_READS_PER_ROUND = 10
MAX_READS_PER_ROUND_CAP = 20
REPORT_TOKEN_CAP = 8_000
REPORT_CHAR_CAP = REPORT_TOKEN_CAP * 4
NO_PROGRESS_LIMIT = 2

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


class LoopResult(BaseModel):
    """Final pure-loop result before tool envelope serialization."""

    question: str
    report: str
    sources: list[dict[str, object]]
    sources_considered: int
    sources_used: int
    confidence: float
    rounds_used: int
    stopped_reason: Literal["confident", "model_finished", "max_rounds", "wall_clock", "no_progress"]
    elapsed_seconds: float
    warnings: list[str]


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
        remaining = self.remaining()
        if remaining <= 0:
            raise _WallClockExpired(label)
        timeout = asyncio.timeout(remaining)
        try:
            async with timeout:
                return await fn()
        except TimeoutError as exc:
            if timeout.expired():
                raise _WallClockExpired(label) from exc
            raise


def clamp_int(value: int, *, minimum: int, maximum: int) -> int:
    """Clamp an integer value to a closed range."""
    return max(minimum, min(maximum, int(value)))


def truncate_report(text: str, max_chars: int = REPORT_CHAR_CAP) -> str:
    """Hard-truncate the rolling report to the configured approximate token cap."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 28].rstrip() + "\n\n[truncated to budget]"


def _validate_structured(value: object, schema: type[StructuredT]) -> StructuredT:
    if isinstance(value, schema):
        return value
    if isinstance(value, str):
        try:
            return schema.model_validate_json(value)
        except ValidationError:
            start = value.find("{")
            end = value.rfind("}")
            if start >= 0 and end > start:
                return schema.model_validate_json(value[start : end + 1])
            raise
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


def _evidence_line(source: SourceRecord, text: str) -> str:
    if text:
        return f"[{source.id}] {source.title} - {source.url}: {text}"
    return f"[{source.id}] {source.title} - {source.url}"


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
    start = clock() if budget_start is None else budget_start
    budget = _WallClockBudget(start=start, seconds=wall_clock_seconds, clock=clock)
    report = ""
    confidence = 0.0
    rounds_used = 0
    stopped_reason: Literal["confident", "model_finished", "max_rounds", "wall_clock", "no_progress"] = "max_rounds"
    sources_by_url: dict[str, SourceRecord] = {}
    sources_considered = 0
    no_progress_counter = 0
    seen_evidence: set[str] = set()

    try:
        seed_hits = _coerce_hits(
            await budget.wait_for(
                "seed search",
                lambda: search_fn(SearchQuery(query=question), results_per_query),
            ),
        )
    except _WallClockExpired as exc:
        warnings.append(_wall_clock_warning(exc.label))
        seed_hits = []
        stopped_reason = "wall_clock"
    except Exception as exc:
        warnings.append(f"seed search failed: {exc}")
        seed_hits = []
    pending_evidence: list[str] = []
    for hit in seed_hits[:results_per_query]:
        sources_considered += 1
        evidence_line = _candidate_line(hit)
        seen_evidence.add(evidence_line)
        pending_evidence.append(evidence_line)
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
            budget_left=f"{max_rounds - rounds_used} rounds; {wall_clock_seconds - int(clock() - start)} seconds",
        )
        fallback_step = ResearchStep(
            thought="Structured reasoner output failed; finishing with current report.",
            updated_report=report,
            open_questions=[],
            confidence=confidence,
            next_action="finish",
        )
        def fallback_reasoner_step(step: ResearchStep = fallback_step) -> ResearchStep:
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

        if confidence >= CONFIDENCE_STOP:
            stopped_reason = "confident"
            break
        if step.next_action == "finish":
            stopped_reason = "model_finished"
            break

        evidence: list[str] = []
        made_progress = False
        if step.next_action == "search":
            for query in step.search_queries[:max_queries_per_round]:
                try:
                    hits = _coerce_hits(
                        await budget.wait_for(
                            f"search for {query.query}",
                            lambda query=query: search_fn(query, results_per_query),
                        ),
                    )
                except _WallClockExpired as exc:
                    stopped_reason = "wall_clock"
                    warnings.append(_wall_clock_warning(exc.label))
                    break
                except Exception as exc:
                    warnings.append(f"search failed for {query.query}: {exc}")
                    continue
                for hit in hits[:results_per_query]:
                    sources_considered += 1
                    evidence_line = _candidate_line(hit)
                    if evidence_line not in seen_evidence:
                        made_progress = True
                    seen_evidence.add(evidence_line)
                    evidence.append(evidence_line)
        elif step.next_action == "read":
            for url in step.read_urls[:max_reads_per_round]:
                sources_considered += 1
                try:
                    page = _coerce_page(
                        await budget.wait_for(f"read for {url}", lambda url=url: read_fn(url)),
                        url,
                    )
                except _WallClockExpired as exc:
                    stopped_reason = "wall_clock"
                    warnings.append(_wall_clock_warning(exc.label))
                    break
                except Exception as exc:
                    warnings.append(f"read failed for {url}: {exc}")
                    continue
                try:
                    extract = await _call_structured_with_retry(
                        extract_fn,
                        extractor_prompt(question=question, url=url, page_text=page.text),
                        Extraction,
                        lambda: Extraction(facts=[], relevant=False),
                        warnings,
                        "extractor",
                        budget,
                    )
                except _WallClockExpired as exc:
                    stopped_reason = "wall_clock"
                    warnings.append(_wall_clock_warning(exc.label))
                    break
                if extract.relevant:
                    source, is_new = _register_source(sources_by_url, url=page.url, title=page.title, snippet="")
                    for fact in extract.facts:
                        fact = fact.strip()
                        if not fact:
                            continue
                        evidence_line = f"[{source.id}] {fact}"
                        if is_new or evidence_line not in seen_evidence:
                            made_progress = True
                        seen_evidence.add(evidence_line)
                        evidence.append(evidence_line)

        if stopped_reason == "wall_clock":
            pending_evidence = evidence or ["(no new evidence)"]
            break

        if not made_progress:
            no_progress_counter += 1
        else:
            no_progress_counter = 0
        if no_progress_counter >= NO_PROGRESS_LIMIT:
            stopped_reason = "no_progress"
            pending_evidence = evidence or ["(no new evidence)"]
            break

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
    )
    fallback_report = _final_workspace(report, pending_evidence).strip()
    if not fallback_report:
        fallback_report = "Research stopped because the wall-clock budget expired before a final report was produced."
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
        stopped_reason = "wall_clock"
        warnings.append(_wall_clock_warning(exc.label))
        final_report = _ensure_sources_section(fallback_report, sources, sources_by_url)
    except Exception as exc:
        warnings.append(f"final synthesis failed: {exc}")
        final_report = _ensure_sources_section(fallback_report, sources, sources_by_url)
    elapsed = budget.elapsed()
    sources_used = _cited_source_count(final_report, sources_by_url)

    return LoopResult(
        question=question,
        report=final_report,
        sources=sources,
        sources_considered=sources_considered,
        sources_used=sources_used,
        confidence=confidence,
        rounds_used=rounds_used,
        stopped_reason=stopped_reason,
        elapsed_seconds=elapsed,
        warnings=warnings,
    )
