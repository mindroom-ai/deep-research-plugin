# ruff: noqa: INP001
# ruff: noqa: ANN401, D103, EM101, TRY003
"""Tests for the pure deep-research loop."""

from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

PACKAGE_NAME = f"mindroom_plugin_{Path(__file__).resolve().parents[1].name.replace('-', '_')}"
loop = importlib.import_module(f"{PACKAGE_NAME}.loop")


async def _empty_search(_query: Any, _limit: int) -> list[Any]:
    return []


async def _unused_read(url: str) -> Any:
    return loop.Page(url=url, title=url, text="")


async def _unused_extract(_prompt: str) -> Any:
    return loop.Extraction(facts=[], relevant=False)


async def _synthesize(_prompt: str) -> str:
    return "final report"


async def _never(*_args: Any, **_kwargs: Any) -> Any:
    await asyncio.Event().wait()


def _almost_expired_clock() -> Any:
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 59.99

    return clock


def _expired_after_start_clock() -> Any:
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 100.0

    return clock


@pytest.mark.asyncio
async def test_stops_on_confidence_round_two() -> None:
    calls = 0

    async def reason(_prompt: str) -> Any:
        nonlocal calls
        calls += 1
        return loop.ResearchStep(
            thought=f"round {calls}",
            updated_report=f"report {calls}",
            open_questions=[],
            confidence=0.4 if calls == 1 else 0.9,
            next_action="search",
            search_queries=[loop.SearchQuery(query="q")],
        )

    result = await loop.run_research_loop(
        question="q",
        max_rounds=10,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=_unused_extract,
        search_fn=_empty_search,
        read_fn=_unused_read,
        synthesize_fn=_synthesize,
    )

    assert result.stopped_reason == "confident"
    assert result.rounds_used == 2
    assert calls == 2


def test_default_depth_constants_track_upstream_react_shape() -> None:
    assert loop.MAX_ROUNDS_CAP == 100
    assert loop.WALL_CLOCK_SECONDS_CAP == 150 * 60
    assert loop.RESULTS_PER_QUERY == 10
    assert loop.RESULTS_PER_QUERY_CAP == 10
    assert loop.MAX_READS_PER_ROUND >= 10
    assert loop.REPORT_TOKEN_CAP >= 8_000


@pytest.mark.asyncio
async def test_stops_on_max_rounds_without_overrun() -> None:
    calls = 0

    async def reason(_prompt: str) -> Any:
        nonlocal calls
        calls += 1
        return loop.ResearchStep(
            thought="continue",
            updated_report=f"report {calls}",
            open_questions=[],
            confidence=0.1,
            next_action="search",
            search_queries=[loop.SearchQuery(query=f"q{calls}")],
        )

    async def search(_query: Any, _limit: int) -> list[Any]:
        return [{"url": f"https://example.com/{calls}", "title": f"t{calls}", "snippet": "s"}]

    result = await loop.run_research_loop(
        question="q",
        max_rounds=3,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=_unused_extract,
        search_fn=search,
        read_fn=_unused_read,
        synthesize_fn=_synthesize,
    )

    assert result.stopped_reason == "max_rounds"
    assert result.rounds_used == 3
    assert calls == 3


@pytest.mark.asyncio
async def test_stops_on_wall_clock_before_reasoner_call() -> None:
    reason_calls = 0

    async def reason(_prompt: str) -> Any:
        nonlocal reason_calls
        reason_calls += 1
        raise AssertionError("reasoner should not be called")

    result = await loop.run_research_loop(
        question="q",
        max_rounds=10,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=_unused_extract,
        search_fn=_empty_search,
        read_fn=_unused_read,
        synthesize_fn=_synthesize,
        clock=_expired_after_start_clock(),
    )

    assert result.stopped_reason == "wall_clock"
    assert result.rounds_used == 0
    assert reason_calls == 0


@pytest.mark.parametrize(
    ("blocked_operation", "warning_fragment"),
    [
        ("reasoner", "reasoner"),
        ("search", "search for round query"),
        ("read", "read for https://example.com/slow"),
        ("extractor", "extractor"),
        ("final_synthesis", "final synthesis"),
    ],
)
@pytest.mark.asyncio
async def test_wall_clock_deadline_wraps_awaited_operations(
    blocked_operation: str,
    warning_fragment: str,
) -> None:
    async def reason(_prompt: str) -> Any:
        if blocked_operation == "reasoner":
            return await _never()
        if blocked_operation == "search":
            return loop.ResearchStep(
                thought="search",
                updated_report="partial",
                open_questions=[],
                confidence=0.1,
                next_action="search",
                search_queries=[loop.SearchQuery(query="round query")],
            )
        if blocked_operation in {"read", "extractor"}:
            return loop.ResearchStep(
                thought="read",
                updated_report="partial",
                open_questions=[],
                confidence=0.1,
                next_action="read",
                read_urls=["https://example.com/slow"],
            )
        return loop.ResearchStep(
            thought="finish",
            updated_report="partial",
            open_questions=[],
            confidence=0.1,
            next_action="finish",
        )

    async def search(query: Any, _limit: int) -> list[Any]:
        if blocked_operation == "search" and query.query == "round query":
            return await _never()
        return []

    async def read(url: str) -> Any:
        if blocked_operation == "read":
            return await _never()
        return loop.Page(url=url, title=url, text="text")

    async def extract(_prompt: str) -> Any:
        if blocked_operation == "extractor":
            return await _never()
        return loop.Extraction(facts=[], relevant=False)

    async def synthesize(_prompt: str) -> str:
        if blocked_operation == "final_synthesis":
            return await _never()
        return "partial final"

    result = await loop.run_research_loop(
        question="q",
        max_rounds=1,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=extract,
        search_fn=search,
        read_fn=read,
        synthesize_fn=synthesize,
        clock=_almost_expired_clock(),
    )

    assert result.stopped_reason == "wall_clock"
    assert any(warning_fragment in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_no_progress_stops_after_duplicate_url_rounds() -> None:
    calls = 0

    async def reason(_prompt: str) -> Any:
        nonlocal calls
        calls += 1
        return loop.ResearchStep(
            thought="search duplicate",
            updated_report=f"report {calls}",
            open_questions=[],
            confidence=0.1,
            next_action="search",
            search_queries=[loop.SearchQuery(query="same")],
        )

    async def search(_query: Any, _limit: int) -> list[Any]:
        return [{"url": "https://example.com/a", "title": "A", "snippet": "same"}]

    result = await loop.run_research_loop(
        question="q",
        max_rounds=10,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=_unused_extract,
        search_fn=search,
        read_fn=_unused_read,
        synthesize_fn=_synthesize,
    )

    assert result.stopped_reason == "no_progress"
    assert result.rounds_used == 2


@pytest.mark.asyncio
async def test_compression_replaces_report_and_pending_evidence() -> None:
    prompts: list[str] = []
    calls = 0

    async def reason(prompt: str) -> Any:
        nonlocal calls
        prompts.append(prompt)
        calls += 1
        if calls == 3:
            return loop.ResearchStep(
                thought="finish",
                updated_report="R3",
                open_questions=[],
                confidence=0.7,
                next_action="finish",
            )
        return loop.ResearchStep(
            thought="search",
            updated_report=f"R{calls}",
            open_questions=[],
            confidence=0.1,
            next_action="search",
            search_queries=[loop.SearchQuery(query=f"q{calls}")],
        )

    search_calls = 0

    async def search(_query: Any, _limit: int) -> list[Any]:
        nonlocal search_calls
        search_calls += 1
        if search_calls == 1:
            return []
        return [
            {
                "url": f"https://example.com/{search_calls}",
                "title": f"T{search_calls}",
                "snippet": f"Snippet {search_calls}",
            },
        ]

    result = await loop.run_research_loop(
        question="q",
        max_rounds=5,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=_unused_extract,
        search_fn=search,
        read_fn=_unused_read,
        synthesize_fn=_synthesize,
    )

    assert result.stopped_reason == "model_finished"
    assert "R2" in prompts[2]
    assert "R1" not in prompts[2]
    assert "Snippet 3" in prompts[2]
    assert "Snippet 2" not in prompts[2]


@pytest.mark.asyncio
async def test_citation_ids_are_stable_and_monotonic() -> None:
    calls = 0
    urls = {
        1: "https://example.com/a",
        2: "https://example.com/b",
        3: "https://example.com/a",
    }

    async def reason(_prompt: str) -> Any:
        nonlocal calls
        calls += 1
        if calls == 4:
            return loop.ResearchStep(
                thought="finish",
                updated_report="Use A [1] and B [2]",
                open_questions=[],
                confidence=0.7,
                next_action="finish",
            )
        return loop.ResearchStep(
            thought="read",
            updated_report=f"report {calls}",
            open_questions=[],
            confidence=0.1,
            next_action="read",
            read_urls=[urls[calls]],
        )

    async def read(url: str) -> Any:
        return loop.Page(url=url, title=url.rsplit("/", 1)[-1].upper(), text=f"text {url}")

    async def extract(prompt: str) -> Any:
        fact = "fact A" if "https://example.com/a" in prompt else "fact B"
        return loop.Extraction(facts=[fact], relevant=True)

    async def synthesize(_prompt: str) -> str:
        return "Use A [1] and B [2]"

    result = await loop.run_research_loop(
        question="q",
        max_rounds=6,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=extract,
        search_fn=_empty_search,
        read_fn=read,
        synthesize_fn=synthesize,
    )

    assert [source["id"] for source in result.sources] == [1, 2]
    assert result.sources[0]["url"] == "https://example.com/a"
    assert result.sources[1]["url"] == "https://example.com/b"
    assert result.sources_used == 2


@pytest.mark.asyncio
async def test_reasoner_prompt_exposes_candidate_urls() -> None:
    prompts: list[str] = []

    async def reason(prompt: str) -> Any:
        prompts.append(prompt)
        return loop.ResearchStep(
            thought="finish",
            updated_report="r",
            open_questions=[],
            confidence=0.2,
            next_action="finish",
        )

    async def search(_query: Any, _limit: int) -> list[Any]:
        return [{"url": "https://example.com/source", "title": "Source", "snippet": "snippet"}]

    await loop.run_research_loop(
        question="q",
        max_rounds=1,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=_unused_extract,
        search_fn=search,
        read_fn=_unused_read,
        synthesize_fn=_synthesize,
    )

    assert "https://example.com/source" in prompts[0]
    assert "Candidate URL: Source - https://example.com/source" in prompts[0]


@pytest.mark.asyncio
async def test_search_hits_are_candidates_not_citeable_sources_until_read() -> None:
    async def search(_query: Any, _limit: int) -> list[Any]:
        return [{"url": "https://example.com/a", "title": "A", "snippet": "seed"}]

    async def reason(_prompt: str) -> Any:
        return loop.ResearchStep(
            thought="finish from candidate only",
            updated_report="candidate-only claim [1]",
            open_questions=[],
            confidence=0.2,
            next_action="finish",
        )

    result = await loop.run_research_loop(
        question="q",
        max_rounds=1,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=_unused_extract,
        search_fn=search,
        read_fn=_unused_read,
        synthesize_fn=lambda _prompt: _await("candidate-only claim [1]"),
    )

    assert result.sources == []
    assert result.sources_considered == 1
    assert "[1]" not in result.report
    assert result.report.endswith("## Sources\n(none)")


@pytest.mark.asyncio
async def test_productive_reads_of_registered_sources_are_progress_and_synthesized() -> None:
    reason_calls = 0
    synthesize_prompts: list[str] = []

    async def search(_query: Any, _limit: int) -> list[Any]:
        return [
            {"url": "https://example.com/a", "title": "A", "snippet": "seed A"},
            {"url": "https://example.com/b", "title": "B", "snippet": "seed B"},
        ]

    async def reason(_prompt: str) -> Any:
        nonlocal reason_calls
        reason_calls += 1
        if reason_calls == 1:
            return loop.ResearchStep(
                thought="read a",
                updated_report="R1",
                open_questions=[],
                confidence=0.1,
                next_action="read",
                read_urls=["https://example.com/a"],
            )
        if reason_calls == 2:
            return loop.ResearchStep(
                thought="read b",
                updated_report="R2 includes fact A [1]",
                open_questions=[],
                confidence=0.1,
                next_action="read",
                read_urls=["https://example.com/b"],
            )
        return loop.ResearchStep(
            thought="finish",
            updated_report="R3 includes fact A [1] and fact B [2]",
            open_questions=[],
            confidence=0.4,
            next_action="finish",
        )

    async def read(url: str) -> Any:
        return loop.Page(url=url, title=url.rsplit("/", 1)[-1].upper(), text=f"text for {url}")

    async def extract(prompt: str) -> Any:
        if "https://example.com/a" in prompt:
            return loop.Extraction(facts=["fact A"], relevant=True)
        return loop.Extraction(facts=["fact B"], relevant=True)

    async def synthesize(prompt: str) -> str:
        synthesize_prompts.append(prompt)
        return "final [1] [2]\n\n## Sources\n[1] A - https://example.com/a\n[2] B - https://example.com/b"

    result = await loop.run_research_loop(
        question="q",
        max_rounds=3,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=extract,
        search_fn=search,
        read_fn=read,
        synthesize_fn=synthesize,
    )

    assert result.stopped_reason == "model_finished"
    assert result.rounds_used == 3
    assert "fact B" in synthesize_prompts[0]


@pytest.mark.asyncio
async def test_final_action_evidence_reaches_synthesis_on_max_rounds() -> None:
    synthesize_prompts: list[str] = []

    async def reason(_prompt: str) -> Any:
        return loop.ResearchStep(
            thought="search",
            updated_report="R",
            open_questions=[],
            confidence=0.1,
            next_action="search",
            search_queries=[loop.SearchQuery(query="final")],
        )

    async def search(query: Any, _limit: int) -> list[Any]:
        if query.query == "q":
            return []
        return [{"url": "https://example.com/final", "title": "Final", "snippet": "final snippet"}]

    async def synthesize(prompt: str) -> str:
        synthesize_prompts.append(prompt)
        return "final [1]"

    result = await loop.run_research_loop(
        question="q",
        max_rounds=1,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=_unused_extract,
        search_fn=search,
        read_fn=_unused_read,
        synthesize_fn=synthesize,
    )

    assert result.stopped_reason == "max_rounds"
    assert "final snippet" in synthesize_prompts[0]


@pytest.mark.asyncio
async def test_search_failures_warn_and_continue_to_synthesis() -> None:
    search_calls = 0

    async def search(_query: Any, _limit: int) -> list[Any]:
        nonlocal search_calls
        search_calls += 1
        raise RuntimeError("serper down")

    async def reason(_prompt: str) -> Any:
        return loop.ResearchStep(
            thought="finish",
            updated_report="partial",
            open_questions=[],
            confidence=0.4,
            next_action="finish",
        )

    result = await loop.run_research_loop(
        question="q",
        max_rounds=1,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=_unused_extract,
        search_fn=search,
        read_fn=_unused_read,
        synthesize_fn=_synthesize,
    )

    assert search_calls == 1
    assert result.stopped_reason == "model_finished"
    assert result.warnings == ["seed search failed: serper down"]


@pytest.mark.asyncio
async def test_round_search_failures_warn_and_continue_to_synthesis() -> None:
    reason_calls = 0

    async def search(query: Any, _limit: int) -> list[Any]:
        if query.query == "q":
            return []
        raise RuntimeError("serper down")

    async def reason(_prompt: str) -> Any:
        nonlocal reason_calls
        reason_calls += 1
        if reason_calls == 1:
            return loop.ResearchStep(
                thought="search",
                updated_report="partial",
                open_questions=[],
                confidence=0.1,
                next_action="search",
                search_queries=[loop.SearchQuery(query="round query")],
            )
        return loop.ResearchStep(
            thought="finish",
            updated_report="partial after failed search",
            open_questions=[],
            confidence=0.4,
            next_action="finish",
        )

    result = await loop.run_research_loop(
        question="q",
        max_rounds=2,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=_unused_extract,
        search_fn=search,
        read_fn=_unused_read,
        synthesize_fn=_synthesize,
    )

    assert result.stopped_reason == "model_finished"
    assert result.warnings == ["search failed for round query: serper down"]


@pytest.mark.asyncio
async def test_final_synthesis_without_valid_citations_retries_with_registry() -> None:
    async def search(_query: Any, _limit: int) -> list[Any]:
        return []

    async def reason(_prompt: str) -> Any:
        return loop.ResearchStep(
            thought="read",
            updated_report="need page",
            open_questions=[],
            confidence=0.1,
            next_action="read",
            read_urls=["https://example.com/a"],
        )

    synthesize_calls = 0

    async def synthesize(_prompt: str) -> str:
        nonlocal synthesize_calls
        synthesize_calls += 1
        if synthesize_calls == 1:
            return "body without citations"
        return "body with citation [1]"

    result = await loop.run_research_loop(
        question="q",
        max_rounds=1,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=lambda _prompt: _await(loop.Extraction(facts=["verified fact"], relevant=True)),
        search_fn=search,
        read_fn=lambda url: _await(loop.Page(url=url, title="A", text="verified text")),
        synthesize_fn=synthesize,
    )

    assert synthesize_calls == 2
    assert result.sources_used == 1
    assert result.report.endswith("## Sources\n[1] A - https://example.com/a")


@pytest.mark.asyncio
async def test_final_sources_section_is_rebuilt_from_registry() -> None:
    async def search(_query: Any, _limit: int) -> list[Any]:
        return []

    async def reason(_prompt: str) -> Any:
        return loop.ResearchStep(
            thought="read",
            updated_report="partial [1]",
            open_questions=[],
            confidence=0.2,
            next_action="read",
            read_urls=["https://example.com/a"],
        )

    async def synthesize(_prompt: str) -> str:
        return (
            "Body cites valid [1], bogus , and mixed [1, 999, 1000].\n\n"
            "## Sources:\n"
            "[1] Fabricated - https://evil.example\n"
            "[999] Ghost - https://ghost.example"
        )

    result = await loop.run_research_loop(
        question="q",
        max_rounds=1,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=lambda _prompt: _await(loop.Extraction(facts=["fact A"], relevant=True)),
        search_fn=search,
        read_fn=lambda url: _await(loop.Page(url=url, title="A", text="text")),
        synthesize_fn=synthesize,
    )

    assert "[999]" not in result.report
    assert "[1000]" not in result.report
    assert "[1, 999, 1000]" not in result.report
    assert "https://evil.example" not in result.report
    assert result.report.endswith("## Sources\n[1] A - https://example.com/a")
    assert result.sources_used == 1


@pytest.mark.parametrize("heading", ["# References", "### Bibliography", "###### Sources:"])
@pytest.mark.asyncio
async def test_final_source_like_heading_variants_are_stripped(heading: str) -> None:
    async def search(_query: Any, _limit: int) -> list[Any]:
        return []

    async def reason(_prompt: str) -> Any:
        return loop.ResearchStep(
            thought="read",
            updated_report="partial [1]",
            open_questions=[],
            confidence=0.2,
            next_action="read",
            read_urls=["https://example.com/a"],
        )

    async def synthesize(_prompt: str) -> str:
        return f"Body [1]\n\n{heading}\n[1] Fake - https://fake.example"

    result = await loop.run_research_loop(
        question="q",
        max_rounds=1,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=lambda _prompt: _await(loop.Extraction(facts=["fact A"], relevant=True)),
        search_fn=search,
        read_fn=lambda url: _await(loop.Page(url=url, title="A", text="text")),
        synthesize_fn=synthesize,
    )

    assert "https://fake.example" not in result.report
    assert result.report.endswith("## Sources\n[1] A - https://example.com/a")


@pytest.mark.asyncio
async def test_final_synthesis_failure_returns_fallback_report_with_warning() -> None:
    async def search(_query: Any, _limit: int) -> list[Any]:
        return []

    reason_calls = 0

    async def reason(_prompt: str) -> Any:
        nonlocal reason_calls
        reason_calls += 1
        if reason_calls == 2:
            return loop.ResearchStep(
                thought="finish",
                updated_report="compressed [1]",
                open_questions=[],
                confidence=0.2,
                next_action="finish",
            )
        return loop.ResearchStep(
            thought="read",
            updated_report="need source",
            open_questions=[],
            confidence=0.2,
            next_action="read",
            read_urls=["https://example.com/a"],
        )

    async def synthesize(_prompt: str) -> str:
        raise RuntimeError("provider down")

    result = await loop.run_research_loop(
        question="q",
        max_rounds=2,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=lambda _prompt: _await(loop.Extraction(facts=["fact A"], relevant=True)),
        search_fn=search,
        read_fn=lambda url: _await(loop.Page(url=url, title="A", text="text")),
        synthesize_fn=synthesize,
    )

    assert result.stopped_reason == "model_finished"
    assert result.report.endswith("## Sources\n[1] A - https://example.com/a")
    assert result.warnings == ["final synthesis failed: provider down"]


@pytest.mark.asyncio
async def test_search_and_read_routing_respect_round_clamps() -> None:
    search_queries: list[str] = []

    async def reason_search(_prompt: str) -> Any:
        return loop.ResearchStep(
            thought="many searches",
            updated_report="r",
            open_questions=[],
            confidence=0.1,
            next_action="search",
            search_queries=[loop.SearchQuery(query=f"q{i}") for i in range(10)],
        )

    async def search(query: Any, _limit: int) -> list[Any]:
        if query.query != "q":
            search_queries.append(query.query)
        return [{"url": f"https://example.com/{query.query}", "title": query.query, "snippet": ""}]

    await loop.run_research_loop(
        question="q",
        max_rounds=1,
        wall_clock_seconds=60,
        reason_fn=reason_search,
        extract_fn=_unused_extract,
        search_fn=search,
        read_fn=_unused_read,
        synthesize_fn=_synthesize,
    )
    assert search_queries == [f"q{i}" for i in range(loop.MAX_QUERIES_PER_ROUND)]

    read_urls: list[str] = []

    async def reason_read(_prompt: str) -> Any:
        return loop.ResearchStep(
            thought="many reads",
            updated_report="r",
            open_questions=[],
            confidence=0.1,
            next_action="read",
            read_urls=[f"https://example.com/{i}" for i in range(10)],
        )

    async def read(url: str) -> Any:
        read_urls.append(url)
        return loop.Page(url=url, title=url, text="text")

    await loop.run_research_loop(
        question="q",
        max_rounds=1,
        wall_clock_seconds=60,
        reason_fn=reason_read,
        extract_fn=_unused_extract,
        search_fn=_empty_search,
        read_fn=read,
        synthesize_fn=_synthesize,
    )
    assert read_urls == [f"https://example.com/{i}" for i in range(loop.MAX_READS_PER_ROUND)]


@pytest.mark.asyncio
async def test_extractor_relevance_gates_facts_and_source_registration() -> None:
    async def reason(_prompt: str) -> Any:
        return loop.ResearchStep(
            thought="read",
            updated_report="r",
            open_questions=[],
            confidence=0.1,
            next_action="read",
            read_urls=["https://example.com/page"],
        )

    result = await loop.run_research_loop(
        question="q",
        max_rounds=1,
        wall_clock_seconds=60,
        reason_fn=reason,
        extract_fn=_unused_extract,
        search_fn=_empty_search,
        read_fn=lambda url: _unused_read(url),
        synthesize_fn=_synthesize,
    )

    assert result.sources == []
    assert result.sources_considered == 1


@pytest.mark.asyncio
async def test_structured_output_string_parse_and_malformed_retry_fallback() -> None:
    raw_json = json.dumps(
        {
            "thought": "parsed",
            "updated_report": "parsed report",
            "open_questions": [],
            "confidence": 0.9,
            "next_action": "finish",
            "search_queries": [],
            "read_urls": [],
        },
    )

    result = await loop.run_research_loop(
        question="q",
        max_rounds=2,
        wall_clock_seconds=60,
        reason_fn=lambda _prompt: _await(raw_json),
        extract_fn=_unused_extract,
        search_fn=_empty_search,
        read_fn=_unused_read,
        synthesize_fn=_synthesize,
    )

    assert result.stopped_reason == "confident"
    assert result.rounds_used == 1
    assert result.warnings == []

    calls = 0

    async def malformed(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "{not json"

    fallback_result = await loop.run_research_loop(
        question="q",
        max_rounds=2,
        wall_clock_seconds=60,
        reason_fn=malformed,
        extract_fn=_unused_extract,
        search_fn=_empty_search,
        read_fn=_unused_read,
        synthesize_fn=_synthesize,
    )

    assert calls == 2
    assert fallback_result.stopped_reason == "model_finished"
    assert fallback_result.rounds_used == 1
    assert "reasoner structured output failed after retry" in fallback_result.warnings[0]


@pytest.mark.asyncio
async def test_structured_retry_adds_json_only_nudge() -> None:
    prompts: list[str] = []

    async def malformed(prompt: str) -> str:
        prompts.append(prompt)
        return "{not json"

    await loop.run_research_loop(
        question="q",
        max_rounds=1,
        wall_clock_seconds=60,
        reason_fn=malformed,
        extract_fn=_unused_extract,
        search_fn=_empty_search,
        read_fn=_unused_read,
        synthesize_fn=_synthesize,
    )

    assert len(prompts) == 2
    assert prompts[0] != prompts[1]
    assert "Return ONLY the JSON object" in prompts[1]


async def _await(value: Any) -> Any:
    return value
