# ruff: noqa: INP001
"""Prompt templates for the deep-research plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def reasoner_prompt(
    *,
    question: str,
    report: str,
    pending_evidence: Sequence[str],
    sources: Sequence[str],
    budget_left: str,
) -> str:
    """Build the structured reasoner prompt."""
    evidence_text = "\n".join(f"- {item}" for item in pending_evidence) or "- (no new evidence)"
    sources_text = "\n".join(f"- {item}" for item in sources) or "- (no sources yet)"
    return f"""You are the planning and compression step in an iterative research loop.

Question:
{question}

Current compressed report, which your updated_report must replace:
{report or "(empty)"}

New pending evidence:
{evidence_text}

Known source registry:
{sources_text}

Budget left:
{budget_left}

Return only the requested structured object. Update the report by compressing
the current report plus any useful pending evidence into a standalone research
workspace. Keep it concise and cite claims with existing [n] source IDs when
available. Lines beginning "Candidate URL:" are discovery leads only; do not
cite them unless they later appear in the source registry with a [n] ID. Decide exactly one next_action:
- search: when more search results are needed; provide up to 3 focused queries.
- read: when specific URLs should be fetched; provide up to 3 URLs.
- finish: when the report can answer the question.
"""


def extractor_prompt(*, question: str, url: str, page_text: str) -> str:
    """Build the structured page extractor prompt."""
    return f"""Extract question-relevant facts from one fetched web page.

Question:
{question}

URL:
{url}

Page text:
{page_text}

Return only the requested structured object. If the page is not useful for the
question, set relevant=false and facts=[].
"""


def synthesize_prompt(*, question: str, report: str, sources: Sequence[dict[str, object]]) -> str:
    """Build the final cited synthesis prompt."""
    source_lines = "\n".join(
        f"[{source['id']}] {source.get('title') or source.get('url')} - {source.get('url')}"
        for source in sources
    )
    return f"""Write the final answer as a concise Markdown research report.

Question:
{question}

Compressed research workspace:
{report or "(empty)"}

Source registry:
{source_lines or "(no sources)"}

Requirements:
- Use inline [n] citations for source-backed claims.
- Include a final section exactly titled "## Sources".
- In the sources section, list every cited source as "[n] title - URL".
- Do not invent citations that are not in the source registry.
"""
