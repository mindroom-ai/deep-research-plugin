# Deep Research

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-plugins-blue)](https://docs.mindroom.chat/plugins/)

<img src="https://media.githubusercontent.com/media/mindroom-ai/mindroom/refs/heads/main/frontend/public/logo.png" alt="MindRoom Logo" align="right" width="120" />

A long-horizon web-research tool for [MindRoom](https://github.com/mindroom-ai/mindroom) agents, powered by the agent's own configured model.

Ask one hard question and get back a cited report. `deep_research` runs a bounded, multi-round research loop — search, read, extract, decide, repeat — that compresses what it has learned into a rolling report between rounds so it can dig deep without blowing the context window. It ports the IterResearch loop pattern (from Alibaba's Tongyi DeepResearch) onto MindRoom's existing model and tools instead of wrapping a dedicated research model, so it runs on whatever model the calling agent already uses (e.g. Vertex Claude) with no GPU and no extra services.

## Features

- Single `deep_research(question)` tool that returns a cited Markdown report as a JSON envelope
- Runs the loop on the caller's active MindRoom model — provider-agnostic, no model bundled
- Reuses MindRoom's existing tools: Serper search and the native website reader
- Rolling summarize-and-replace report keeps long runs inside the context budget
- Append-only per-source fact bank, so extracted facts survive compression and feed final synthesis
- Stable `[n]` citations backed by a verified source registry that is the single source of truth
- Hard wall-clock deadline bounds every step, plus per-operation timeouts and transparent retries for search/read
- Searches and page reads run concurrently within a round (bounded concurrency)
- Duplicate queries and already-fetched URLs are skipped across rounds, and the reasoner is told what was already tried
- Confidence-based stopping is evidence-gated (requires at least one registered source), plus no-progress stopping
- A transiently failing reasoner call skips the round instead of ending the run
- Streams per-round progress into the thread, or runs quietly on request
- Pure plugin: no core changes and no plugin-local dependencies

## How It Works

1. An agent calls `deep_research(question)` in a thread.
2. The plugin resolves the caller's active model and builds an ephemeral, tool-less reasoning agent.
3. Each round: the reasoner plans searches and/or page reads (both may run in the same round), which execute concurrently with retries and per-operation timeouts; extracted facts are folded into a rolling report with stable citations and banked per source.
4. Between rounds the report is compressed (summarize-and-replace) so context stays bounded; the fact bank preserves evidence the compression drops.
5. The loop stops on high evidence-backed confidence, on no further progress, or when the round/wall-clock budget runs out. Repeated queries and re-reads are skipped so a looping model cannot burn budget.
6. A final synthesis pass sees the compressed report, the fact bank, and the source registry; the `## Sources` section is rebuilt from the registry using only citations that actually appear in the body.

## Agent Tools

| Tool | Purpose |
|------|---------|
| `deep_research(question, max_rounds=100, wall_clock_seconds=9000, model=None, verbosity="progress", max_queries_per_round=5, results_per_query=10, max_reads_per_round=10, page_char_limit=150000, report_token_cap=8000)` | Run a bounded, cited web-research loop for one question and return a JSON report envelope |

The returned envelope includes `status`, `report` (Markdown with `[n]` citations), `sources`, `confidence`, `rounds_used`, `stopped_reason`, `elapsed_seconds`, any `warnings`, and `stats` (counts of searches, reads, extractions, retries skipped as duplicates, and failures).

Parameters:

- `question` — the research question (required, non-empty).
- `max_rounds` — soft round budget (default `100`, mirroring the original repository's default LLM-call budget as this loop's round cap).
- `wall_clock_seconds` — hard time budget (default `9000`, matching the original repository's 150-minute timeout).
- `model` — override the model name; defaults to the caller's active model.
- `verbosity` — `"progress"` streams per-round updates into the thread; `"silent"` returns only the final report.
- `max_queries_per_round` — maximum planned search queries per search round (default `5`, capped at `10`).
- `results_per_query` — search results fetched per query (default `10`, capped at `30`; Serper accepts larger `num` values, the cap keeps candidate lists within prompt budget).
- `max_reads_per_round` — maximum URLs read in one read round (default `10`, capped at `20`).
- `page_char_limit` — maximum page text passed to extraction (default `150000` chars, capped at `600000`).
- `report_token_cap` — approximate rolling report token budget (default `8000`, capped at `64000`).

## Configuration

`deep_research` uses whatever model the calling agent is configured with, and reuses MindRoom's Serper search and native website reader. Make sure the built-in Serper tool has an API key configured before enabling this plugin.

## Setup

1. Copy this plugin to `~/.mindroom/plugins/deep-research`.
2. Add the plugin to `config.yaml`:
   ```yaml
   plugins:
     - path: plugins/deep-research
   ```
3. Add `deep_research` to the agent's tools list.
4. Restart MindRoom.
