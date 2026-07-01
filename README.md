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
- Stable `[n]` citations backed by a source registry that is the single source of truth
- Hard wall-clock deadline bounds every step, including progress updates and final synthesis
- Confidence-based and no-progress stopping so it ends as soon as the answer is solid
- Streams per-round progress into the thread, or runs quietly on request
- Pure plugin: no core changes and no plugin-local dependencies

## How It Works

1. An agent calls `deep_research(question)` in a thread.
2. The plugin resolves the caller's active model and builds an ephemeral, tool-less reasoning agent.
3. Each round: search the web, read the most promising pages, extract facts, and fold them into a rolling report with stable citations.
4. Between rounds the report is compressed (summarize-and-replace) so context stays bounded.
5. The loop stops on high confidence, on no further progress, or when the round/wall-clock budget runs out.
6. A final synthesis pass produces the report, and the `## Sources` section is rebuilt from the registry using only citations that actually appear in the body.

## Agent Tools

| Tool | Purpose |
|------|---------|
| `deep_research(question, max_rounds=10, wall_clock_seconds=300, model=None, verbosity="progress")` | Run a bounded, cited web-research loop for one question and return a JSON report envelope |

The returned envelope includes `status`, `report` (Markdown with `[n]` citations), `sources`, `confidence`, `rounds_used`, `stopped_reason`, `elapsed_seconds`, and any `warnings`.

Parameters:

- `question` — the research question (required, non-empty).
- `max_rounds` — soft round budget (default `10`, capped at `40`).
- `wall_clock_seconds` — hard time budget (default `300`, min `60`, capped at `900`).
- `model` — override the model name; defaults to the caller's active model.
- `verbosity` — `"progress"` streams per-round updates into the thread; `"silent"` returns only the final report.

## Configuration

`deep_research` uses whatever model the calling agent is configured with, and reuses MindRoom's Serper search and native website reader. Make sure the runtime already has a Serper API key configured for search.

## Setup

1. Copy this plugin to `~/.mindroom/plugins/deep-research`.
2. Add the plugin to `config.yaml`:
   ```yaml
   plugins:
     - path: plugins/deep-research
   ```
3. Add `deep_research` to the agent's tools list.
4. Restart MindRoom.