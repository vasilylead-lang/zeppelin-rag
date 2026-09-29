# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- `uv sync` — install (Python 3.12+, uv-managed `.venv`)
- `uv run pytest` — offline test suite (no API key, no model download)
- `uv run pytest tests/test_graph.py::test_irrelevant_chunks_trigger_rewrite_then_generate` — single test
- `uv run ruff check .` and `uv run ruff format .` — lint and format (CI runs both plus pytest)
- `uv run zeppelin-ask "вопрос" --trace` — one question through the graph (needs `ANTHROPIC_API_KEY` in `.env`)
- `uv run zeppelin-eval [--limit N]` — RAGAS evaluation, writes `reports/summary.md` and `reports/ragas_scores.csv`. Spends API credits.

## Architecture

Corrective RAG over a Russian-language Markdown knowledge base about zeppelins.

- `knowledge.py` splits every `data/knowledge/*.md` file into one `Chunk` per `## ` section. The `# ` title is prepended to the chunk text. New content only needs well-sized `##` sections.
- `retriever.py` `HybridIndex` mixes min-max normalized dense cosine scores (fastembed multilingual MiniLM, local ONNX) with BM25 over 5-letter word prefixes (a crude Russian stemmer). `lexical_weight` controls the mix. The lexical part exists because the small embedder misses rare terms. `search_many` runs several queries and fuses their rankings with reciprocal rank fusion (`rrf_k=60`).
- `graph.py` is a LangGraph `StateGraph`: `decompose -> retrieve -> grade -> (generate | rewrite -> retrieve)`. `decompose` asks the fast model for up to `max_subqueries` keyword sub-queries and keeps the question itself as the first query. `retrieve` fuses results into at most `max_chunks`. With `multi_query=False` (`ZEPPELIN_MULTI_QUERY=0`), `decompose` makes no LLM call and search uses only the question. `rewrite` replaces the query list with one rewritten query. The number of rewrites is capped by `max_rewrites`. When the cap is hit with nothing graded relevant, `generate` falls back to the raw retrieved chunks. `steps` uses an `operator.add` reducer as a trace.
- `llm.py` holds all Claude calls through the official `anthropic` SDK, not LangChain chat models. `decompose` and `grade` use `messages.parse` structured output on the fast model. The `DECOMPOSE_PROMPT` demands concrete names and terms: vague sub-queries ("оснащение", "свойства") pulled generic sections and hurt recall. `answer` uses `client.beta.messages.create` with `fallbacks="default"` (beta `server-side-fallback-2026-07-01`) and checks `stop_reason == "refusal"`. The graph depends only on the `RagLLM` protocol, which is how tests inject `FakeLLM`.
- `evaluate.py` runs the graph over `data/eval/testset.jsonl`, then scores the answers with RAGAS 0.4 **collections** metrics (`ragas.metrics.collections`, async `ascore`), not the legacy `ragas.evaluate()`. The judge comes from `llm_factory(provider="anthropic")`.

## Gotchas

- `make_judge` overwrites `judge.model_args`. RAGAS defaults send `temperature` and `top_p`, which current Claude models reject. Instructor forces `tool_choice`, which requires `thinking: disabled`. Keep that override when changing the judge model.
- The repo sits in iCloud-synced `~/Documents`. iCloud sets the macOS `hidden` flag on everything in a real `.venv` directory, and Python 3.12.12+ skips hidden `.pth` files, so the editable install breaks with `ModuleNotFoundError: zeppelin_rag`. Here `.venv` is a symlink to `.venv.nosync/`, which iCloud ignores. Never recreate `.venv` as a plain directory. Diagnose with `ls -lO .venv/lib/python3.12/site-packages/*.pth`.
- `langchain-community` is pinned `<0.4.2`: 0.4.2 removed `chat_models.vertexai`, which `ragas` 0.4.3 imports at load time.
- Model IDs and settings live in `config.py` `Settings` and can be overridden with `ZEPPELIN_*` env vars (see `.env.example`).
- Experiments go to `reports/experiments/<name>/` via `zeppelin-eval --out`; `reports/` itself holds the run for the default configuration. The README compares them.
- Knowledge base and test set are in Russian. When editing facts, keep `data/eval/testset.jsonl` reference answers consistent with `data/knowledge/`.
