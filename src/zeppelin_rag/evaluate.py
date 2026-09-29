"""`zeppelin-eval` — run the test set through the graph and score it with RAGAS.

Metrics (all judged by Claude, see Settings.judge_model):
  faithfulness                      — answer claims are supported by the retrieved context
  answer_relevancy                  — answer addresses the question (uses local embeddings)
  context_recall                    — retrieved context covers the reference answer
  context_precision_with_reference  — relevant chunks are ranked first
  factual_correctness               — answer claims match the reference answer (F1)
"""

import argparse
import asyncio
import csv
import json
import statistics
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from zeppelin_rag.config import REPORTS_DIR, TESTSET_PATH, Settings

METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision_with_reference",
    "factual_correctness",
]


def load_testset(path: Path = TESTSET_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_judge(settings: Settings):
    from anthropic import AsyncAnthropic
    from ragas.llms import llm_factory

    from zeppelin_rag.llm import client_options

    client = AsyncAnthropic(**client_options())
    judge = llm_factory(settings.judge_model, provider="anthropic", client=client)
    # RAGAS defaults to temperature=0.01 and top_p=0.1; current Claude models reject
    # sampling parameters. Instructor forces a tool call, which requires thinking off.
    judge.model_args = {"max_tokens": 8192, "thinking": {"type": "disabled"}}
    return judge


def make_embeddings(embedder):
    from ragas.embeddings.base import BaseRagasEmbedding

    class LocalEmbeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs) -> list[float]:
            return embedder.embed([text])[0].tolist()

        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            return self.embed_text(text)

    return LocalEmbeddings()


def make_metrics(judge, embeddings) -> dict:
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )

    return {
        "faithfulness": Faithfulness(llm=judge),
        "answer_relevancy": AnswerRelevancy(llm=judge, embeddings=embeddings),
        "context_recall": ContextRecall(llm=judge),
        "context_precision_with_reference": ContextPrecisionWithReference(llm=judge),
        "factual_correctness": FactualCorrectness(llm=judge),
    }


async def score_sample(metrics: dict, sample: dict) -> dict:
    question, answer = sample["question"], sample["answer"]
    contexts, reference = sample["contexts"], sample["reference"]
    calls = {
        "faithfulness": metrics["faithfulness"].ascore(
            user_input=question, response=answer, retrieved_contexts=contexts
        ),
        "answer_relevancy": metrics["answer_relevancy"].ascore(
            user_input=question, response=answer
        ),
        "context_recall": metrics["context_recall"].ascore(
            user_input=question, retrieved_contexts=contexts, reference=reference
        ),
        "context_precision_with_reference": metrics["context_precision_with_reference"].ascore(
            user_input=question, reference=reference, retrieved_contexts=contexts
        ),
        "factual_correctness": metrics["factual_correctness"].ascore(
            response=answer, reference=reference
        ),
    }
    results = await asyncio.gather(*calls.values(), return_exceptions=True)
    scores = {}
    for name, result in zip(calls, results, strict=True):
        if isinstance(result, Exception):
            print(f"  ! {name} failed on {question[:40]!r}: {result}")
            scores[name] = None
        else:
            scores[name] = float(result.value)
    return scores


async def score_all(metrics: dict, samples: list[dict], concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run(sample: dict) -> dict:
        async with semaphore:
            return await score_sample(metrics, sample)

    return await asyncio.gather(*(run(s) for s in samples))


def summarize(rows: list[dict]) -> dict[str, float | None]:
    summary = {}
    for name in METRIC_NAMES:
        values = [r[name] for r in rows if r.get(name) is not None]
        summary[name] = statistics.mean(values) if values else None
    return summary


def write_reports(rows: list[dict], summary: dict, settings: Settings, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ragas_scores.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", *METRIC_NAMES, "answer"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    lines = [
        f"# RAGAS report — {date.today().isoformat()}",
        "",
        f"- samples: {len(rows)}",
        f"- answer model: `{settings.answer_model}` (effort `{settings.answer_effort}`)",
        f"- grader / rewriter: `{settings.fast_model}`",
        f"- judge: `{settings.judge_model}`",
        f"- embeddings: `{settings.embedding_model}`, top_k={settings.top_k}, "
        f"lexical_weight={settings.lexical_weight}",
        f"- multi-query: {'on' if settings.multi_query else 'off'}"
        + (
            f", max_subqueries={settings.max_subqueries}, max_chunks={settings.max_chunks}"
            if settings.multi_query
            else ""
        ),
        "",
        "| metric | mean |",
        "|---|---|",
    ]
    for name, value in summary.items():
        lines.append(f"| {name} | {'n/a' if value is None else f'{value:.3f}'} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG graph with RAGAS")
    parser.add_argument("--limit", type=int, help="Only the first N questions")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    load_dotenv()
    from zeppelin_rag.graph import build_default_graph

    settings = Settings.from_env()
    testset = load_testset()[: args.limit]
    graph, index = build_default_graph(settings)

    samples = []
    for i, item in enumerate(testset, start=1):
        print(f"[{i}/{len(testset)}] {item['question']}")
        state = graph.invoke({"question": item["question"], "steps": []})
        samples.append(
            {
                "question": item["question"],
                "reference": item["reference"],
                "answer": state["answer"],
                "contexts": [chunk.text for chunk in state["documents"]],
            }
        )

    print("Scoring with RAGAS...")
    metrics = make_metrics(make_judge(settings), make_embeddings(index.embedder))
    scores = asyncio.run(score_all(metrics, samples, args.concurrency))
    rows = [{**sample, **score} for sample, score in zip(samples, scores, strict=True)]

    summary = summarize(rows)
    write_reports(rows, summary, settings, args.out)
    for name, value in summary.items():
        print(f"{name:34s} {'n/a' if value is None else f'{value:.3f}'}")
    print(f"Reports written to {args.out}")


if __name__ == "__main__":
    main()
