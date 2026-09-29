"""Paths and runtime settings, overridable through environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = ROOT / "data" / "knowledge"
TESTSET_PATH = ROOT / "data" / "eval" / "testset.jsonl"
REPORTS_DIR = ROOT / "reports"


@dataclass(frozen=True)
class Settings:
    # Writes the final answer.
    answer_model: str = "claude-opus-5"
    # Cheap worker for document grading and query rewriting.
    fast_model: str = "claude-haiku-4-5"
    # LLM judge for RAGAS metrics.
    judge_model: str = "claude-sonnet-5"
    answer_effort: str = "medium"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    top_k: int = 4
    lexical_weight: float = 0.5
    max_rewrites: int = 2
    # Split the question into sub-queries and fuse their results (reciprocal rank fusion).
    multi_query: bool = True
    max_subqueries: int = 3
    # Upper bound on fused chunks passed to the grader in multi-query mode.
    max_chunks: int = 8

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            answer_model=os.getenv("ZEPPELIN_ANSWER_MODEL", defaults.answer_model),
            fast_model=os.getenv("ZEPPELIN_FAST_MODEL", defaults.fast_model),
            judge_model=os.getenv("ZEPPELIN_JUDGE_MODEL", defaults.judge_model),
            answer_effort=os.getenv("ZEPPELIN_ANSWER_EFFORT", defaults.answer_effort),
            embedding_model=os.getenv("ZEPPELIN_EMBEDDING_MODEL", defaults.embedding_model),
            top_k=int(os.getenv("ZEPPELIN_TOP_K", defaults.top_k)),
            lexical_weight=float(os.getenv("ZEPPELIN_LEXICAL_WEIGHT", defaults.lexical_weight)),
            max_rewrites=int(os.getenv("ZEPPELIN_MAX_REWRITES", defaults.max_rewrites)),
            multi_query=os.getenv("ZEPPELIN_MULTI_QUERY", "1" if defaults.multi_query else "0")
            not in ("0", "false", "no"),
            max_subqueries=int(os.getenv("ZEPPELIN_MAX_SUBQUERIES", defaults.max_subqueries)),
            max_chunks=int(os.getenv("ZEPPELIN_MAX_CHUNKS", defaults.max_chunks)),
        )
