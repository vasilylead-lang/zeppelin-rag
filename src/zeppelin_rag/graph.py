"""Corrective RAG with multi-query retrieval as a LangGraph state machine.

decompose -> retrieve -> grade +--(relevant chunks found)--> generate -> END
                                +--(nothing relevant)--> rewrite -> retrieve  (max_rewrites)

decompose splits the question into sub-queries (the question itself stays first);
retrieve searches each and fuses the rankings. With multi_query off, the question
is the only query and decompose makes no LLM call. If rewrites are exhausted,
generate still runs on the last retrieved chunks and the answer prompt tells the
model to admit when the knowledge base has no answer.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from zeppelin_rag.config import Settings
from zeppelin_rag.knowledge import Chunk
from zeppelin_rag.llm import RagLLM
from zeppelin_rag.retriever import HybridIndex


class RAGState(TypedDict, total=False):
    question: str
    queries: list[str]
    retrieved: list[Chunk]
    documents: list[Chunk]
    answer: str
    rewrites: int
    steps: Annotated[list[str], operator.add]


def build_graph(index: HybridIndex, llm: RagLLM, settings: Settings):
    def decompose(state: RAGState) -> RAGState:
        question = state["question"]
        if not settings.multi_query:
            return {"queries": [question]}
        queries = [question]
        for sub in llm.decompose(question, settings.max_subqueries):
            if sub not in queries:
                queries.append(sub)
        return {"queries": queries, "steps": [f"decompose: {' | '.join(queries[1:])}"]}

    def retrieve(state: RAGState) -> RAGState:
        queries = state["queries"]
        if len(queries) == 1:
            chunks = index.search(queries[0], settings.top_k)
        else:
            chunks = index.search_many(queries, settings.top_k, settings.max_chunks)
        return {"retrieved": chunks, "steps": [f"retrieve: {len(queries)} queries"]}

    def grade(state: RAGState) -> RAGState:
        retrieved = state["retrieved"]
        keep = llm.grade(state["question"], retrieved)
        documents = [retrieved[i] for i in keep]
        return {"documents": documents, "steps": [f"grade: {len(documents)}/{len(retrieved)}"]}

    def rewrite(state: RAGState) -> RAGState:
        query = llm.rewrite(state["question"], " | ".join(state["queries"]))
        return {
            "queries": [query],
            "rewrites": state.get("rewrites", 0) + 1,
            "steps": [f"rewrite: {query}"],
        }

    def generate(state: RAGState) -> RAGState:
        documents = state.get("documents") or state["retrieved"]
        answer = llm.answer(state["question"], documents)
        return {"documents": documents, "answer": answer, "steps": ["generate"]}

    def after_grade(state: RAGState) -> str:
        if state["documents"] or state.get("rewrites", 0) >= settings.max_rewrites:
            return "generate"
        return "rewrite"

    builder = StateGraph(RAGState)
    builder.add_node("decompose", decompose)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("rewrite", rewrite)
    builder.add_node("generate", generate)
    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade", after_grade, {"generate": "generate", "rewrite": "rewrite"}
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)
    return builder.compile()


def build_default_graph(settings: Settings | None = None):
    """Wires the real embedder and Claude client; needs ANTHROPIC_API_KEY."""
    from zeppelin_rag.knowledge import load_chunks
    from zeppelin_rag.llm import ClaudeRagLLM
    from zeppelin_rag.retriever import FastEmbedEmbedder

    settings = settings or Settings.from_env()
    index = HybridIndex(
        load_chunks(), FastEmbedEmbedder(settings.embedding_model), settings.lexical_weight
    )
    return build_graph(index, ClaudeRagLLM(settings), settings), index
