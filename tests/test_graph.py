from conftest import FakeLLM, KeywordEmbedder

from zeppelin_rag.config import Settings
from zeppelin_rag.graph import build_graph
from zeppelin_rag.retriever import HybridIndex

SINGLE = Settings(top_k=2, max_rewrites=2, multi_query=False)
MULTI = Settings(top_k=1, max_rewrites=2, multi_query=True, max_subqueries=3, max_chunks=3)


def run(chunks, llm, settings=SINGLE, question="гелий или водород?"):
    graph = build_graph(HybridIndex(chunks, KeywordEmbedder()), llm, settings)
    return graph.invoke({"question": question, "steps": []})


def test_relevant_chunks_go_straight_to_generate(chunks):
    llm = FakeLLM(grades=[[0]])

    state = run(chunks, llm)

    assert llm.calls == ["grade", "answer"]
    assert [c.section for c in state["documents"]] == ["Раздел 1"]
    assert state["answer"] == "ответ по 1 фрагментам"


def test_irrelevant_chunks_trigger_rewrite_then_generate(chunks):
    llm = FakeLLM(grades=[[], [1]])

    state = run(chunks, llm)

    assert llm.calls == ["grade", "rewrite", "grade", "answer"]
    assert state["rewrites"] == 1
    assert state["queries"] == ["гелий или водород? (уточнено)"]
    assert [s.split(":")[0] for s in state["steps"]] == [
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
    ]


def test_rewrites_are_capped_and_fall_back_to_retrieved(chunks):
    llm = FakeLLM(grades=[[], [], []])

    state = run(chunks, llm)

    assert llm.calls.count("rewrite") == SINGLE.max_rewrites
    assert llm.calls[-1] == "answer"
    assert len(state["documents"]) == SINGLE.top_k


def test_multi_query_retrieves_chunks_for_every_subquery(chunks):
    llm = FakeLLM(grades=[[0, 1]], subqueries=["двигатель Гинденбурга"])

    state = run(chunks, llm, MULTI)

    assert llm.calls == ["decompose", "grade", "answer"]
    assert state["queries"] == ["гелий или водород?", "двигатель Гинденбурга"]
    # top_k=1 per query: each query contributes its own best chunk.
    assert {c.section for c in state["retrieved"]} == {"Раздел 1", "Раздел 3"}
    assert state["steps"][0] == "decompose: двигатель Гинденбурга"


def test_multi_query_drops_duplicate_of_question(chunks):
    llm = FakeLLM(grades=[[0]], subqueries=["гелий или водород?"])

    state = run(chunks, llm, MULTI)

    assert state["queries"] == ["гелий или водород?"]
    assert len(state["retrieved"]) == MULTI.top_k
