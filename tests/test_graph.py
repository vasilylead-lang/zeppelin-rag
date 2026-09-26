from conftest import FakeLLM, KeywordEmbedder

from zeppelin_rag.config import Settings
from zeppelin_rag.graph import build_graph
from zeppelin_rag.retriever import HybridIndex

SETTINGS = Settings(top_k=2, max_rewrites=2)


def run(chunks, llm, question="гелий или водород?"):
    graph = build_graph(HybridIndex(chunks, KeywordEmbedder()), llm, SETTINGS)
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
    assert state["query"].endswith("(уточнено)")
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

    assert llm.calls.count("rewrite") == SETTINGS.max_rewrites
    assert llm.calls[-1] == "answer"
    assert len(state["documents"]) == SETTINGS.top_k
