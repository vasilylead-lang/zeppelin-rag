from conftest import KeywordEmbedder

from zeppelin_rag.retriever import BM25, HybridIndex


def test_search_ranks_matching_chunk_first(chunks):
    index = HybridIndex(chunks, KeywordEmbedder())

    assert index.search("чем гелий лучше водорода", k=1)[0].section == "Раздел 1"
    assert index.search("удлинение и сопротивление", k=1)[0].section == "Раздел 2"


def test_search_returns_k_results(chunks):
    index = HybridIndex(chunks, KeywordEmbedder())

    assert len(index.search("двигатель", k=2)) == 2


def test_bm25_matches_word_forms():
    bm25 = BM25(["Блау-газ хранили в ячейках", "Двигатели Maybach"])

    scores = bm25.scores("зачем нужен блау-газ")

    assert scores[0] > 0 and scores[1] == 0


def test_lexical_weight_rescues_rare_term(chunks):
    hybrid = HybridIndex(chunks, KeywordEmbedder(), lexical_weight=0.5)
    query = "Daimler-Benz"  # not in the embedder vocabulary

    assert hybrid.search(query, k=1)[0].section == "Раздел 3"
