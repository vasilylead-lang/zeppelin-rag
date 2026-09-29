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


def test_search_many_fuses_rankings_without_duplicates(chunks):
    index = HybridIndex(chunks, KeywordEmbedder())

    results = index.search_many(["двигатель", "гелий водород", "гелий"], k=1, limit=3)

    # "Раздел 1" is the top hit of two queries, so it outranks "Раздел 3" (one query)
    # even though "двигатель" came first; the repeated hit appears once.
    assert [c.section for c in results] == ["Раздел 1", "Раздел 3"]


def test_search_many_respects_limit(chunks):
    index = HybridIndex(chunks, KeywordEmbedder())

    assert len(index.search_many(["гелий", "двигатель", "удлинение"], k=2, limit=2)) == 2
