# RAGAS report — 2026-09-28

- samples: 25
- answer model: `claude-opus-5` (effort `medium`)
- grader / rewriter: `claude-haiku-4-5`
- judge: `claude-sonnet-5`
- embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, top_k=4, lexical_weight=0.5

| metric | mean |
|---|---|
| faithfulness | 0.981 |
| answer_relevancy | 0.829 |
| context_recall | 0.970 |
| context_precision_with_reference | 0.960 |
| factual_correctness | 0.458 |
