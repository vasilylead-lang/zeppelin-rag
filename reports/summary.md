# RAGAS report — 2026-09-28

- samples: 17
- answer model: `claude-opus-5` (effort `medium`)
- grader / rewriter: `claude-haiku-4-5`
- judge: `claude-sonnet-5`
- embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, top_k=4, lexical_weight=0.5

| metric | mean |
|---|---|
| faithfulness | 0.968 |
| answer_relevancy | 0.866 |
| context_recall | 0.980 |
| context_precision_with_reference | 0.961 |
| factual_correctness | 0.537 |
