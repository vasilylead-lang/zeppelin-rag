# RAGAS report — 2026-09-28

- samples: 25
- answer model: `claude-opus-5` (effort `medium`)
- grader / rewriter: `claude-haiku-4-5`
- judge: `claude-sonnet-5`
- embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, top_k=4, lexical_weight=0.5
- multi-query: off

| metric | mean |
|---|---|
| faithfulness | 0.974 |
| answer_relevancy | 0.823 |
| context_recall | 0.899 |
| context_precision_with_reference | 0.920 |
| factual_correctness | 0.724 |
