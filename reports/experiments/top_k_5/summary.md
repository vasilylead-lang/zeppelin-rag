# RAGAS report — 2026-09-29

- samples: 25
- answer model: `claude-opus-5` (effort `medium`)
- grader / rewriter: `claude-haiku-4-5`
- judge: `claude-sonnet-5`
- embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, top_k=5, lexical_weight=0.5
- multi-query: off

| metric | mean |
|---|---|
| faithfulness | 0.986 |
| answer_relevancy | 0.838 |
| context_recall | 0.897 |
| context_precision_with_reference | 0.917 |
| factual_correctness | 0.667 |
