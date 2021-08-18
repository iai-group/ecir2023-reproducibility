# Runs

## First pass - BM25

  * `bm25.trec` -> raw questions
  * `bm25_manual.trec` -> Manual rewrites
  * `bm25_automatic.trec` -> Automatic rewrites
  * `bm25_manual_k1_446_b_082.trec` -> Manual rewrites, organizers bm25 parameters 
  * `bm25_manual_body.trec` -> Manual rewrites, body field only
  * `bm25_manual_body_k1_446_b_082.trec` -> Manual rewrites, body field only, organizers bm25 parameters

## Reranking

  * `bm25_manual_rerank_t5.trec` -> Manual rewrites, T5 reranker