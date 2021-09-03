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

# Scores against pseudo qrels

| *Method* | Rewrite | *recall@1000* | *MRR* |
| -- | -- | -- | -- |
| [BM25](/config/bm25.yaml) | None | 0.1555 | 0.0212 |
| [BM25](/config/bm25_automatic.yaml) | Automatic | 0.3808 | 0.0497 |
| [BM25](/config/bm25_manual.yaml) | Manual | **0.4100** | 0.0503 |
| [BM25 (k1=4.46, b=0.82)](/config/bm25_k1_446_b_082_manual.yaml) | Manual | 0.3891 | 0.0305 |
| [BM25 (body)](/config/bm25_body_manual.yaml) | Manual | 0.3933 | 0.0417 |
| [BM25 (body, k1=4.46, b=0.82)](/config/bm25_body_k1_446_b_082_manual.yaml) | Manual | 0.3598 | 0.0316 |
| -- | -- | -- | -- |
| [BM25 + BERT MSMarco](config/fine_tuned/bm25_manual_msmarco.yaml) | Manual | **0.4100** | 0.0669 |
| [BM25 + BERT MSMarco (finetuned on treccast(Y1Y2))](config/fine_tuned/bm25_manual_msmarco_treccast.yaml) | Manual | **0.4100** | 0.0727 |
| [BM25 + BERT MSMarco (finetuned on Wizard of Wikipedia)](config/fine_tuned/bm25_manual_msmarco_wow.yaml) | Manual | **0.4100** | 0.0705 |
| [BM25 +  BERT MSMarco (finetuned on treccast(Y1Y2) + Wizard of Wikipedia)](config/fine_tuned/bm25_manual_msmarco_treccst_wow.yaml) | Manual | **0.4100** | 0.0741 |
| [BM25 + T5](/config/bm25_manual_rerank_t5.yaml) | Manual | **0.4100** | **0.1109** |
