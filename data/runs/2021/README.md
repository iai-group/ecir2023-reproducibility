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
| [BM25 + BERT MSMarco](/config/fine_tuning/bm25_manual_msmarco.yaml) | Manual | **0.4100** | 0.0669 |
| [BM25 + BERT MSMarco (finetuned on treccast(2019-2020))](/config/fine_tuning/bm25_manual_msmarco_treccast.yaml) | Manual | **0.4100** | 0.0727 |
| [BM25 + BERT MSMarco (finetuned on Wizard of Wikipedia)](/config/fine_tuning/bm25_manual_msmarco_wow.yaml) | Manual | **0.4100** | 0.0705 |
| [BM25 +  BERT MSMarco (finetuned on treccast(2019-2020) + Wizard of Wikipedia)](/config/fine_tuning/bm25_manual_msmarco_treccast_wow.yaml) | Manual | **0.4100** | 0.0741 |
| [BM25 (Ensemble) + BERT MSMarco](/config/fine_tuning/bm25_ensemble_manual_msmarco.yaml) | Manual | **0.4812** | 0.0702 |
| [BM25 (Ensemble) + BERT MSMarco (finetuned on treccast(2019-2020))](/config/fine_tuning/bm25_ensemble_manual_msmarco_treccast.yaml) | Manual | **0.4728** | 0.0705 |
| [BM25 (Ensemble) + BERT MSMarco (finetuned on Wizard of Wikipedia)](/config/fine_tuning/bm25_ensemble_manual_msmarco_wow.yaml) | Manual | **0.4686** | 0.0703 |
| [BM25 (Ensemble) +  BERT MSMarco (finetuned on treccast(2019-2020) + Wizard of Wikipedia)](/config/fine_tuning/bm25_ensemble_manual_msmarco_treccast_wow.yaml) | Manual | **0.4854** | 0.0725 |
| [BM25 + T5](/config/bm25_manual_rerank_t5.yaml) | Manual | **0.4100** | **0.1109** |


# Reproduced results

All the runfiles with metadata config files are placed under `$DATA/runs/2021`. First-pass retrieval output is placed under `$DATA/first_pass/2021`. Additionally metadata config files are added to `/data/runs/2021` on git.

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- | -- |
| [BM25 (ms_marco_kilt_wapo_clean) + T5](reproduce_raw_2021.meta.yaml) | None (raw) | 0.3497 | 0.1217 | 0.2875 | 0.2605 | 0.2041 |
| [BM25 (ms_marco_kilt_wapo_clean) + T5](reproduce_manual_2021.meta.yaml) | Manual | 0.7729 | 0.3858 | 0.7326 | 0.6291 | 0.5654 |
| [BM25 (ms_marco_kilt_wapo_clean) + T5](reproduce_automatic_2021.meta.yaml) | Automatic | 0.6319 | 0.2684 | 0.5575 | 0.4842 | 0.3969 |
| [BM25 (ms_marco_kilt_wapo_clean) + T5 (monoT5 + duoT5)](mono-duo-t5_automatic-2021.meta.yaml) | Automatic | 0.6319 | 0.2620 | 0.5544 | 0.4841 | 0.4094 |
| [T5-QReCC rewriter (WaterlooClarke) + BM25 (ms_marco_trec_car_clean) + T5](t5_qrecc_rewriter_2021.meta.yaml) | Automatic | 0.6095 | 0.2461 | 0.5269 | 0.4582 | 0.3741 | 
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_trec_car_clean) + T5](t5_canard_rewriter_2021.meta.yaml) | Automatic | 0.6666 | 0.2716 | 0.5335 | 0.5038 | 0.4035 |
