# Runs

All the runfiles with metadata config files are placed under `$DATA/runs/2021`. First-pass retrieval output is placed under `$DATA/first_pass/2021`. Additionally metadata config files are added to `/data/runs/2021` on git.

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- | -- |
| [BM25 (ms_marco_kilt_wapo_clean) + T5](reproduce_raw_2021.meta.yaml) | None (raw) | 0.3497 | 0.1217 | 0.2875 | 0.2605 | 0.2041 |
| [BM25 (ms_marco_kilt_wapo_clean) + T5](reproduce_manual_2021.meta.yaml) | Manual | 0.7729 | 0.3858 | 0.7326 | 0.6291 | 0.5654 |
| [BM25 (ms_marco_kilt_wapo_clean) + T5](reproduce_automatic_2021.meta.yaml) | Automatic | 0.6319 | 0.2684 | 0.5575 | 0.4842 | 0.3969 |
| [BM25 (ms_marco_kilt_wapo_clean) + T5 (monoT5 + duoT5)](mono-duo-t5_automatic-2021.meta.yaml) | Automatic | 0.6319 | 0.2620 | 0.5544 | 0.4841 | 0.4094 |
| [T5-QReCC rewriter (WaterlooClarke) + BM25 (ms_marco_trec_car_clean) + T5](t5_qrecc_rewriter_2021.meta.yaml) | Automatic | 0.6095 | 0.2461 | 0.5269 | 0.4582 | 0.3741 | 
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_trec_car_clean) + T5](t5_canard_rewriter_2021.meta.yaml) | Automatic | 0.6666 | 0.2716 | 0.5335 | 0.5038 | 0.4035 |


## First-pass retrieval 

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* | 
| -- | -- | -- | -- | -- | -- | -- |
| [BM25 (ms_marco_kilt_wapo_clean)](first_pass_retireval/raw_bm25_2021.meta.yaml) | None (raw) | 0.3497 | 0.0797 | 0.2483 | 0.2203 | 0.1477 |
| [BM25 (ms_marco_kilt_wapo_clean)](first_pass_retireval/manual_bm25_2021.meta.yaml) | Manual | 0.7732 | 0.2470 | 0.6213 | 0.5310 | 0.4073 |
| [BM25 (ms_marco_kilt_wapo_clean)](first_pass_retireval/automatic_bm25_2021.meta.yaml) | Automatic | 0.6319 | 0.1770 | 0.4878 | 0.4184 | 0.3114 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_kilt_wapo_clean)](first_pass_retireval/t5_canard_bm25_2021.meta.yaml) | Automatic | 0.6666 | 0.1743 | 0.4763 | 0.4301 | 0.3031 | 
