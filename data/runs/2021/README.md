# Experiments 2021

All the runfiles with metadata config files are placed under `$DATA/runs/2021`. First-pass retrieval output is placed under `$DATA/first_pass/2021`. Additionally metadata config files are added to `/data/runs/2021` on git.

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- | -- |
| [BM25 (ms_marco_kilt_wapo_clean) + T5](raw_bm25_mono-t5_2021.meta.yaml) | None (raw) | 0.3497 | 0.1217 | 0.2875 | 0.2605 | 0.2041 |
| [BM25 (ms_marco_kilt_wapo_clean) + T5](automatic_bm25_mono-t5_2021.meta.yaml) | Automatic | 0.6319 | 0.2684 | 0.5575 | 0.4842 | 0.3969 |
| [T5-QReCC rewriter (WaterlooClarke) + BM25 (ms_marco_trec_car_clean) + T5](t5-qrecc_bm25_mono-t5_2021.meta.yaml) | Automatic | 0.6051 | 0.2531 | 0.5369 | 0.4684 | 0.3919 | 
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_trec_car_clean) + T5](t5-canard_bm25_mono-t5_2021.meta.yaml) | Automatic | 0.6519 | 0.2629 | 0.5354 | 0.4901 | 0.3975 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5 + duoT5)](ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_mono-duo-t5_2021.meta.yaml) | Automatic | 0.7393 | 0.2918 | 0.5573 | 0.5360 | 0.4089 |
| [T5-QReCC rewriter + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5 + duoT5)](ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_mono-duo-t5_2021.meta.yaml) | Automatic | 0.6987 | 0.2866 | 0.5702 | 0.5203 | 0.4102 |

## Different cascade architectures

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- | -- |
| [T5-CANARD rewriter + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5-QReCC rewriter + T5 (monoT5 + duoT5)](ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_t5-qrecc_mono-duo-t5_2021.trec) | Automatic | 0.7406 | 0.2885 | 0.5781 | 0.5361 | 0.4174 |
| [T5-QReCC rewriter + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5-CANARD rewriter + T5 (monoT5 + duoT5)](ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_t5-canard_mono-duo-t5_2021.trec) | Automatic | 0.6988 | 0.2942 | 0.5697 | 0.5264 | 0.4149 |
