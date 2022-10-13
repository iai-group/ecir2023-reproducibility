# Experiments 2020

All the runfiles with metadata config files are placed under `$DATA/runs/2020`. First-pass retrieval output is placed under `$DATA/first_pass/2020`. Additionally metadata config files are added to `/data/runs/2020` on git.

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* | 
| -- | -- | -- | -- | -- | -- | -- |
| [BM25 (ms_marco_trec_car_clean) + T5](reproduce_raw_2020.meta.yaml) | None (raw) | 0.2624 | 0.0979 | 0.2572 | 0.2098 | 0.1777 |
| [BM25 (ms_marco_trec_car_clean) + T5](reproduce_automatic_2020.meta.yaml) | Automatic | 0.4814 | 0.1953 | 0.4900 | 0.3907 | 0.3289 |
| (to be re-evaluated) [BM25 (ms_marco_trec_car_clean) + T5 (monoT5 + duoT5)](mono-duo-t5_automatic-2020.meta.yaml) | Automatic | 0.4820 | 0.1889 | 0.4851 | 0.3918 | 0.3446 |
| [T5-QReCC rewriter (WaterlooClarke) + BM25 (ms_marco_trec_car_clean) + T5](t5-qrecc_bm25_mono-t5_2020.meta.yaml) | Automatic | 0.5098 | 0.2056 | 0.5102 | 0.4065 | 0.3496 | 
| [T5-QReCC rewriter + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5 + duoT5)](ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_mono-duo-t5_2020.meta.yaml) | Automatic | 0.6608 | 0.2451 | 0.5355 | 0.4840 | 0.3846 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_trec_car_clean) + T5](t5-canard_bm25_mono-t5_2020.meta.yaml) | Automatic | 0.5275 | 0.2191 | 0.5455 | 0.4352 | 0.3675 | 
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + BM25 (ms_marco_trec_car_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5 + duoT5)](ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_mono-duo-t5_2020.meta.yaml) | Automatic | 0.6878 | 0.2555 | 0.5541 | 0.5063 | 0.3891 |

## Different cascade architectures

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- | -- |
| [T5-CANARD rewriter + PRF (RM3-17-26) + BM25 (ms_marco_trec_car_clean, b=0.45m k1=0.95) + ANCE + T5-QReCC rewriter + T5 (monoT5 + duoT5)](ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_t5-qrecc_mono-duo-t5_2020.trec) | Automatic | 0.6878 | 0.2385 | 0.5266 | 0.4883 | 0.3721 |
| [T5-QReCC rewriter + PRF (RM3-17-26) + BM25 (ms_marco_trec_car_clean, b=0.45m k1=0.95) + ANCE + T5-CANARD rewriter + T5 (monoT5 + duoT5)](ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_t5-canard_mono-duo-t5_2020.trec) | Automatic | 0.6608 | 0.2432 | 0.5437 | 0.4842 | 0.3846 |
