# Experiments 2020

There should be a script corresponding to each table row containing the exact parameterization.
Scripts are to be run from root.

All the runfiles with metadata config files are placed under `$DATA/runs/2020`. First-pass retrieval output is placed under `$DATA/first_pass/2020`. Additionally metadata config files are added to `/data/runs/2020` on git.

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* | 
| -- | -- | -- | -- | -- | -- | -- |
| [BM25 (ms_marco_trec_car_clean) + T5](reproduce_raw_2020.meta.yaml) | None (raw) | 0.2624 | 0.0979 | 0.2572 | 0.2098 | 0.1777 |
| [BM25 (ms_marco_trec_car_clean) + T5](reproduce_manual_2020.meta.yaml) | Manual | 0.6936 | 0.3384 | 0.7156 | 0.5839 | 0.5137 |
| [BM25 (ms_marco_trec_car_clean) + T5](reproduce_automatic_2020.meta.yaml) | Automatic | 0.4814 | 0.1953 | 0.4900 | 0.3907 | 0.3289 |
| (to be re-evaluated) [BM25 (ms_marco_trec_car_clean) + T5 (monoT5 + duoT5)](mono-duo-t5_automatic-2020.meta.yaml) | Automatic | 0.4820 | 0.1889 | 0.4851 | 0.3918 | 0.3446 |
| [PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5 + duoT5)](ance/prf-17-26_manual_ance_bm25-b-45-k-95_mono-duo-t5_2020.meta.yaml) | Manual | 0.8587 | 0.3814 | 0.7264 | 0.6596 | 0.5369 |
| [T5-QReCC rewriter (WaterlooClarke) + BM25 (ms_marco_trec_car_clean) + T5](t5-qrecc_bm25_mono-t5_2020.meta.yaml) | Automatic | 0.5098 | 0.2056 | 0.5102 | 0.4065 | 0.3496 | 
| [T5-QReCC rewriter (WaterlooClarke) + BM25 (ms_marco_trec_car_clean) + T5 (monoT5 + duoT5)](t5-qrecc_bm25_mono-duo-t5_2020.meta.yaml) | Automatic | 0.5098 | 0.2136 | 0.5299 | 0.4142 | 0.3707 | 
| [T5-QReCC rewriter + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5)](ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_mono-t5_2020.meta.yaml) | Automatic | 0.6608 | 0.2332 | 0.5116 | 0.4737 | 0.3622 |
| [T5-QReCC rewriter + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5 + duoT5)](ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_mono-duo-t5_2020.meta.yaml) | Automatic | 0.6608 | 0.2451 | 0.5355 | 0.4840 | 0.3846 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_trec_car_clean) + T5](t5-canard_bm25_mono-t5_2020.meta.yaml) | Automatic | 0.5275 | 0.2191 | 0.5455 | 0.4352 | 0.3675 | 
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_trec_car_clean) + T5 (monoT5 + duoT5)](t5-canard_bm25_mono-duo-t5_2020.meta.yaml) | Automatic | 0.5275 | 0.2218 | 0.5507 | 0.4387 | 0.3785 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + BM25 (ms_marco_trec_car_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5)](ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_mono-t5_2020.meta.yaml) | Automatic | 0.6878 | 0.2488 | 0.5205 | 0.4991 | 0.3697 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + BM25 (ms_marco_trec_car_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5 + duoT5)](ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_mono-duo-t5_2020.meta.yaml) | Automatic | 0.6878 | 0.2555 | 0.5541 | 0.5063 | 0.3891 |

## First-pass retrieval 

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* | 
| -- | -- | -- | -- | -- | -- | -- |
| [BM25 (ms_marco_trec_car_clean)](first_pass_retireval/raw_bm25_2020.meta.yaml) | None (raw) | 0.2624 | 0.0392 | 0.1148 | 0.1515 | 0.0771 |
| [BM25 (ms_marco_trec_car_clean)](first_pass_retireval/manual_bm25_2020.meta.yaml) | Manual | 0.6936 | 0.1393 | 0.3891 | 0.4192 | 0.2473 |
| [BM25 (ms_marco_trec_car_clean)](first_pass_retireval/automatic_bm25_2020.meta.yaml) | Automatic | 0.4820 | 0.0795 | 0.2450 | 0.2872 | 0.1542 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_trec_car_clean)](first_pass_retireval/t5_canard_bm25_2020.meta.yaml) | Automatic | 0.5275 | 0.0869 | 0.2519 | 0.3066 | 0.1559 | 
| [T5-QReCC rewriter + BM25 (ms_marco_trec_car_clean)](first_pass_retireval/t5_qrecc_bm25_2020.meta.yaml) | Automatic | 0.5098 | 0.0840 | 0.2662 | 0.2969 | 0.1612 | 
| [PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_raw_bm25-b-45-k-95_2020.meta.yaml) | None (raw) | 0.2922 | 0.0552 | 0.1506 | 0.1765 | 0.0942 |
| [PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_manual_bm25-b-45-k-95_2020.meta.yaml) | Manual | 0.7473 | 0.1676 | 0.4126 | 0.4619 | 0.2745 |
| [PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_automatic_bm25-b-45-k-95_2020.meta.yaml) | Automatic | 0.5351 | 0.1035 | 0.2838 | 0.3262 | 0.1756 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_t5-canard_bm25-b-45-k-95_2020.meta.yaml) | Automatic | 0.5800 | 0.1096 | 0.3037 | 0.3494 | 0.1940 |
| [T5-QReCC rewriter + PRF (RM3-17-26) + ANCE + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_t5-qrecc_bm25-b-45-k-95_2020.meta.yaml) | Automatic | 0.5594 | 0.1069 | 0.2983 | 0.3348 | 0.1810 |
| [PRF (RM3-17-26) + ANCE + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_raw_ance_bm25-b-45-k-95_2020.meta.yaml) | None (raw) | 0.3541 | 0.0831 | 0.2352 | 0.2307 | 0.1553 |
| [PRF (RM3-17-26) + ANCE + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_manual_ance_bm25-b-45-k-95_2020.meta.yaml) | Manual | 0.8587 | 0.2581 | 0.5606 | 0.5785 | 0.4016 |
| [PRF (RM3-17-26) + ANCE + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_automatic_ance_bm25-b-45-k-95_2020.meta.yaml) | Automatic | 0.6486 | 0.1607 | 0.3899 | 0.4140 | 0.2619 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + ANCE + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_t5-canard_ance_bm25-b-45-k-95_2020.meta.yaml) | Automatic | 0.6878 | 0.1642 | 0.4080 | 0.4364 | 0.2712 |
| [T5-QReCC rewriter + PRF (RM3-17-26) + ANCE + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_2020.meta.yaml) | Automatic | 0.6608 | 0.1692 | 0.4232 | 0.4279 | 0.2838 |
