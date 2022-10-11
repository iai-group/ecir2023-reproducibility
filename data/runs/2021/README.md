# Runs

All the runfiles with metadata config files are placed under `$DATA/runs/2021`. First-pass retrieval output is placed under `$DATA/first_pass/2021`. Additionally metadata config files are added to `/data/runs/2021` on git.

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- | -- |
| [BM25 (ms_marco_kilt_wapo_clean) + T5](reproduce_raw_2021.meta.yaml) | None (raw) | 0.3497 | 0.1217 | 0.2875 | 0.2605 | 0.2041 |
| [BM25 (ms_marco_kilt_wapo_clean) + T5](reproduce_manual_2021.meta.yaml) | Manual | 0.7733 | 0.3858 | 0.7326 | 0.6293 | 0.5654 |
| [BM25 (ms_marco_kilt_wapo_clean) + T5](reproduce_automatic_2021.meta.yaml) | Automatic | 0.6319 | 0.2684 | 0.5575 | 0.4842 | 0.3969 |
| (to be re-evaluated) [BM25 (ms_marco_kilt_wapo_clean) + T5 (monoT5 + duoT5)](mono-duo-t5_automatic-2021.meta.yaml) | Automatic | 0.6319 | 0.2620 | 0.5544 | 0.4841 | 0.4094 |
| [T5-QReCC rewriter (WaterlooClarke) + BM25 (ms_marco_trec_car_clean) + T5](t5-qrecc_bm25_mono-t5_2021.meta.yaml) | Automatic | 0.6051 | 0.2531 | 0.5369 | 0.4684 | 0.3919 | 
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_trec_car_clean) + T5](t5-canard_bm25_mono-t5_2021.meta.yaml) | Automatic | 0.6519 | 0.2629 | 0.5354 | 0.4901 | 0.3975 |
| [PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5 + duoT5)](ance/prf-17-26_manual_ance_bm25-b-45-k-95_mono-duo-t5_2021.meta.yaml) | Manual | 0.8790 | 0.4287 | 0.7465 | 0.6824 | 0.5856 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_kilt_wapo_clean) + T5 (monoT5 + duoT5)](t5-canard_bm25_mono-duo-t5_2021.meta.yaml) | Automatic | 0.6519 | 0.2662 | 0.5596 | 0.4957 | 0.4109 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5 + duoT5)](ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_mono-duo-t5_2021.meta.yaml) | Automatic | 0.7393 | 0.2918 | 0.5573 | 0.5360 | 0.4089 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5)](ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_mono-t5_2021.meta.yaml) | Automatic | 0.7393 | 0.2881 | 0.5520 | 0.5327 | 0.3997 |
| [T5-QReCC rewriter + BM25 (ms_marco_kilt_wapo_clean) + T5 (monoT5 + duoT5)](/t5-qrecc_bm25_mono-duo-t5_2021.meta.yaml) | Automatic | 0.6038 | 0.2539 | 0.5508 | 0.4723 | 0.4051 |
| [T5-QReCC rewriter + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5 + duoT5)](ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_mono-duo-t5_2021.meta.yaml) | Automatic | 0.6987 | 0.2866 | 0.5702 | 0.5203 | 0.4102 |
| [T5-QReCC rewriter + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE + T5 (monoT5)](ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_mono-t5_2021.meta.yaml) | Automatic | 0.6987 | 0.2837 | 0.5584 | 0.5165 | 0.4025 |


## First-pass retrieval 

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* | 
| -- | -- | -- | -- | -- | -- | -- |
| [BM25 (ms_marco_kilt_wapo_clean)](first_pass_retrieval/raw_bm25_2021.meta.yaml) | None (raw) | 0.3497 | 0.0797 | 0.2483 | 0.2203 | 0.1477 |
| [BM25 (ms_marco_kilt_wapo_clean)](first_pass_retrieval/manual_bm25_2021.meta.yaml) | Manual | 0.7732 | 0.2470 | 0.6213 | 0.5310 | 0.4073 |
| [BM25 (ms_marco_kilt_wapo_clean)](first_pass_retrieval/automatic_bm25_2021.meta.yaml) | Automatic | 0.6319 | 0.1770 | 0.4878 | 0.4184 | 0.3114 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_kilt_wapo_clean)](first_pass_retrieval/t5_canard_bm25_2021.meta.yaml) | Automatic | 0.6519 | 0.1721 | 0.4697 | 0.4201 | 0.2942 | 
| [T5-QReCC rewriter + BM25 (ms_marco_kilt_wapo_clean)](first_pass_retrieval/t5-qrecc_bm25_2021.meta.yaml) | Automatic | 0.6052 | 0.1649 | 0.4637 | 0.4058 | 0.2975 | 
| [PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE](first_pass_retrieval/prf-17-26_raw_ance_bm25-b-45-k-95_2021.meta.yaml) | None (raw) | 0.4153 | 0.1249 | 0.3021 | 0.2768 | 0.1977 |
| [PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE](first_pass_retrieval/prf-17-26_manual_ance_bm25-b-45-k-95_2021.meta.yaml) | Manual | 0.8790 | 0.3452 | 0.6830 | 0.6303 | 0.4989 |
| [PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE](first_pass_retrieval/prf-17-26_automatic_ance_bm25-b-45-k-95_2021.meta.yaml) | Automatic | 0.7193 | 0.2426 | 0.5194 | 0.4937 | 0.3698 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE](first_pass_retrieval/prf-17-26_t5-canard_ance_bm25-b-45-k-95_2021.meta.yaml) | Automatic | 0.7393 | 0.2399 | 0.5247 | 0.5008 | 0.3655 |
| [T5-QReCC rewriter + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95) + ANCE](first_pass_retrieval/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_2021.meta.yaml) | Automatic | 0.6987 | 0.2359 | 0.5507 | 0.4900 | 0.3765 |
| [PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_raw_bm25-b-45-k-95_2021.meta.yaml) | None (raw) | 0.3620 | 0.0869 | 0.2457 | 0.2303 | 0.1542 |
| [PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_manual_bm25-b-45-k-95_2021.meta.yaml) | Manual | 0.7851 | 0.2576 | 0.6230 | 0.5417 | 0.4130 |
| [PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_automatic_bm25-b-45-k-95_2021.meta.yaml) | Automatic | 0.6494 | 0.1894 | 0.4801 | 0.4325 | 0.3152 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_t5-canard_bm25-b-45-k-95_2021.meta.yaml) | Automatic | 0.6601 | 0.1804 | 0.4585 | 0.4284 | 0.2950 |
| [T5-QReCC rewriter + PRF (RM3-17-26) + BM25 (ms_marco_kilt_wapo_clean, b=0.45m k1=0.95)](first_pass_retrieval/prf-17-26_t5-qrecc_bm25-b-45-k-95_2021.meta.yaml) | Automatic | 0.6248 | 0.1733 | 0.4419 | 0.4177 | 0.2952 |

## Evaluation on 2021 synthetic qrels

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@3* | *NDCG@5* | 
| -- | -- | -- | -- | -- | -- | -- | -- |
| [BM25 (ms_marco_trec_car_clean) + T5](reproduce_raw_2021.meta.yaml) | None (raw) | 0.1597 | 0.0255 | 0.0255 | 0.0503 | 0.0200 | 0.0253 |
| [BM25 (ms_marco_trec_car_clean) + T5](reproduce_manual_2021.meta.yaml) | Manual | 0.2594 | 0.0692 | 0.0692 | 0.1076 | 0.0550 | 0.0689 |
| [BM25 (ms_marco_trec_car_clean) + T5](reproduce_automatic_2021.meta.yaml) | Automatic | 0.2343 | 0.0567 | 0.0567 | 0.0926 | 0.0419 | 0.0603 |
| [BM25 (ms_marco_trec_car_clean) + T5 (monoT5 + duoT5)](mono-duo-t5_automatic_2021.meta.yaml) | Automatic | 0.2343 | 0.0757 | 0.0757 | 0.1071 | 0.0670 | 0.0775 |
| [T5-QReCC rewriter (WaterlooClarke) + BM25 (ms_marco_trec_car_clean) + T5](t5_qrecc_rewriter_2021.meta.yaml) | Automatic | 0.2176 | 0.0461 | 0.0461 | 0.0798 | 0.0388 | 0.0526 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_trec_car_clean) + T5](t5_canard_rewriter_2021.meta.yaml) | Automatic | 0.2301 | 0.0600 | 0.0600 | 0.0945 | 0.0493 | 0.0646 |
