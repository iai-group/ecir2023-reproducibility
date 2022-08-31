# Runs

All the runfiles with metadata config files are placed under `$DATA/runs/2022`. First-pass retrieval output is placed under `$DATA/first_pass/2022`. Additionally, metadata config files are added to `/data/runs/2022` on git.

## Evaluation

Evaluation is performed on the [qrels](../../qrels/2022.txt) generated from provenance field.

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@3* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- | -- | -- |
| raw - [raw rewrites + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](raw_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Raw | 0.1631 | 0.0371 | 0.0533 | 0.0677 | 0.0295 | 0.0374 |
| automatic - [automatic rewrites + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](automatic_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Automatic | 0.5056 | 0.1670 | 0.2058 | 0.2460 | 0.1603 | 0.1778 |
| manual - [manual rewrites + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](manual_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Manual | 0.6598 | 0.2402 | 0.2776 | 0.3409 | 0.2174 | 0.2477 |
| raw - [raw rewrites + PRF (RM3-17-26) + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](prf-17-26_raw_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Raw | 0.1701 | 0.0336 | 0.0496 | 0.0662 | 0.0244 | 0.0361 |
| automatic - [automatic rewrites + PRF (RM3-17-26) + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](prf-17-26_automatic_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Automatic | 0.5237 | 0.1665 | 0.2055 | 0.2488 | 0.1615 | 0.1787 |
| manual - [manual rewrites + PRF (RM3-17-26) + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](manual_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Manual | 0.6718 | 0.2432 | 0.2797 | 0.3442 | 0.2199 | 0.2513 |
| iai_smallboat - [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_v2_kilt_wapo) + T5](t5_canard_rewrites_2022.meta.yaml) | Automatic | 0.4452 | 0.1414 | 0.1779 | 0.2116 | 0.1323 | 0.1519 |
| iai_duoboat - [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](t5-canard_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Automatic | 0.4452 | 0.1344 | 0.1644 | 0.2054 | 0.1241 | 0.1449 |
| iai_loadedboat_default_PRF_params [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3) + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](prf_t5-canard_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Automatic | 0.4711 | 0.1293 | 0.1580 | 0.2053 | 0.1169 | 0.1372 |
| iai_loadedboat - [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](prf-17-26_t5-canard_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Automatic | 0.4711 | 0.1329 | 0.1628 | 0.2083 | 0.1255 | 0.1378 |
