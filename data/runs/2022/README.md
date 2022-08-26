# Runs

All the runfiles with metadata config files are placed under `$DATA/runs/2022`. First-pass retrieval output is placed under `$DATA/first_pass/2022`. Additionally, metadata config files are added to `/data/runs/2022` on git.

## Evaluation

Evaluation is performed on the [qrels](../../qrels/2022.txt) generated from provenance field.

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@3* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- | -- | -- |
| iai_smallboat - [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_v2_kilt_wapo) + T5](t5_canard_rewrites_2022.meta.yaml) | Automatic | 0.4452 | 0.1414 | 0.1779 | 0.2116 | 0.1323 | 0.1519 |
| iai_duoboat - [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](t5-canard_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Automatic | 0.4452 | 0.1344 | 0.1644 | 0.2054 | 0.1241 | 0.1449 |
| (needs to be regenerated!) [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3) + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](prf_t5-canard_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Automatic | 0.4331 | 0.1064 | 0.1297 | 0.1775 | 0.1009 | 0.1151 |
| iai_loadedboat - [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](prf-17-26_t5-canard_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Automatic | 0.4711 | 0.1329 | 0.1628 | 0.2083 | 0.1255 | 0.1378 |
