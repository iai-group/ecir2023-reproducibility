# Runs

All the runfiles with metadata config files are placed under `$DATA/runs/2022`. First-pass retrieval output is placed under `$DATA/first_pass/2022`. Additionally, metadata config files are added to `/data/runs/2022` on git.

## Evaluation

Evaluation is performed on the [qrels](../../qrels/2022.txt) generated from provenance field.

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@3* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- | -- | -- |
| iai_smallboat - [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_v2_kilt_wapo) + T5](t5_canard_rewrites_2022.meta.yaml) | Automatic | 0.4158 | 0.1138 | 0.1465 | 0.1817 | 0.1055 | 0.1203 |
| iai_duoboat - [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](t5-canard_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Automatic | 0.4511 | 0.1275 | 0.1416 | 0.1976 | 0.1208 | 0.1373 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3) + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](prf_t5-canard_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Automatic | 0.4331 | 0.1064 | 0.1297 | 0.1775 | 0.1009 | 0.1151 |
| iai_loadedboat - [T5-CANARD rewriter (`castorini/t5-base-canard`) + PRF (RM3-17-26) + BM25 (ms_marco_v2_kilt_wapo, b=0.45m k1=0.95) + monoT5 + duoT5](prf-17-26_t5-canard_bm25-b-45-k-95_mono-duo-t5_2022.meta.yaml) | Automatic | 0.4331 | 0.1071 | 0.1304 | 0.1785 | 0.1009 | 0.1151 |
