# Runs

All the runfiles with metadata config files are placed under `$DATA/runs/2022`. First-pass retrieval output is placed under `$DATA/first_pass/2022`. Additionally, metadata config files are added to `/data/runs/2022` on git.

## Evaluation

Evaluation is performed on the [qrels](../../qrels/2022.txt) generated from provenance field.

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@3* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- | -- | -- |
| iai_smallboat - [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_v2_kilt_wapo) + T5](t5_canard_rewrites_2022.meta.yaml) | Automatic | 0.4158 | 0.1138 | 0.1465 | 0.1817 | 0.1055 | 0.1203 |