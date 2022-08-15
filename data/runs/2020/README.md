# Experiments 2020

Evaluation scores are generated using trec_eval (note that relevance threshold is >=2 for binary measures!):
```
$ {TREC_EVAL_PATH}/trec_eval -m all_trec -l2 data/qrels/2020.txt data/runs-2020/{RUNID}.trec
```

  * It assumes that [trec_eval](https://github.com/usnistgov/trec_eval) is installed locally under `{TREC_EVAL_PATH}`.

There should be a script corresponding to each table row containing the exact parameterization.
Scripts are to be run from root.

All the runfiles with metadata config files are placed under `$DATA/runs/2020`. First-pass retrieval output is placed under `$DATA/first_pass/2020`. Additionally metadata config files are added to `/data/runs/2020` on git.

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* | 
| -- | -- | -- | -- | -- | -- | -- |
| [BM25 (ms_marco_trec_car_clean) + T5](reproduce_raw_2020.meta.yaml) | None (raw) | 0.2624 | 0.0979 | 0.2572 | 0.2098 | 0.1777 |
| [BM25 (ms_marco_trec_car_clean) + T5](reproduce_manual_2020.meta.yaml) | Manual | 0.6936 | 0.3384 | 0.7156 | 0.5839 | 0.5137 |
| [BM25 (ms_marco_trec_car_clean) + T5](reproduce_automatic_2020.meta.yaml) | Automatic | 0.4820 | 0.1953 | 0.4900 | 0.3909 | 0.3291 |
| [BM25 (ms_marco_trec_car_clean) + T5 (monoT5 + duoT5)](mono-duo-t5_automatic-2020.meta.yaml) | Automatic | 0.4820 | 0.1889 | 0.4851 | 0.3918 | 0.3446 |
| [T5-QReCC rewriter (WaterlooClarke) + BM25 (ms_marco_trec_car_clean) + T5](t5_qrecc_rewriter_2020.meta.yaml) | Automatic | 0.5122 | 0.2083 | 0.4979 | 0.4084 | 0.3515 | 
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_trec_car_clean) + T5](t5_canard_rewriter_2020.meta.yaml) | Automatic | 0.5270 | 0.2264 | 0.5323 | 0.4396 | 0.3768 | 

## First-pass retrieval 

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* | 
| -- | -- | -- | -- | -- | -- | -- |
| [BM25 (ms_marco_trec_car_clean)](first_pass_retireval/raw_bm25_2020.meta.yaml) | None (raw) | 0.2624 | 0.0392 | 0.1148 | 0.1515 | 0.0771 |
| [BM25 (ms_marco_trec_car_clean)](first_pass_retireval/manual_bm25_2020.meta.yaml) | Manual | 0.6936 | 0.1393 | 0.3891 | 0.4192 | 0.2473 |
| [BM25 (ms_marco_trec_car_clean)](first_pass_retireval/automatic_bm25_2020.meta.yaml) | Automatic | 0.4820 | 0.0795 | 0.2450 | 0.2872 | 0.1542 |
| [T5-CANARD rewriter (`castorini/t5-base-canard`) + BM25 (ms_marco_trec_car_clean)](first_pass_retireval/t5_canard_bm25_2020.meta.yaml) | Automatic | 0.5270 | 0.0892 | 0.2734 | 0.3088 | 0.1640 | 
