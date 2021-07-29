# TREC CAsT 2021

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the IAI group's participation at the TREC 2021 Conversational Assistance Track (CAsT).

  * [TREC CAsT website](http://www.treccast.ai/)
  * [Y3 guidelines](https://docs.google.com/document/d/1Eo0IqQedYc_rfTw-YxbvUGTpoYSmejU0iDlUzQWj3_w/edit?usp=sharing)


## Architecture

Our system follows a conventional conversational passage retrieval pipeline (see, e.g., [Chatty Goose](https://dl.acm.org/doi/10.1145/3404835.3462782)) consisting of the following components:

  * [Indexer](treccast/indexer): Indexing for first-pass retrieval (ElasticSearch)
  * [Retriever](treccast/retriever): First-pass retrieval (BM25)
  * [Reranker](treccast/reranker): Neural reranker (BERT or T5)
  * [Rewriter](treccast/rewriter): Query rewriter (optional, can be applied to first-pass retrieval and/or reranking)


## Experiments 2020

Evaluation scores are generated using trec_eval:
```
$ {TREC_EVAL_PATH}/trec_eval -m all_trec data/qrels/2020.txt data/runs-2020/{RUNID}.trec
```

  * It assumes that [trec_eval](https://github.com/usnistgov/trec_eval) is installed locally under `{TREC_EVAL_PATH}`.

| *Method* | *Script* | *Index* | *MAP* | *MRR* | *NDCG* | *NDCG@5* | *recall@1000* |
| -- | -- | -- | -- | -- | -- | -- | -- |
| BM25 (default parameters) | `scripts/cast_bm25_default.sh` | `ms_marco_trec_car` | 0.0409 | 0.1780 | 0.1247 | 0.0614 ||
| BM25 (k1=4.46, b=0.82) | `scripts/cast_bm25_optimized.sh` | `ms_marco_trec_car` | 0.0229 | 0.1259 | 0.0870 | 0.0373 ||
| *Reference* ||||||
| Best @TREC2020 || 0.302 | 0.593 | 0.526 | ||
| TREC Organzers' auto baseline || 0.134 | 0.408 | 0.284 | ||

Running first-pass BM25 ranker on 2020 data:
```
  scripts/cast_bm25.sh 
```
