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

Evaluation scores are generated using trec_eval (note that relevance threshold is >=2 for binary measures!):
```
$ {TREC_EVAL_PATH}/trec_eval -m all_trec -l2 data/qrels/2020.txt data/runs-2020/{RUNID}.trec
```

  * It assumes that [trec_eval](https://github.com/usnistgov/trec_eval) is installed locally under `{TREC_EVAL_PATH}`.

There should be a script corresponding to each table row containing the exact parameterization.
Scripts are to be run from root.

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- | -- |
| [BM25](scripts/2020/cast_bm25_default.sh) | None (raw) | 0.2093 | 0.0293 | 0.1080 | 0.1247 | 0.0614 |
| [BM25 (Stopword, KStem)](scripts/2020/cast_bm25_clean.sh) | None (raw) | 0.2608 | 0.0385 | 0.1078 | 0.1509 | 0.0787 |
| BM25 | Automatic | | | | | |
| BM25 | Manual | | | | | |
| [BM25+BERT base](scripts/2020/cast_bm25_default_rerank.sh) | None (raw) | 0.2093 | 0.0792 | 0.2291 | 0.1801 | 0.1586 |
| BM25+BERT base | Automatic | | | | | |
| BM25+BERT base | Manual | | | | | |
| *Reference* |||||
| Organizers' baseline (??) | None (raw) | 0.1480 | 0.0658 | 0.2245 | 0.1437 | 0.1591 |
| Organizers' baseline (BERT) | Automatic | 0.3084 | 0.1344 | 0.4084 | 0.2840 | 0.2865 |
| Organizers' baseline (BERT) | Manual | 0.4981 | 0.2524 | 0.6516 | 0.4513 | 0.4609 |
| Organizers' baseline (SDM) | Manual | 0.7701 | 0.1949 | 0.4794 | 0.4926 | 0.3113 |
| Best @TREC2020 (grill_bmDuo) | Manual | 0.747 | 0.302 | 0.684 | 0.571 | |
| Best @TREC2020 (h2oloo_RUN4) | Automatic | 0.633 | 0.302 | 0.593 | 0.526 | |
| Best @TREC2020 w/ canonical results (h2oloo_RUN2) | Automatic | 0.705 | 0.326 | 0.621 | 0.575 |
