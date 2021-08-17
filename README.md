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
| [BM25](scripts/2020/cast_bm25_default.sh) | None (raw) | 0.2133 | 0.0306 | 0.1145 | 0.1272 | 0.0642 |
| [BM25](scripts/2020/cast_bm25_default_Automatic.sh) | Automatic | 0.4387 | 0.0796 | 0.3408 | 0.2377 | 0.1255 |
| [BM25](scripts/2020/cast_bm25_default_manual.sh) | Manual | 0.6301 | 0.1405 | 0.5246 | 0.3611 | 0.2169 |
| [BM25 (Stopword, KStem)](scripts/2020/cast_bm25_clean.sh) | None (raw) | 0.2620 | 0.0379 | 0.1097 | 0.1497 | 0.0750 |
| [BM25 (Stopword, KStem)](scripts/2020/cast_bm25_clean_automatic.sh) | Automatic | 0.5121 | 0.1113 | 0.3844 | 0.2872 | 0.1542 |
| [BM25 (Stopword, KStem)](scripts/2020/cast_bm25_clean_manual.sh) | Manual | 0.7202 | 0.1837 | 0.5551 | 0.4192 | 0.2473 |
| [BM25 + BERT](scripts/2020/cast_bm25_default_rerank_bert.sh) | None (raw) | 0.2093 | 0.0792 | 0.2291 | 0.1801 | 0.1586 |
| [BM25 + BERT](scripts/2020/cast_bm25_default_automatic_rerank_bert.sh) | Automatic | 0.4095 | 0.1638 | 0.4467 | 0.3427 | 0.3020 |
| [BM25 + BERT](scripts/2020/cast_bm25_default_manual_rerank_bert.sh) | Manual | 0.6138 | 0.2802 | 0.6336 | 0.5154 | 0.4689 |
| [BM25 (Stopword, KStem) + BERT](scripts/2020/cast_bm25_clean_rerank_bert.sh) | None (raw) | 0.2620 | 0.0906 | 0.2470 | 0.2050 | 0.1696 |
| [BM25 (Stopword, KStem) + BERT](scripts/2020/cast_bm25_clean_automatic_rerank_bert.sh) | Automatic | 0.4820 | 0.1886 | 0.4606 | 0.3839 | 0.3123 |
| [BM25 (Stopword, KStem) + BERT](scripts/2020/cast_bm25_clean_manual_rerank_bert.sh) | Manual | 0.6936 | 0.3016 | 0.6325 | 0.5581 | 0.4732 |
| [BM25 (Stopword, KStem) + T5](scripts/2020/cast_bm25_clean_rerank_t5.sh) | None (raw) | 0.2620 | 0.0959 | 0.2545 | 0.2080 | 0.1746 |
| [BM25 (Stopword, KStem) + T5](scripts/2020/cast_bm25_clean_automatic_rerank_t5.sh) | Automatic | 0.4820 | 0.1953 | 0.4900 | 0.3909 | 0.3291 |
| [BM25 (Stopword, KStem) + T5](scripts/2020/cast_bm25_clean_manual_rerank_t5.sh) | Manual | 0.6936 | 0.3384 | 0.7156 | 0.5839 | 0.5137 |
| *Reference* |||||
| Organizers' baseline (??) | None (raw) | 0.1480 | 0.0658 | 0.2245 | 0.1437 | 0.1591 |
| Organizers' baseline (BERT) | Automatic | 0.3084 | 0.1344 | 0.4084 | 0.2840 | 0.2865 |
| Organizers' baseline (BERT) | Manual | 0.4981 | 0.2524 | 0.6516 | 0.4513 | 0.4609 |
| Organizers' baseline (SDM) | Manual | 0.7701 | 0.1949 | 0.4794 | 0.4926 | 0.3113 |
| Best @TREC2020 (grill_bmDuo) | Manual | 0.747 | 0.302 | 0.684 | 0.571 | |
| Best @TREC2020 (h2oloo_RUN4) | Automatic | 0.633 | 0.302 | 0.593 | 0.526 | |
| Best @TREC2020 w/ canonical results (h2oloo_RUN2) | Automatic | 0.705 | 0.326 | 0.621 | 0.575 |