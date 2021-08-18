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
