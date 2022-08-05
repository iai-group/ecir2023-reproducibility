# TREC CAsT 2021

![CI build](https://github.com/iai-group/trec-cast-2021/actions/workflows/python-package-conda.yaml/badge.svg)
[![codecov](https://codecov.io/gh/iai-group/trec-cast/branch/main/graph/badge.svg?token=4EZNRUV7B7)](https://codecov.io/gh/iai-group/trec-cast)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the IAI group's participation at the TREC Conversational Assistance Track (CAsT).

  * [TREC CAsT website](http://www.treccast.ai/)
  * [2021 guidelines](https://docs.google.com/document/d/1Eo0IqQedYc_rfTw-YxbvUGTpoYSmejU0iDlUzQWj3_w/edit?usp=sharing)


## Architecture

Our system follows a conventional conversational passage retrieval pipeline (see, e.g., [Chatty Goose](https://dl.acm.org/doi/10.1145/3404835.3462782)) consisting of the following components:

  * [Indexer](treccast/indexer): Indexing for first-pass retrieval (ElasticSearch)
  * [Retriever](treccast/retriever): First-pass retrieval (BM25)
  * [Reranker](treccast/reranker): Neural reranker (BERT or T5)
  * [Rewriter](treccast/rewriter): Query rewriter (optional, can be applied to first-pass retrieval and/or reranking)


## Running

To set the desired parameters create a config file as described [here](config/README.md) then run:

```
python -m treccast.main --config <path_to_config> --year <2020|2021>
```


Alternatively, use command line arguments. For help issue command:

```
python -m treccast.main -h
```