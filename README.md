# TREC CAsT 2021

![CI build](https://github.com/iai-group/trec-cast-2021/actions/workflows/python-package-conda.yaml/badge.svg)
[![codecov](https://codecov.io/gh/iai-group/trec-cast/branch/main/graph/badge.svg?token=4EZNRUV7B7)](https://codecov.io/gh/iai-group/trec-cast)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the IAI group's participation at the TREC Conversational Assistance Track (CAsT).

  * [TREC CAsT website](http://www.treccast.ai/)
  * [2021 guidelines](https://docs.google.com/document/d/1Eo0IqQedYc_rfTw-YxbvUGTpoYSmejU0iDlUzQWj3_w/edit?usp=sharing)


## Architecture

Our system follows a conventional conversational passage retrieval pipeline (see, e.g., [Chatty Goose](https://dl.acm.org/doi/10.1145/3404835.3462782)) consisting of the following components:

  * [Indexer](treccast/indexer): Indexing for sparse retrieval (ElasticSearch), option to expand passages with doc2query
  * [Encoder](treccast/encoder): Encoding for dense retrieval &#8594; Pyserini ANCE encoder
  * [Retriever](treccast/retriever): First-pass retrieval &#8594; BM25 (sparse), ANCE (dense - *in progress*), ScaNN (dense - *in progress*), SPLADE (sparse learned - *to implement*)
  * [Reranker](treccast/reranker) (optional): Neural reranker &#8594; T5, DuoT5, BERT (**not used**)
  * [Rewriter](treccast/rewriter) (optional, can be applied to first-pass retrieval): Query rewriter &#8594; T5 (fine-tuned on CANARD or QReCC)
  * [Expander](treccast/expander) (optional, can be applied to first-pass retrieval): Query expansion by pseudo-relevance-feedback &#8594; RM3


## Running

To set the desired parameters create a config file as described [here](config/README.md) then run:

```
python -m treccast.main --config <path_to_config> --year <2020|2021>
```


Alternatively, use command line arguments. For help issue command:

```
python -m treccast.main -h
```