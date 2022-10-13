# TREC CAsT 2021

![CI build](https://github.com/iai-group/trec-cast-2021/actions/workflows/python-package-conda.yaml/badge.svg)
[![codecov](https://codecov.io/gh/iai-group/trec-cast/branch/main/graph/badge.svg?token=4EZNRUV7B7)](https://codecov.io/gh/iai-group/trec-cast)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code associated with ECIR'23 Reproducibility Paper.


## Architecture

Our system follows a conventional conversational passage retrieval pipeline (see, e.g., [Chatty Goose](https://dl.acm.org/doi/10.1145/3404835.3462782)) consisting of the following components:

  * [Indexer](treccast/indexer): Indexing for sparse retrieval (ElasticSearch)
  * [Retriever](treccast/retriever): First-pass retrieval &#8594; BM25 (sparse), ANCE (dense)
  * [Reranker](treccast/reranker) (optional): Neural reranker &#8594; T5, DuoT5
  * [Rewriter](treccast/rewriter) (optional): Query rewriter &#8594; T5 (fine-tuned on CANARD or QReCC)
  * [Expander](treccast/expander) (optional): Query expansion by pseudo-relevance-feedback &#8594; RM3


## Installation

  - Conda environment
  - Pyterrier
  - ANCE repo
  - ANCE model checkpoint

## Running

To set the desired parameters create a config file as described [here](config/README.md) then run:

```
python -m treccast.main --config <path_to_config> --year <2020|2021>
```


Alternatively, use command line arguments. For help issue command:

```
python -m treccast.main -h
```