# Reproducibility Study on TREC CAsT

![CI build](https://github.com/iai-group/trec-cast-2021/actions/workflows/python-package-conda.yaml/badge.svg)
[![codecov](https://codecov.io/gh/iai-group/trec-cast/branch/main/graph/badge.svg?token=4EZNRUV7B7)](https://codecov.io/gh/iai-group/trec-cast)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code associated with **Conversational Search in TREC CAsT Setting -
Reproducibility of Best Performing Solutions** paper submitted to ECIR'23. It implements a baseline conversational search system and reproduces the state of the art approach presented at TREC CAsT'21.

## Architecture

Our system follows a conventional conversational passage retrieval pipeline (see, e.g., [Chatty Goose](https://dl.acm.org/doi/10.1145/3404835.3462782)) consisting of the following components:

  * [Indexer](treccast/indexer): Indexing for sparse retrieval (ElasticSearch)
  * [Retriever](treccast/retriever): First-pass retrieval &#8594; BM25 (sparse), ANCE (dense)
  * [Expander](treccast/expander) (optional): Query expansion by pseudo-relevance-feedback &#8594; RM3
  * [Reranker](treccast/reranker) (optional): Neural reranker &#8594; T5, DuoT5
  * [Rewriter](treccast/rewriter) (optional): Query rewriter &#8594; T5 (fine-tuned on CANARD or QReCC)

The detailed documentation is available [here](data/README.md).

## Installation

To install our converational search system and all its dependencies make sure you have anaconda distribution installed in your system and use the following commands:
  - Create and activate a new conda environment using the following commands:
  ```
  conda env update -f environment.yaml
  conda activate treccast
  ```
  - Install ANCE Pyterrier 
  ```
  pip install --upgrade git+https://github.com/WerLaj/pyterrier_ance.git
  ```
  - Download [ANCE model checkpoint](https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource/Passage_ANCE_FirstP_Checkpoint.zip) and place it under [/data/retrieval/ance/](/data/retrieval/ance/)


## Indices

To create the ElasticSearch indices:
  - Make sure that you have an ElasticSearch instance up and running
  - Download the collections using the links provided [here](data/README.md) and place them in [data/collections/](data/collections/)
  - Run the [2020 script](scripts/index/2020.sh) and/or [2021 script](scripts/index/2021.sh) (make sure that the paths to the cownloaded collections are correct in the scripts and in the [indexer](treccast/indexer/indexer.py))

To create ANN indiced used in ANCE dense retrieval:
  - Download the collections using the links provided [here](data/README.md) and place them in [data/collections/](data/collections/)
  - Run the scripts ...


## Models

You can use the provided code to train your own models or you can use the provided model checkopoints to repeat our experiments. The models can be downloaded from Google Drive. More information about commands used for models fine-tuning can be found [here](data/fine_tuning/README.md)


## Running conversational search system

To set the desired parameters create a config file as described [here](config/README.md) then run:

```
python -m treccast.main --config <path_to_config> --year <2020|2021>
```


Alternatively, use command line arguments. For help issue command:

```
python -m treccast.main -h
```

The configuration files used for the runs presented in the paper can be found [here](data/runs/2020/) for 2020 and [here](data/runs/2021/) for 2021.


## Acknowledgments

We thank [IAI research group](https://iai.group/) for the codebase this work builds on. We thank WaterlooClarke group at School of Computer Science, University of Waterloo, Canada for their support in our efforts to reproduce [their approach](https://trec.nist.gov/pubs/trec28/papers/WaterlooClarke.C.pdf) presented at TREC CAsT'21.

