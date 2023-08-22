# Reproducibility Study on TREC CAsT

This repository provides resources developed within the following article [[PDF](https://arxiv.org/abs/2301.10493)]:

> W. Lajewska and K. Balog. **From Baseline to Top Performer: A Reproducibility Study of Approaches at the TREC 2021 Conversational Assistance Track**. In: Advances in Information Retrieval, 45th European Conference on Information Retrieval (ECIR'23). Springer. Dublin, Ireland. April 2023. [10.1007/978-3-031-28241-6_12](https://doi.org/10.1007/978-3-031-28241-6_12) 

## Summary

This paper reports on an effort of reproducing the organizers’ baseline as well as the top performing participant submission at the 2021 edition of the TREC Conversational Assistance track. TREC systems are commonly regarded as reference points for effectiveness comparison. Yet, the papers accompanying them have less strict requirements than peer-reviewed publications, which can make reproducibility challenging. Our results indicate that key practical information is indeed missing. While the results can be reproduced within a 19% relative margin with respect to the main evaluation measure, the relative difference between the baseline and the top performing approach shrinks from the reported 18% to 5%. Additionally, we report on a new set of experiments aimed at understanding the impact of various pipeline components. We show that end-to-end system performance can indeed benefit from advanced retrieval techniques in either stage of a two-stage retrieval pipeline. We also measure the impact of the dataset used for fine-tuning the query rewriter and find that employing different query rewriting methods in different stages of the retrieval pipeline might be beneficial. Moreover, these results are shown to generalize across the 2020 and 2021 editions of the track. We conclude our study with a list of lessons learned and practical suggestions.

## Architecture

Our system follows a conventional conversational passage retrieval pipeline (see, e.g., [Chatty Goose](https://dl.acm.org/doi/10.1145/3404835.3462782)) consisting of the following components:

  * [Indexer](treccast/indexer): Indexing for sparse retrieval (inverted index with ElasticSearch) and for dense retrieval (ANN index with Pyterrier)
  * [Rewriter](treccast/rewriter) (optional): Query rewriter &#8594; T5 (fine-tuned on CANARD or QReCC)
  * [Retriever](treccast/retriever): First-pass retrieval &#8594; BM25 (sparse), ANCE (dense)
  * [Expander](treccast/expander) (optional): Query expansion by pseudo-relevance-feedback &#8594; RM3
  * [Reranker](treccast/reranker) (optional): Neural reranker &#8594; T5, DuoT5

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
  - Run the [2020 script](scripts/index/2020.sh) and/or [2021 script](scripts/index/2021.sh) (make sure that the paths to the downloaded collections are correct in the scripts and in the [indexer](treccast/indexer/indexer.py))

To create ANN indices used in ANCE dense retrieval:
  - Download the collections using the links provided [here](data/README.md) and place them in [data/collections/](data/collections/)
  - Run the [2020 script](scripts/ance/2020.sh) and/or [2021 script](scripts/ance/2021.sh) (make sure that the paths to the downloaded collections are correct in the scripts and in the [indexer](treccast/retriever/ance_dense_retriever.py))


## Models

You can use the provided code to train your own models or you can use the provided model checkopoints to repeat our experiments. The models can be downloaded from the [shared server folder](https://gustav1.ux.uis.no/downloads/ecir2023-reproducibility/). More information about commands used for models fine-tuning can be found [here](data/fine_tuning/README.md).


## Running the conversational search system

To set the desired parameters create a config file as described [here](config/README.md) then run:

```
python -m treccast.main --config <path_to_config> --year <2020|2021>
```


Alternatively, use command line arguments. For help issue command:

```
python -m treccast.main -h
```

The configuration files used for the runs presented in the paper can be found [here](data/runs/2020/) for 2020 and [here](data/runs/2021/) for 2021.


## Reproducibility experiments

In order to reproduce the results reported in the paper, run the conversational search system using the config files linked in the tables below. The results in row **WaterlooClarke@TREC'21 (runfile)** were generated using the official [runfile](https://trec.nist.gov/results/trec30/cast.primary.input.html) provided by the organizers. We remove from it the passage IDs and deduplicate the rankings using [this script](treccast/core/util/ranking_deduplication.py). The results in row **BaselineOrganizers@TREC'21 (runfile)** were generated using the official [runfile](https://github.com/daltonj/treccastweb/blob/master/2021/baselines/document_runs/org_automatic_results_1000.v1.0.run) provided by the organizers. This is the version of the run converted from passage to document IDs and deduplicated. This is the copy of the original runfile as of Jan 6th, 2023 that can be found [here](data/runs/2021/baseline_organizers_runfile.trec).

Evaluation results are computed using the official [trec_eval](https://github.com/usnistgov/trec_eval) tool using the parameters described [here](data/runs/README.md).

### Reproducibility experiments on the TREC CAsT’21 dataset

| Approach  | R@500 | MAP@500 | MRR | NDCG  | NDCG@3 |
|----------------------------------------------------------|----------------|-----------------|--------|--------|--------|
| [Raw + BM25 + monoT5](data/runs/2021/raw_bm25_mono-t5_2021.meta.yaml)  | 0.3467  | 0.1216 | 0.2875 | 0.2593 | 0.2051 |
| [TREC-Auto + BM25 + monoT5](data/runs/2021/automatic_bm25_mono-t5_2021.meta.yaml)| 0.6292  | 0.2686 | 0.5582 | 0.4831 | 0.3999 |
| BaselineOrganizers@TREC'21 (in TREC CAsT'21 overview) | 0.636 | 0.291  | 0.607  | 0.504  | 0.436  |
| BaselineOrganizers@TREC'21 (runfile) | 0.6228 | 0.2818 | 0.5966 | 0.4934 | 0.4242 |
| [BaselineOrganizers-QR-BM25](data/runs/2021/t5-canard-org_bm25_mono-t5_2021.meta.yaml) | 0.5632  | 0.2268 | 0.4947 | 0.4317 | 0.3457 |
| [BaselineOrganizers-BM25](data/runs/2021/t5-canard_bm25-b-446-k-82_mono-t5_2021.meta.yaml)  | 0.5894  | 0.2546 | 0.5405 | 0.4672 | 0.3966 |
| [BaselineOrganizers](data/runs/2021/t5-canard_bm25_mono-t5_2021.meta.yaml) | 0.6472  | 0.2628 | 0.5354 | 0.4885 | 0.3968 |
| WaterlooClarke@TREC'21 (in TREC CAsT'21 overview)  | 0.869 | 0.362  | 0.684  | 0.640  | 0.514  |
| WaterlooClarke@TREC'21 (runfile)| 0.8534  | 0.3494 | 0.6626 | 0.6240 | 0.4950 |
| [WaterlooClarke reproduced by us](data/runs/2021/ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_mono-duo-t5_2021.meta.yaml)  | 0.6915  | 0.2864 | 0.5712 | 0.5176 | 0.4151 |

## Additional experiments

We have reproduced two approaches, BaselineOrganizers and WaterlooClarke, which follow the same basic two-stage retrieval pipeline, but differ in each of the query rewriting, first-pass retrieval, and re-ranking components. We experiment with different configurations of this basic pipeline to understand which changes contribute most to overall performance. Additionally, we consider a different pipeline architecture. In both sets of experiments, we are interested in the generalizability of findings, therefore we also report results on the TREC CAsT’20 dataset. (Note that the rank cut-off for 2020 collection is 1000, while for 2021 it is 500.)

Evaluation results are computed using the official [trec_eval](https://github.com/usnistgov/trec_eval) tool using the parameters described [here](data/runs/README.md).

### Variants of a two-stage retrieval pipeline on TREC CAsT’20

| Approach  | R@1000 | MAP@1000 | MRR | NDCG  | NDCG@3 |
|----------------------------------------------------------|----------------|-----------------|--------|--------|--------|
| [T5\_CANARD + BM25 + monoT5](data/runs/2020/t5-canard_bm25_mono-t5_2020.meta.yaml)              | 0.5276              | 0.2191           | 0.5457          | 0.4353          | 0.3789          |
| [T5\_QReCC + BM25 + monoT5](data/runs/2020/t5-qrecc_bm25_mono-t5_2020.meta.yaml)               | 0.5100              | 0.2056           | 0.5106          | 0.4065          | 0.3618          |
| [T5\_CANARD + ANCE/BM25 + mono/duoT5](data/runs/2020/ance/t5-canard_ance_bm25-b-45-k-95_mono-duo-t5_2020.meta.yaml)     | 0.6781              | 0.2540           | 0.5512          | 0.5027          | 0.4052          |
| [T5\_QReCC + ANCE/BM25 + mono/duoT5](data/runs/2020/ance/t5-qrecc_ance_bm25-b-45-k-95_mono-duo-t5_2020.meta.yaml)      | 0.6449              | 0.2443           | 0.5357          | 0.4804          | 0.4061          |
| [T5\_CANARD + ANCE/BM25/PRF + mono/duoT5](data/runs/2020/ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_mono-duo-t5_2020.meta.yaml) | **0.6878**    | **0.2555** | **0.5541**| **0.5063**| **0.4086**|
| [T5\_QReCC + ANCE/BM25/PRF + mono/duoT5](data/runs/2020/ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_mono-duo-t5_2020.meta.yaml)  | 0.6608              | 0.2451           | 0.5355          | 0.4840          | 0.4052          |

### Variants of a two-stage retrieval pipeline on TREC CAsT’21

| Approach  | R@500 | MAP@500 | MRR | NDCG  | NDCG@3 |
|----------------------------------------------------------|----------------|-----------------|--------|--------|--------|
| [T5\_CANARD + BM25 + monoT5](data/runs/2021/t5-canard_bm25_mono-t5_2021.meta.yaml)              | 0.6472              | 0.2628           | 0.5354          | 0.4885          | 0.3968          |
| [T5\_QReCC + BM25 + monoT5](data/runs/2021/t5-qrecc_bm25_mono-t5_2021.meta.yaml)               | 0.6018              | 0.2530           | 0.5369          | 0.4670          | 0.3933          |
| [T5\_CANARD + ANCE/BM25 + mono/duoT5](data/runs/2021/ance/t5-canard_ance_bm25-b-45-k-95_mono-duo-t5_2021.meta.yaml)     | 0.7259              | 0.2886           | 0.5575          | 0.5316          | 0.4068          |
| [T5\_QReCC + ANCE/BM25 + mono/duoT5](data/runs/2021/ance/t5-qrecc_ance_bm25-b-45-k-95_mono-duo-t5_2021.meta.yaml)      | 0.6799              | 0.2843           | 0.5702          | 0.5135          | **0.4159**|
| [T5\_CANARD + ANCE/BM25/PRF + mono/duoT5](data/runs/2021/ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_mono-duo-t5_2021.meta.yaml) | **0.7306**    | **0.2915** | 0.5573          | **0.5330**| 0.4061          |
| [T5\_QReCC + ANCE/BM25/PRF + mono/duoT5](data/runs/2021/ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_mono-duo-t5_2021.meta.yaml)  | 0.6915              | 0.2864           | **0.5712**| 0.5176          | 0.4151          |

### Performance of query rewriting approaches with different variants of two-stage pipeline on the TREC CAsT’20 

| Approach  | R@1000 | MAP@1000 | MRR | NDCG  | NDCG@3 |
|----------------------------------------------------------|----------------|-----------------|--------|--------|--------|
| [T5\_QReCC + T5\_QReCC](data/runs/2020/ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_mono-duo-t5_2020.meta.yaml)     | 0.6608              | 0.2451                | 0.5355          | 0.4840          | 0.4052          | 0.3846          |
| [T5\_QReCC + T5\_CANARD](data/runs/2020/ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_t5-canard_mono-duo-t5_2020.meta.yaml)    | 0.6608              | 0.2432                | 0.5437          | 0.4842          | **0.4086** | 0.3846          |
| [T5\_CANARD + T5\_QReCC](data/runs/2020/ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_t5-qrecc_mono-duo-t5_2020.meta.yaml)    | **0.6878**     | 0.2385                | 0.5266          | 0.4883          | 0.3923          | 0.3721          |
| [T5\_CANARD + T5\_CANARD](data/runs/2020/ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_mono-duo-t5_2020.meta.yaml)   | **0.6878**     | **0.2555**       | **0.5541** | **0.5063** | **0.4086** | **0.3891** |

### Performance of query rewriting approaches with different variants of two-stage pipeline on the TREC CAsT’21

| Approach  | R@500 | MAP@500 | MRR | NDCG  | NDCG@3 |
|----------------------------------------------------------|----------------|-----------------|--------|--------|--------|
| [T5\_QReCC + T5\_QReCC](data/runs/2021/ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_mono-duo-t5_2021.meta.yaml)     | 0.6915              | 0.2864                | 0.5712          | 0.5176           | 0.4151          | 0.4103          |
| [T5\_QReCC + T5\_CANARD](data/runs/2021/ance/prf-17-26_t5-qrecc_ance_bm25-b-45-k-95_t5-canard_mono-duo-t5_2021.meta.yaml)    | 0.6879              | **0.2940**       | 0.5697          | 0.5238          | **0.4176** | 0.4149          |
| [T5\_CANARD + T5\_QReCC](data/runs/2021/ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_t5-qrecc_mono-duo-t5_2021.meta.yaml)    | 0.7267              | 0.2883                | **0.5781** | **0.5331** | 0.4166          | **0.4174** |
| [T5\_CANARD + T5\_CANARD](data/runs/2021/ance/prf-17-26_t5-canard_ance_bm25-b-45-k-95_mono-duo-t5_2021.meta.yaml)   | **0.7306**     | 0.2916                | 0.5573          | 0.5330          | 0.4061          | 0.4089          |


## Acknowledgments

We thank IAI research group for the codebase this work builds on. This research was supported by the Norwegian Research Center for AI Innovation, NorwAI (Research Council of Norway, project number 309834). We thank the members of the WaterlooClarke group (School of Computer Science, University of Waterloo, Canada), Xinyi Yan and  Charlie Clarke for supporting our efforts to reproduce [their TREC CAsT’21 submission](https://trec.nist.gov/pubs/trec28/papers/WaterlooClarke). We also thank the TREC CAsT organizers for their efforts in coordinating the track and for providing us with additional technical details regarding their baseline.


## Citation

If you use the resources presented in this repository, please cite:

```
@inproceedings{Lajewska:2023:ECIR,
  author =    {Weronika Łajewska and Krisztian Balog},
  title =     {From Baseline to Top Performer: A Reproducibility Study of Approaches at the TREC 2021 Conversational Assistance Track},
  booktitle = {European Conference on Information Retrieval},
  series =    {ECIR '23},
  pages =     {177--191}
  year =      {2023},
  doi =       {10.1007/978-3-031-28241-6_12},
  publisher = {Springer}
}
```


## Contact

Should you have any questions, please contact `Weronika Łajewska` at `weronika.lajewska`[AT]uis.no (with [AT] replaced by @).
