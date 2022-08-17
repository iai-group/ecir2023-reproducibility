# Query rewrites

The files `[1..11].tsv` are taken from the [GitHub repository](https://github.com/svakulenk0/cast_evaluation) accompanying the paper "Vakulenko et al. A Comparison of Question Rewriting Methods for Conversational Passage Retrieval. ECIR'21" [[PDF](https://arxiv.org/pdf/2101.07382.pdf)].

The best performing rewriting methods:

| *Method* | *Query rewriting* | *recall@1000* | *MAP* | *MRR* | *NDCG* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- | -- |
| Initial (BM25) | 5_QuReTeC_QnA | 0.5312 | 0.0927 | 0.2621 | 0.3109 | 0.1703 |
| Reranking (BM25+BERT) | 5_QuReTeC_QnA | 0.5312 | 0.2084 | 0.4888 | 0.4233 | 0.3540 |
| Initial (BM25) | 7_Self_learn_Q_QuReTeC_QnA | 0.5667 | 0.0940 | 0.2567 | 0.3257 | 0.1669 |
| Reranking (BM25+BERT) | 7_Self_learn_Q_QuReTeC_QnA | 0.5667 | 0.2220 | 0.5128 | 0.4457 | 0.3625 |
| Initial (BM25) | 11_Human | 0.7070 | 0.1439 | 0.3777 | 0.4232 | 0.2431 |
| Reranking (BM25+BERT) | 11_Human | 0.7070 | 0.3269 | 0.6912 | 0.5830 | 0.5116 |

# Generating query rewrites

You can generate query rewrites with chosen model as described in [data/fine_tuning/README.md](../fine_tuning/README.md). 

# Query rewritten with fine-tuned models

The following files contain rewrites generated with T5 model fine-tuned using different datasets:
  * `12_T5_QReCC.tsv` - T5 fine-tuned using QReCC (implementation based on Simple Transformers).
  * `13_T5_CANARD.tsv` - T5 fine-tuned using CANARD (`castorini/t5-base-canard` HuggingFace model used)
    - Rewrites for 2020 were generated with:
    `python -m treccast.rewriter.t5_rewriter --model_dir castorini/t5-base-canard --output_dir data/rewrites/2020/13_T5_CANARD.tsv --year 2020 --separator "|||" --index_name ms_marco_trec_car_clean`
    - Rewrites for 2021 were generated with:
    `python -m treccast.rewriter.t5_rewriter --model_dir castorini/t5-base-canard --output_dir data/rewrites/2021/13_T5_CANARD.tsv --separator "|||"`
    - Rewrites for 2022 were generated with:
    `python -m treccast.rewriter.t5_rewriter --model_dir castorini/t5-base-canard --output_dir data/rewrites/2022/13_T5_CANARD.tsv --year 2022 --separator "|||" --index_name ms_marco_v2_kilt_wapo`

They are all located under `data/rewrites/2020` and `data/rewrites/2021`.

Performance of the rewriting methods in the whole retrieval-reranking pipeline is reported in [data/runs/2020/README.md](../runs/2020/README.md) and [data/runs/2021/README.md](../runs/2021/README.md).