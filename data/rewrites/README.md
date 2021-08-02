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
