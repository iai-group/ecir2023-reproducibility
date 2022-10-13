# Query rewrites

All rewrites are stored in TSV files located under `data/rewrites/{2020|2021}`.

The performance of the selected rewriting methods in the retrieval-reranking pipeline is reported in [data/runs/2020/README.md](../runs/2020/README.md) and [data/runs/2021/README.md](../runs/2021/README.md).

## Generating query rewrites

You can generate query rewrites with chosen model as described in [data/fine_tuning/README.md](../fine_tuning/README.md). 

## Query rewritten with fine-tuned models

The following files contain rewrites generated with T5 model fine-tuned using different datasets:
  * `12_T5_QReCC.tsv` - T5 fine-tuned using QReCC (implementation based on Simple Transformers).
  * `13_T5_CANARD.tsv` - T5 fine-tuned using CANARD (`castorini/t5-base-canard` HuggingFace model used)
    - Rewrites for 2020 were generated with:
    `python -m treccast.rewriter.t5_rewriter --model_dir castorini/t5-base-canard --output_dir data/rewrites/2020/13_T5_CANARD.tsv --year 2020 --separator "|||" --index_name ms_marco_trec_car_clean`
    - Rewrites for 2021 were generated with:
    `python -m treccast.rewriter.t5_rewriter --model_dir castorini/t5-base-canard --output_dir data/rewrites/2021/13_T5_CANARD.tsv --separator "|||"`

