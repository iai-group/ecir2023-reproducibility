# Query rewrites

All rewrites are stored in TSV files located under `data/rewrites/{2020|2021}`.

The performance of the selected rewriting methods in the retrieval-reranking pipeline is reported in [data/runs/2020/README.md](../runs/2020/README.md) and [data/runs/2021/README.md](../runs/2021/README.md).

## Generating query rewrites

You can generate query rewrites with chosen model as described in [data/fine_tuning/README.md](../fine_tuning/README.md). 

## Query rewritten with fine-tuned models

The following files contain rewrites generated with T5 model fine-tuned using different datasets:
  * `12_T5_QReCC.tsv` - T5 fine-tuned using QReCC (implementation based on Simple Transformers), all the previous rewritten utterances and the canonical response for the last utterance are used as context
  * `13_T5_CANARD.tsv` - T5 fine-tuned using CANARD (`castorini/t5-base-canard` HuggingFace model used), all the previous rewritten utterances and the canonical response for the last utterance are used as context
    - Rewrites for 2020 were generated with:
    `python -m treccast.rewriter.t5_rewriter --model_dir castorini/t5-base-canard --output_dir data/rewrites/2020/13_T5_CANARD.tsv --year 2020 --separator "|||"`
    - Rewrites for 2021 were generated with:
    `python -m treccast.rewriter.t5_rewriter --model_dir castorini/t5-base-canard --output_dir data/rewrites/2021/13_T5_CANARD.tsv --separator "|||"`
  * `13_T5_CANARD_org.tsv` - T5 fine-tuned using CANARD (`castorini/t5-base-canard` HuggingFace model used), all previous turn queries and the three previous canonical passage responses as context
    - Rewrites for 2020 were generated with:
    `python -m treccast.rewriter.t5_rewriter --model_dir castorini/t5-base-canard --output_dir data/rewrites/2020/13_T5_CANARD_org.tsv --year 2020 --separator "|||" --use_canonical_response 3`
    - Rewrites for 2021 were generated with:
    `python -m treccast.rewriter.t5_rewriter --model_dir castorini/t5-base-canard --output_dir data/rewrites/2021/13_T5_CANARD_org.tsv --separator "|||" --use_canonical_response 3`

