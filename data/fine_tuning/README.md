# Reranker - fine-tuning

## Generating fine-tuning data

All the preprocessed data collections used for fine-tuning and the models are stored in `$DATA` folder on gustav1 (`$DATA` refers to `gustav1:/data/scratch/trec-cast/data`).

### TREC-CAsT 2019-2020 data

  * TBD.
  * `$DATA/fine_tuning/trec-cast/Y1Y2_manual_qrels.tsv`.
  * `$DATA/fine_tuning/trec-cast/Y1Y2_manual_or_raw_qrels.tsv`.

### Wizard of Wikipedia data

  * Download the data from http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz.
  * Extract it under `data/fine_tuning/wizard_of_wikipedia/`.
  * Then run `python -m treccast.core.util.finetuning.wiz_of_wiki_parse.py`.
  * It should generate files `data/fine_tuning/wizard_of_wikipedia/wow_finetune_[train|val|test].tsv`.
  * You can skip the above steps and use the generated files that are stored under `$DATA/fine_tuning/wizard_of_wikipedia/`.

## Fine-tuning the reranking models

  * Modify the script in [scripts/finetune/finetune_bert.sh](../../scripts/finetune/finetune_bert.sh) to change parameters:
    - `--lr` for learning rate.
    - `--dropout` for droupout rate.
    - `--val_metric` Validation metric type to be used for early termination.
      - `val_RetrievalNormalizedDCG` for NDCG.
      - `val_RetrievalMRR` for MRR.
      - `val_RetrievalMAP` for MAP.
    - `--val_patience` for number of epochs to wait before early stopping.
    - `--use_wow` to use WoW data.
  * Run the [scripts/finetune/finetune_bert.sh](../../scripts/finetune/finetune_bert.sh) script (Note: this should be run on a GPU server, like gorina6).
  * The best checkpoint according to the specified `--val_metric` should be stored under the model lightning version folder (can be modified using `--save_dir`) for example, `data/models/fine_tuned_models/lightning_logs/version_107/checkpoints/epoch=0-step=1329.ckpt`.

## Generate run file using the fine-tuned model

  * Modify the config file to specify `finetuned_checkpoint_path` as the checkpoint from previous step.
    - For pre fine-tuned models on gustav1 check [data/models/](../models).
  * Set the `reranker` as `bert_finetuned`.
  * Set the other parameters according to your requriements.
  * Run `python -m treccast.main -C config/bert_finetune.yaml` (Note: this should be run on gustav1).

# Rewriter - fine-tuning

## Generating fine-tuning data

All the preprocessed data collections used for fine-tuning and the models are stored in `$DATA` folder on gustav1 (`$DATA` refers to `gustav1:/data/scratch/trec-cast/data`).

### QReCC

  * The dataset is located under `$DATA/fine-tuning/qrecc`.
  * It is split into two files: `train.json` and `test.json`.

### CANARD 

  * The dataset is located under `$DATA/fine-tuning/canard`.
  * It is split into three files: `train.json`, `dev.json` and `test.json`.

## Fine-tuning the rewriter models

  * Run [treccast/rewriter/simpletransformers_rewriter_finetuning.py](../../treccast/rewriter/simpletransformers_rewriter_finetuning.py) file with chosen arguments:
    - `base_model_name` - the name of the base model to be used for fine-tuning, defaults to `t5`.
    - `model_type` - specific model type to be used for fine-tuning, defaults to `t5-base`.
    - `dataset` - the path to the dataset to be used for fine-tuning, defaults to `data/fine_tuning/rewriter/qrecc/`.
    - `model_dir` - the output directory for the fine-tuned model, defaults to `data/fine_tuning/rewriter/qrecc/T5-QReCC_st/`.
  * Exemplary configuration for qrecc with default arguments:
`CUDA_VISIBLE_DEVICES=0 python -m treccast.rewriter.simpletransformers_rewriter_finetuning`
  * Fine-tuned models are stored under `$DATA/models/fine_tuned_models/rewriter/qrecc/`:
    - `T5_QReCC_WaterlooClarke-full` - T5 fine-tuned with the QReCC training dataset, using original test partition of the QReCC dataset as a validation set (implementation based on Simple Transformers).
    - `T5_QReCC_WaterlooClarke-train` - T5 fine-tuned on QReCC collection split to train/validation/test partitions (implementation based on Simple Transformers).

## Generate file with queries rewritten with fine-tuned model

  * Run [treccast/rewriter/t5_rewriter.py](../../treccast/rewriter/t5_rewriter.py) file with chosen arguments:
    - `model_name` - the path to the fine-tuned model to be loaded and used for rewriting, defaults to `data/fine_tuning/rewriter/qrecc/T5-QReCC`.
    - `year` - year for which the rewrites should be generated, defaults to `2020`.
    - `max_length` - max sequence length for the model, defaults to `128`.
    - `output_dir` - the path to the output directory for generated query rewrites `data/rewrites/2020/12_T5_QReCC.tsv`.
  * Exemplary configuration for generating rewrites using model fine-tuned with QReCC for topics from 2021:
`python -m treccast.rewriter.t5_rewriter --year 2021 --output_dir data/rewrites/2021/12_T5_QReCC.tsv`