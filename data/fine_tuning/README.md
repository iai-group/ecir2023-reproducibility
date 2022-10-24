# Rewriter - fine-tuning

## Generating fine-tuning data

All the preprocessed data collections used for fine-tuning and the models are linked in the [general README.md file](../README.md).

### QReCC

  * The dataset can be downloaded from `https://github.com/apple/ml-qrecc`.
  * It is split into two files: `train.json` and `test.json`.

### CANARD 

  * The dataset can be downloaded form `https://sites.google.com/view/qanta/projects/canard`.
  * It is split into three files: `train.json`, `dev.json` and `test.json`.

## Fine-tuning the rewriter models

  * Run [treccast/rewriter/simpletransformers_rewriter_finetuning.py](../../treccast/rewriter/simpletransformers_rewriter_finetuning.py) file with chosen arguments:
    - `base_model_name` - the name of the base model to be used for fine-tuning, defaults to `t5`.
    - `model_type` - specific model type to be used for fine-tuning, defaults to `t5-base`.
    - `dataset` - the path to the dataset to be used for fine-tuning, defaults to `data/fine_tuning/rewriter/qrecc/`.
    - `model_dir` - the output directory for the fine-tuned model, defaults to `data/fine_tuning/rewriter/qrecc/T5_QReCC_WaterlooClarke-full/`.
  * Exemplary configuration for qrecc with default arguments:
`CUDA_VISIBLE_DEVICES=0 python -m treccast.rewriter.simpletransformers_rewriter_finetuning`
  * Fine-tuned models are stored under `$DATA/models/rewriter/qrecc/`:
    - `T5_QReCC_WaterlooClarke-full` - T5 fine-tuned with the QReCC training dataset, using original test partition of the QReCC dataset as a validation set (implementation based on Simple Transformers).
    - `T5_QReCC_WaterlooClarke-train` - T5 fine-tuned on QReCC collection split to train/validation/test partitions (implementation based on Simple Transformers).

## Generate file with queries rewritten with fine-tuned model

  * Run [treccast/rewriter/t5_rewriter.py](../../treccast/rewriter/t5_rewriter.py) file with chosen arguments:
    - `model_name` - the path to the fine-tuned model to be loaded and used for rewriting, defaults to `data/fine_tuning/rewriter/qrecc/T5_QReCC_WaterlooClarke-full`.
    - `year` - year for which the rewrites should be generated, defaults to `2020`.
    - `max_length` - max sequence length for the model, defaults to `128`.
    - `output_dir` - the path to the output directory for generated query rewrites `data/rewrites/2020/12_T5_QReCC.tsv`.
  * Exemplary configuration for generating rewrites using model fine-tuned with QReCC for topics from 2021:
`python -m treccast.rewriter.t5_rewriter --year 2021 --output_dir data/rewrites/2021/12_T5_QReCC.tsv`