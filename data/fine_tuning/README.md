
# Generating fine-tuning data

## Trec-CaST Y1-Y2 data
    - TBD
    - fine_tuning/trec-cast/Y1Y2_manual_qrels.tsv
    - fine_tuning/trec-cast/Y1Y2_manual_or_raw_qrels.tsv
## Wizard of Wikipedia data

- Download the data from [here](http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz)
- Extract it under `data/fine_tuning/wizard_of_wikipedia/`
- Then run `python -m treccast.core.util.finetuning.wiz_of_wiki_parse.py`
- It should generate files `data/fine_tuning/wizard_of_wikipedia/wow_finetune_[train|val|test].tsv`

# Fine tuning the models

- Modify the script in scripts/finetune/finetune_bert.sh to change parameters
    - `--lr` for learning rate
    - `--dropout` for droupout rate
    - `--val_metric` Validation metric type to be used for early termination
        - val_RetrievalNormalizedDCG for NDCG
        - val_RetrievalMRR for MRR
        - val_RetrievalMAP for MAP
    - `--val_patience` for number of epochs to wait before early stopping
    - `--use_wow` to use WoW data
- Run the script `scripts/finetune/finetune_bert.sh` (Note this should be run on gorina6 or a gpu server)
- The best checkpoint according to the specified `--val_metric` should be stored under the model lightning version folder (can be modified using `--save_dir`) for example, `data/models/fine_tuned_models/lightning_logs/version_107/checkpoints/epoch=0-step=1329.ckpt`

# Generate run file using the fine-tuned model

- Modify the config file to specify `finetuned_checkpoint_path` as the checkpoint from previous step.
    - For pre fine-tuned models on gustav1 see [here](https://github.com/iai-group/trec-cast-2021/tree/main/data/models)
- Set the `reranker` as `bert_finetuned`
- Set the other parameters according to your requriements 
- Run `python -m treccas.main -C config/bert_finetune.yaml` (Note this should be run on gustav1)
