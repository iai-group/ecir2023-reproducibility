
# Models

- $DATA refers to `/data/scratch/trec-cast-2021/data/` on g1.

| *Base model* | *Fine-tuning data* | *Location on g1* | *Hyperparameters* |
| -- | -- | -- | -- | 
| nboost/pt-bert-base-uncased-msmarco | TREC-CaST Y1-Y2 data and WoW data | $DATA/models/fine_tuned_models/bert_TRECY1Y2_WoW.ckpt | --lr 3e-05 --val_patience 5 --
val_metric val_RetrievalMAP --dropout 0.1 | 
| nboost/pt-bert-base-uncased-msmarco | TREC-CaST Y1-Y2 | $DATA/models/fine_tuned_models/bert_TRECY1Y2.ckpt | --lr 3e-05 --val_patience 5 --val_metric val_RetrievalMAP --dropout 0.1 | 
| nboost/pt-bert-base-uncased-msmarco | WoW | $DATA/models/fine_tuned_models/bert_WoW.ckpt | --lr 3e-05 --val_patience 5 --val_metric val_RetrievalMAP --dropout 0.1 | 
| nboost/pt-bert-base-uncased-msmarco | TREC-CaST Y1 and WoW | $DATA/fine_tuned_models/simpletransformers_bert_nboost_pt-bert-base-uncased-msmarco_treccastY1_wow/ | --lr 3e-05 --val_patience 5 --val_metric val_RetrievalMAP --dropout 0.1 | 
| nboost/pt-bert-base-uncased-msmarco (without classification head) | TREC-CaST Y2  | $DATA/models/fine_tuned_models/nboost/pt-bert-base-uncased-msmarco/ | --lr 3e-05 --val_patience 5 --val_metric val_RetrievalMAP --dropout 0.1 | 
