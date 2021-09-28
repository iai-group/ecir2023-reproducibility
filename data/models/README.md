
# Models

- $DATA refers to `/data/scratch/trec-cast-2021/data/` on g1.

| *Base model* | *Fine-tuning data* | *Location on g1* | *Hyperparameters* |
| -- | -- | -- | -- | 
| nboost/pt-bert-base-uncased-msmarco | TREC-CaST Y1-Y2 data and WoW data | $DATA/models/fine_tuned_models/simpletransformers_bert_nboost_pt-bert-base-uncased-msmarco_both/best_model | learning_rate: 6e-6, hidden_dropout_prob: 0.3, train_batch_size: 64  | 
| nboost/pt-bert-base-uncased-msmarco | TREC-CaST Y1-Y2 | $DATA/models/fine_tuned_models/simpletransformers_bert_nboost_pt-bert-base-uncased-msmarco_treccast/best_model | learning_rate: 6e-6, hidden_dropout_prob: 0.3, train_batch_size: 64  | 
| nboost/pt-bert-base-uncased-msmarco | WoW | $DATA/models/fine_tuned_models/simpletransformers_bert_nboost_pt-bert-base-uncased-msmarco_wow/best_model | learning_rate: 6e-6, hidden_dropout_prob: 0.3, train_batch_size: 64  | 
| nboost/pt-bert-base-uncased-msmarco | TREC-CaST Y1 and WoW | $DATA/fine_tuned_models/simpletransformers_bert_nboost_pt-bert-base-uncased-msmarco_treccastY1_wow/ | --lr 3e-05 --val_patience 5 --val_metric val_RetrievalMAP --dropout 0.1 | 