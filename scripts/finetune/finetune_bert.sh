 #!/bin/bash
# python -m treccast.reranker.train.bert_reranker_train --gpus 1 2 6 7 \
#         --accelerator dp  --warmup_steps 1000 --max_epochs 20 --save_top_k=3 \
#         --lr 3e-05 --val_patience 5 --val_metric val_RetrievalNormalizedDCG \
#         --dropout 0.3
CUDA_VISIBLE_DEVICES=1,2,6,7 python -m treccast.reranker.train.simpletransformers_reranker_trainer \
        --dataset treccast --base_bert_type bert --bert_model_path nboost/pt-bert-base-uncased-msmarco
CUDA_VISIBLE_DEVICES=1,2,6,7 python -m treccast.reranker.train.simpletransformers_reranker_trainer \
        --dataset wow --base_bert_type bert --bert_model_path nboost/pt-bert-base-uncased-msmarco
CUDA_VISIBLE_DEVICES=1,2,6,7 python -m treccast.reranker.train.simpletransformers_reranker_trainer \
        --dataset both --base_bert_type bert --bert_model_path nboost/pt-bert-base-uncased-msmarco
CUDA_VISIBLE_DEVICES=1,2,6,7 python -m treccast.reranker.train.simpletransformers_reranker_trainer \
        --dataset treccast --base_bert_type roberta --bert_model_path roberta-base
CUDA_VISIBLE_DEVICES=1,2,6,7 python -m treccast.reranker.train.simpletransformers_reranker_trainer \
        --dataset wow --base_bert_type roberta --bert_model_path roberta-base
CUDA_VISIBLE_DEVICES=1,2,6,7 python -m treccast.reranker.train.simpletransformers_reranker_trainer \
        --dataset both --base_bert_type roberta --bert_model_path roberta-base
CUDA_VISIBLE_DEVICES=0,3,4 python -m treccast.reranker.train.simpletransformers_reranker_trainer \
        --dataset both --base_bert_type bert --bert_model_path data/models/BERT_Large_trained_on_MSMARCO
CUDA_VISIBLE_DEVICES=1,2,6,7 python -m treccast.reranker.train.simpletransformers_reranker_trainer \
        --dataset both --base_bert_type roberta --bert_model_path roberta-base