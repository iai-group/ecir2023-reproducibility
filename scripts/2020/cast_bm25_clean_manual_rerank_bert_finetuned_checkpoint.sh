#!/bin/bash
python -m treccast.main --retrieval \
    --topics data/topics/2020/2020_manual_evaluation_topics_v1.0.json \
    --output data/runs/2020/bm25_clean_manual_rerank_bert_finetuned_checkpoint.trec \
    --reranker bert_finetuned \
    --es_index ms_marco_trec_car_clean \
    --utterance_type manual
