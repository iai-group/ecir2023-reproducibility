#!/bin/bash
python -m treccast.main --retrieval \
    --topics data/topics/2020/2020_automatic_evaluation_topics_v1.0.json \
    --output data/runs/2020/bm25_clean_automatic_rerank_t5.trec \
    --reranker t5 --es_index ms_marco_trec_car_clean \
    --utterance_type automatic
