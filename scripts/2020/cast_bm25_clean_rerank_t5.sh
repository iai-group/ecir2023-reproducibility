#!/bin/bash
python -m treccast.main --retrieval \
    --topics data/topics/2020/automatic_evaluation_topics_annotated_v1.1.json \
    --output data/runs/2020/bm25_clean_rerank_t5.trec \
    --reranker t5 --es_index ms_marco_trec_car_clean
