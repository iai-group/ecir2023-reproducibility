#!/bin/bash
python -m treccast.main --retrieval \
    --topics data/topics/2021/2021_manual_evaluation_topics_v1.0.json \
    --output data/runs/2021/bm25_manual_body_k1_446_b_082.trec \
    --es_index ms_marco_kilt_wapo_clean \
    --es_field body \
    --utterance_type manual \
    --es_k1 4.46 --es_b 0.82
