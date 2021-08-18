#!/bin/bash
python -m treccast.main --retrieval \
    --topics data/topics/2021/2021_manual_evaluation_topics_v1.0.json \
    --output data/runs/2021/bm25_manual_body.trec \
    --es_index ms_marco_kilt_wapo_clean \
    --es_field body \
    --utterance_type manual
