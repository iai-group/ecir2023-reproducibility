#!/bin/bash
python -m treccast.main --retrieval \
    --topics data/topics/2021/2021_automatic_evaluation_topics_v1.0.json \
    --output data/runs/2021/bm25_automatic.trec \
    --es_index ms_marco_kilt_wapo_clean \
    --es_field catch_all \
    --utterance_type automatic
