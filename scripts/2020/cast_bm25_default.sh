#!/bin/bash
python -m treccast.main --retrieval \
    --topics data/topics/2020/2020_automatic_evaluation_topics_v1.0.json \
    --output data/runs/2020/bm25_default_v1.0.trec \
    --preprocess
