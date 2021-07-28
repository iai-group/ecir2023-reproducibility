#!/bin/bash
python -m treccast.main --retrieval \
    --topics data/topics-2020/automatic_evaluation_topics_annotated_v1.1.json \
    --output data/runs-2020/bm25_k1_4_46_b_0_82.trec \
    --es.k1 4.46 \
    --es.b 0.82
