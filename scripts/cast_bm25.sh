#!/bash/bin
python -m treccast.main --retrieval \
    --topics data/topics-2020/automatic_evaluation_topics_annotated_v1.1.json \
    --output data/runs-2020/bm25.trec