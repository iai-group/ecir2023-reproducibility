#!/bin/bash
python -m treccast.ance_dense_retriever \
    --index_path data/retrieval/ance/ms_marco_trec_car_ance \
    --year 2020 \
    --reset_index \
    --es_host_name localhost:9204 \
    --es_index_name ms_marco_trec_car_clean \
    --k 1000 \
    --collections /data/collections/