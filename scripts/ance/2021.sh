#!/bin/bash
python -m treccast.ance_dense_retriever \
    --index_path data/retrieval/ance/trecweb_ms_marco_kilt_wapo_ance \
    --year 2021 \
    --reset_index \
    --es_host_name localhost:9204 \
    --es_index_name ms_marco_kilt_wapo_clean \
    --k 1000 \
    --collections /data/collections/