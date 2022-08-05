#!/bin/bash
python -m treccast.indexer.indexer -i ms_marco_kilt_wapo_clean -r \
 --host gustav1.ux.uis.no:9204 \
 --trecweb /data/collections/trec-cast/msmarco-docs.trecweb \
 /data/collections/trec-cast/kilt_knowledgesource.trecweb \
 /data/collections/trec-cast/TREC_Washington_Post_collection.v4.trecweb
