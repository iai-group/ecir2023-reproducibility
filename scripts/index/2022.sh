#!/bin/bash
python -m treccast.indexer.indexer -i ms_marco_v2_kilt_wapo -r --host \
 gustav1.ux.uis.no:9204 --trecweb \
 /data/collections/trec-cast/kilt_knowledgesource.trecweb \
 /data/collections/trec-cast/TREC_Washington_Post_collection.v4.trecweb

 for id in {0..791}
 do
    python -m treccast.indexer.indexer -i ms_marco_v2_kilt_wapo --host \
     gustav1.ux.uis.no:9204 --trecweb \
     /data/collections/trec-cast/ms_marco/trecweb/MARCO_${id}.trecweb
 done