#!/bin/bash

python -m treccast.indexer.indexer -i ms_marco_v2_kilt_wapo_new -r --host \
 gustav1.ux.uis.no:9204

 for filename in /data/collections/trec-cast/2022/trecweb/*.trecweb
 do
    python -m treccast.indexer.indexer -i ms_marco_v2_kilt_wapo_new --host \
     gustav1.ux.uis.no:9204 --trecweb \
     /data/collections/trec-cast/2022/trecweb/${filename}.trecweb
 done