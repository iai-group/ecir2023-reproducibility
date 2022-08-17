#!/bin/bash
python -m treccast.indexer.indexer -i ms_marco_wapo_v2_trec_car -r \
 --host localhost:9204 -m \
 /local/scratch/trec-cast/collections/trecweb_data/collection.tar.gz \
 -c \
 /local/scratch/trec-cast/collections/trecweb_data/dedup.articles-paragraphs.cbor

 for id in {0..5}
 do
    python -m treccast.indexer.indexer -i ms_marco_wapo_v2_trec_car --host \
     localhost:9204 --trecweb \
     /local/scratch/trec-cast/collections/trecweb_data/trecweb/WaPo_${id}.trecweb
 done
