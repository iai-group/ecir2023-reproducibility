#!/bin/bash
for id in {0..791}
do
  python -m treccast.indexer.indexer -i ms_marco_v2 \
   --host localhost:9204 \
   --trecweb /local/scratch/trec-cast/collections/trecweb_data/ms_marco/trecweb/MARCO_${id}.trecweb
done

