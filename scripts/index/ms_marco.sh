#!/bin/bash
python -m treccast.indexer.indexer -i ms_marco -r \
 --trecweb /local/scratch/trec-cast/collections/trecweb_data/msmarco-docs.trecweb \
 --host localhost:9204

