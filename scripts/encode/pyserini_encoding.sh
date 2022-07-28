#!/bin/bash

python -m treccast.encoder.pyserini_encoder -c \
 -o data/embeddings/trec-cast-embeddings.hdf5 \
 --trecweb /data/collections/trec-cast/TREC_Washington_Post_collection.v4.trecweb \
 --encoder castorini/ance-msmarco-passage 