#!/bin/bash
python -m treccast.indexer.indexer -i ms_marco_trec_car_clean -r \
 --ms_marco --trec_car \
 --host localhost:9204