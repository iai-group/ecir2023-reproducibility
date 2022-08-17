#!/bin/bash

year=2019

for b in $(seq 0.3 0.1 0.9)
do
    for k1 in $(seq 0.5 0.1 2.0)
    do
        python -m treccast.main --year $year --output_name bm25_tuning/year_$year-b_$b-k1_$k1 --query_rewrite manual --es.host_name localhost:9204 --es.k1 $k1 --es.b $b
        tools/trec_eval/trec_eval -m all_trec data/qrels/$year.txt data/runs/$year/bm25_tuning/year_$year-b_$b-k1_$k1.trec >> data/fine_tuning/bm25/year_$year-b_$b-k1_$k1.csv
    done
done
python -m treccast.retriever.bm25_tuning
