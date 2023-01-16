# Runs 

## Evaluation

Evaluation scores are generated using trec_eval (note that relevance threshold is >=2 for binary measures!).

For 2021:
  * To compute all trec_eval metrics:
        `{TREC_EVAL_PATH}/trec_eval -q -c -m all_trec -l2 -M500 data/qrels/{YEAR}.txt data/runs/{YEAR}/{RUNID}.trec`
  * To compute only Recall@1000, MAP, MRR, NDCG, and NDCG@3: 
        `{TREC_EVAL_PATH}/trec_eval trec_eval -q -c -m map -m P.1,3 -m ndcg_cut.1,3,5 -m recip_rank -m all_trec -l2 -M500 data/qrels/{YEAR}.txt data/runs/{YEAR}/{RUNID}.trec` 

For 2020:
  * To compute all trec_eval metrics:
        `{TREC_EVAL_PATH}/trec_eval -q -c -m all_trec -l2 -M1000 data/qrels/{YEAR}.txt data/runs/{YEAR}/{RUNID}.trec`
  * To compute only Recall@1000, MAP, MRR, NDCG, and NDCG@3: 
        `{TREC_EVAL_PATH}/trec_eval trec_eval -q -c -m map -m P.1,3 -m ndcg_cut.1,3,5 -m recip_rank -m all_trec -l2 -M1000 data/qrels/{YEAR}.txt data/runs/{YEAR}/{RUNID}.trec` 

It assumes that [trec_eval](https://github.com/usnistgov/trec_eval) is installed locally under `{TREC_EVAL_PATH}`.