# Runs 

## Evaluation

Evaluation scores are generated using trec_eval (note that relevance threshold is >=2 for binary measures!).

For 2021:
  * To compute all trec_eval metrics:
        `{TREC_EVAL_PATH}/trec_eval -c -m all_trec -l2 -M500 data/qrels/{YEAR}.txt data/runs/{YEAR}/{RUNID}.trec`
  * To compute only Recall@1000, MAP, MRR, NDCG, and NDCG@3: 
        `{TREC_EVAL_PATH}/trec_eval trec_eval -c -m map -m ndcg_cut.3 -m ndcg -m recip_rank -m recall.500 -l2 -M500 data/qrels/{YEAR}.txt data/runs/{YEAR}/{RUNID}.trec` 

For 2020:
  * To compute all trec_eval metrics:
        `{TREC_EVAL_PATH}/trec_eval -c -m all_trec -l2 -M1000 data/qrels/{YEAR}.txt data/runs/{YEAR}/{RUNID}.trec`
  * To compute only Recall@1000, MAP, MRR, NDCG, and NDCG@3: 
        `{TREC_EVAL_PATH}/trec_eval -c -m map -m ndcg_cut.3 -m ndcg -m recip_rank -m recall.1000 -l2 -M1000 data/qrels/{YEAR}.txt data/runs/{YEAR}/{RUNID}.trec` 

It assumes that [trec_eval](https://github.com/usnistgov/trec_eval) is installed locally under `{TREC_EVAL_PATH}`.
