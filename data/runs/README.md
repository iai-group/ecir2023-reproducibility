# Runs 

## Evaluation

Evaluation scores are generated using trec_eval (note that relevance threshold is >=2 for binary measures!):
  * To compute all trec_eval metrics:
        `{TREC_EVAL_PATH}/trec_eval -m all_trec -l2 data/qrels/{YEAR}.txt data/runs/{YEAR}/{RUNID}.trec`
  * To compute only Recall@1000, MAP, MRR, NDCG, and NDCG@3: 
        `{TREC_EVAL_PATH}/trec_eval -m recall.1000 -m map -m recip_rank -m ndcg -m ndcg_cut.3 -l2 data/qrels/{YEAR}.txt data/runs/{YEAR}/{RUNID}.trec` 

It assumes that [trec_eval](https://github.com/usnistgov/trec_eval) is installed locally under `{TREC_EVAL_PATH}`.