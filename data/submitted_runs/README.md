# Submitted files


## Run 1 "raft"

  * First-pass: Default BM25 params with stopword removal and stemming using catch-all field (document title+body), using manual query rewrite
  * Reranking: T5 fine-tuned on MS-MARCO, using manual query rewrite
  * TREC form:
    - SUBMISSION CATEGORY: Manually rewritten utterances 
    - CONVERSATIONAL QUERY UNDERSTANDING METHOD:
      - None
    - CONVERSATIONAL QUERY UNDERSTANDING DATA:
      - [x] method uses CAsT Y3 provided manually rewritten utterances (this makes the ru a manual run)
    - PASSAGE RANKING METHOD:
      - [x] method uses a pre-trained neural language model (BERT, Roberta, T5, etc.) (please describe specifics in the description field below)
    - PASSAGE RANKING DATA:
      - [x] method is trained with TREC Deep Learning Track and/or MS MARCO dataset
    - CONVERSATIONAL CONTEXT:
      - None
    - DESCRIPTION: First-pass retrieval using BM25 with default parameters, followed by T5 reranking, which has been fine-tuned on MS MARCO. Both steps use the manual query rewrites provided by organizers. No external data or conversational context is utilized.
    - JUDGING PRECEDENCE: 1 (highest priority)  

## Internal results using synthetic qrels

| *Run* | *Recall@1000* | *MRR* |
| -- | -- | -- |
| run_1 "UiS_raft" | 0.4100 | 0.1109 |
