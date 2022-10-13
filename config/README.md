# Config

- When `treccast/main.py` is run, it will load the default configuration from `defaults/general.yaml` and `defaults/{{year}}.yaml`. **NB! Do not modify these files unless you are certain the current pipeline will not break**
- Defaults can be overwritten by creating a new configuration file and specifying desired parameters.

```yaml
# General config
# Number of paragraphs to retrieve
k: 1000
# Number of previous turns results to add to the pool of results
num_prev_turns: 0


# Output path
# Name for the output file without extension
output_name: filename


# Rewrite defaults
# Value for organizers query rewrites 
# choose between null|automatic|manual (null defaults to raw)
query_rewrite: null

# If true and path is specified, loads custom rewrites from file
rewrite: False
rewrite_path: null

# Query expansion by pseudo relevance feedback
prf:
  # currently implemented type "RM3"
  type: null
  # how many documents to use for feedback
  num_documents: 10
  # how many highest scoring tokens to add to the query
  num_tokens: 10

# Retrieval defaults
# If first pass file is specified, loads results from those, otherwise
# retrieve passages from elasticsearch index
first_pass_file: null
es:
 host_name: "localhost:9204"
 k1: 1.2
 b: 0.75
 index_name: "ms_marco_kilt_wapo_clean"
 field: "catch_all"


# Re-ranking parameters
# Choose option (t5)
reranker: null

# Reranking with duoT5
# Change to True to use pairwise duoT5 reranker and specify the top k documents
# for reranking. 
duot5: False
duot5_topk: 50

# ANCE dense retrieval
# Change to yes and specify the path to ANN index.
ance: no
ance_index: 

# Rewriter for re-ranking
# Specify the path to the rewrites that you want to use for re-ranking stage if
# they should be different from the ones used for first-pass retrieval.
reranker_rewrite_path:

