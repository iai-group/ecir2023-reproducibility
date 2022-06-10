# Config

- When `treccast/main.py` is run, it will load the default configuration from `defaults/general.yaml` and `defaults/{{year}}.yaml`. **NB! These files should not be modified unless certain current pipeline will not break**
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


# Retrieval defaults
# If first pass file is specified, loads results from those, otherwise
# retrieve passages from elasticsearch index
first_pass_file: null
es:
 host_name: "gustav1.ux.uis.no:9204"
 k1: 1.2
 b: 0.75
 index_name: "ms_marco_kilt_wapo_clean"
 field: "catch_all"


# Re-ranking parameters
# Choose option between bert|t5
reranker: null
bert:
 base_model: "bert-base-uncased"
 reranker_path: null
```
