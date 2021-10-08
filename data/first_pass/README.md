# First-pass data

- First pass files can be generated with following command: `python -m treccast.main --config config/config_file.yaml`
- Necessary config fileds to set in the config file `config_file.yaml`:
    - retrieval: True
    - first_pass_file: null
    - year: "2021"|"2020" (Specifies which collection is used, 2021 is MSMARCO and WAPO, first pass results will be stored under the folder with year name. Currently 2020 and 2021 are supported.)
    - query_rewrite: "manual"|"automatic"|null (Defines which query rewriting to use, leave it as null for raw)
    - output_name: "bm25_2021_manual" (Output file name you prefer)
    - k: 10000 (number of documents to retrieve per query, set it to 10000 so that we can filter for smaller k)
- The first pass generation creates two files:
    - `data/first_pass/year/output_name.tsv` (TSV file with query_id, query, passage_id, passage, label fields)
    - `data/runs/year/output_name.trec` (TREC run format for using first pass scores for submission)
- Pre-computed first pass files for 2021 are located on gustav1 DATA=`/data/scratch/trec-cast-2021/data/`:
    - `$DATA/first_pass/2021/bm25_2021_manual.tsv`
    - `$DATA/runs/2021/bm25_2021_manual.trec`
    - `$DATA/first_pass/2021/bm25_2021_raw.tsv`
    - `$DATA/runs/2021/bm25_2021_raw.trec`
    - `$DATA/first_pass/2021/bm25_2021_automatic.tsv`
    - `$DATA/runs/2021/bm25_2021_automatic.trec`
