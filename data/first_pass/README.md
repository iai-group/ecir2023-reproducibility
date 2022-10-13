# First-pass data

- First pass files can be generated with following command: `python -m treccast.main --config config_file --year year`
    - `--year` specifies which collection is used, 2021 is MSMARCO and WAPO, first pass results will be stored under the folder with year name.
- When the script is run it will load default configuration from `config/defaults/general.yaml` and `defaults/year.yaml`. 
- Necessary config fields to set in the config file `config_file` are only the ones that overwrite default configuration.
    - output_name: "bm25_2021_manual" (Output file name you prefer)
- The first pass generation creates two files:
    - `data/first_pass/year/output_name.tsv` (TSV file with query_id, query, passage_id, passage, label fields)
    - `data/runs/year/output_name.trec` (TREC run format for using first pass scores for submission)
