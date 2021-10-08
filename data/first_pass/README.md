# First-pass data

- First pass files can be generated with following command: `python -m treccast.main --config config_file --year year`
    - `--year` specifies which collection is used, 2021 is MSMARCO and WAPO, first pass results will be stored under the folder with year name. Currently 2020 and 2021 are supported.
- When the script is run it will load default configuration from `defaults/config_default.yaml` and `defaults/year.yaml`. **NB! These file should not be modified unless certain current pipeline will not break**
- Necessary config fields to set in the config file `config_file` are only the ones that overwrite default configuration.
    - output_name: "bm25_2021_manual" (Output file name you prefer)
    - query_rewrite: "manual"|"automatic"|null (Defines which query rewriting to use, leave it as null for raw)
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
**New files**
    - `$DATA/{first_pass|runs}/{2020|2021}/{raw_{1|2|5|10}k.{trec|tsv}`
    - `$DATA/{first_pass|runs}/{2020|2021}/{manual_{1|2|5|10}k.{trec|tsv}`
    - `$DATA/{first_pass|runs}/{2020|2021}/{automatic_{1|2|5|10}k.{trec|tsv}`
    - `$DATA/{first_pass|runs}/{2020|2021}/{raw_prev_{1|2|3|4|5|all}.{trec|tsv}`
    - `$DATA/{first_pass|runs}/{2020|2021}/{manual_prev_{1|2|3|4|5|all}.{trec|tsv}`
    - `$DATA/{first_pass|runs}/{2020|2021}/{automatic_prev_{1|2|3|4|5|all}.{trec|tsv}`
