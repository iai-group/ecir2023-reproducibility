# Data

*General rule*: small files (<1 MB) on git, large files on gustav1 (with their documentation under git!). 

*Specifically*: 
  * On git: topic files, qrels, config files, and annotation files (rewrites, answer types, topic turn detection, etc.).
  * On gustav1: datasets, indices, and runfiles.
    - `$COLLECTIONS` refers to `gustav1:/data/collections`
    - `$DATA` refers to `gustav1:/data/scratch/trec-cast/data`

## Datasets

### 2020

  * [MS MARCO Passage Ranking collection](https://github.com/microsoft/MSMARCO-Passage-Ranking) => `$COLLECTIONS/msmarco-passage/collection.tar.gz` (988M)  
  * [TREC CAR paragraph collection v2.0](http://trec-car.cs.unh.edu/datareleases/) => `$COLLECTIONS/trec-car/paragraphCorpus.v2.0.tar.gz` (6.9G)
  * `2020/2020_topic_shift_labels.tsv` - Manually labeled queries from 2020, where `1` indicates a topic shift from the previous turn.

### 2021

  * Raw collections
    - [KILT Wikipedia](https://github.com/facebookresearch/KILT/) => `$COLLECTIONS/kilt/kilt_knowledgesource.json` (35G)
    - [MS MARCO (Documents)](https://github.com/microsoft/MSMARCO-Document-Ranking) => `$COLLECTIONS/msmarco-doc/msmarco-docs.tsv.gz` (7.9G)
    - [Washington Post 2020](https://trec.nist.gov/data/wapost/) => `$COLLECTIONS/wapo/WashingtonPost.v4.tar.gz` (2.4G)
  * Pre-processed collections (in TREC Web format) provided by the organizers => `$COLLECTIONS/trec-cast`
    - KILT `kilt_knowledgesource.trecweb` (18G)
    - MS MARCO `msmarco-docs.trecweb` (21G)
    - WaPo `TREC_Washington_Post_collection.v4.trecweb` (3.4G)

### 2022
  * Raw collections
    - [MS MARCO V2 (Documents)](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2021#document-ranking-dataset) => `$COLLECTIONS/msmarco-v2/msmarco_v2_doc.tar` (33G)
    - [KILT Wikipedia](https://github.com/facebookresearch/KILT/) => same as in 2021
    - [Washington Post 2020](https://trec.nist.gov/data/wapost/) => same as in 2021
  * Pre-processed collections (in TREC Web format) =>`$COLLECTIONS/trec-cast`
    - MS MARCO `/ms_marco/MARCO_{0-791}.trecweb` (109G)
    - KILT => same as in 2021
    - WaPo => same as in 2021

## Indices

Elasticsearch servers in use are `gustav1.ux.uis.no:9204` and `gorina39.ux.uis.no:9204`.

If the connection from gorina to the index on gustav1 fails, the workaround is to make a ssh connection with port forwarding using the following command:
`ssh -L 9204:gustav1.ux.uis.no:9204 -N -f gustav1.ux.uis.no`

### 2020

Available both on `gustav1` and `gorina39`:
  * `ms_marco_trec_car` => Basic inverted index without any preprocessing.
  * `ms_marco_trec_car_clean` => Inverted index with stopword removal and KStemming.

### 2021

Available both on `gustav1` and `gorina39`:
  * `ms_marco_kilt_wapo_clean` => Inverted index with stopword removal and KStemming.
    - document ID: `[MARCO|KILT|WAPO]_document_id-passage_index`
    - fields: 
      - title: Document title.
      - body: Passage text.
      - catch_all: Concatenation of title and body.

### 2022

So far available only on `gorina39`:
  * `ms_marco_v2_kilt_wapo` => Analogous to `ms_marco_kilt_wapo_clean` but using MS MARCO V2 collection. 

### Separate MS MARCO indices

Available only on `gorina39`:
  * ms_marco => Analogous to `ms_marco_kilt_wapo_clean` but using only MS MARCO collection.
  * ms_marco_v2 => Analogous to `ms_marco_v2_kilt_wapo` but using only MS MARCO V2 collection.

## Topics

  * [2019 topics](topics/2019)
    - [train_topics_v1.0.json](topics/2019/train_topics_v1.0.json): Training topics (30).
    - [evaluation_topics_v1.0.json](data/topics/2019/evaluation_topics_v1.0.json): Original test topics by the organizers (50).
    - [2019_manual_evaluation_topics_v1.0.json](topics/2019/2019_manual_evaluation_topics_v1.0.json): Test topics enriched with manual query rewrites (given by the organizers in a separate [TSV file](data/topics/2019/evaluation_topics_annotated_resolved_v1.0.tsv)) to follow the 2020/2021 format (generated using [this script](treccast/core/util/topics/create_2019_topics_file.py)).
  * [2020 topics](topics/2020)
    - [2020_manual_evaluation_topics_v1.0.json](topics/2020/2020_manual_evaluation_topics_v1.0.json): Manual query rewrites.
    - [2020_automatic_evaluation_topics_v1.0.json](topics/2020/2020_automatic_evaluation_topics_v1.0.json): Automatic query rewrites.
    - [automatic_evaluation_topics_annotated_v1.1.json](topics/2020/automatic_evaluation_topics_annotated_v1.1.json): Turns annotated with query/results dependences. Unlike what the filename suggests, these are actually manual rewrites. Also, the turns don't match those in the v1.0 files.
  * [2021 topics](topics/2021)
    - [2021_manual_evaluation_topics_v1.0.json](topics/2021/2021_manual_evaluation_topics_v1.0.json): Manual query rewrites.
    - [2021_automatic_evaluation_topics_v1.0.json](topics/2021/2021_automatic_evaluation_topics_v1.0.json): Automatic query rewrites.
  * [2022 topics](topics/2022)
    - [2022_manual_evaluation_topics_tree_v1.0.json](topics/2022/2022_manual_evaluation_topics_tree_v1.0.json): Conversation trees with manual query rewrites.
    - [2022_automatic_evaluation_topics_tree_v1.0.json](topics/2022/2022_automatic_evaluation_topics_tree_v1.0.json): Conversation trees with automatic query rewrites.
    - [2022_manual_evaluation_topics_v1.0.json](topics/2022/2022_manual_evaluation_topics_v1.0.json): Flattened conversation trees with duplicates with manual query rewrites.
    - [2022_automatic_evaluation_topics_v1.0.json](topics/2022/2022_automatic_evaluation_tree_v1.0.json): Flattened conversation trees with duplicates with automatic query rewrites.
    - [2022_evaluation_topics_turn_ids.json](topics/2022/2022_evaluation_topics_turn_ids.json): IDs that responses/ranked passages need to be returned for.

## Qrels

  * [qrels/2019_train.txt](qrels/2019_train.txt): 2019 training topics; only 5 are judged on a three-point scale (2 very relevant, 1 relevant, and 0 not relevant).
  * [qrels/2019.txt](qrels/2019.txt): 2019 test topics, judged on a five-point scale. Only 20 of the 50 test topics have judgments.
  * [qrels/2020.txt](qrels/2020.txt): 2020 test topics. Note: Turns with fewer than three relevant documents do not appear in the judgment file.

## Annotations/rewrites

  * Answer type prediction results under `answer_types`
  * Rewrites under `rewrites/{year}`, further documented [here](rewrites/README.md)  

## Fine-tuning data and models
 
  * Generated data for fine-tuning under `$DATA/fine_tuning` => [TODO](https://github.com/iai-group/trec-cast-2021/issues/162): document [here](fine_tuning/README.md)  
  * Trained models under `$DATA/models` => [TODO](https://github.com/iai-group/trec-cast-2021/issues/163): document [here](models/README.md)

## Runs

  * First-pass retrieval results under `$DATA/first_pass/{year}`, further documented [here](first_pass/README.md)   
  * Runfiles under `$DATA/runs/{year}`, further documented for [2020](runs/2020/README.md) and [2021](runs/2021/README.md)
