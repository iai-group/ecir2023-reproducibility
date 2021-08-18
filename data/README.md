# Data

## Datasets

`$COLLECTIONS` refers to `gustav1:/data/collections`

### Y2

  * [MS MARCO Passage Ranking collection](https://github.com/microsoft/MSMARCO-Passage-Ranking) => `$COLLECTIONS/msmarco-passage/collection.tar.gz` (988M)  
  * [TREC CAR paragraph collection v2.0](http://trec-car.cs.unh.edu/datareleases/) => `$COLLECTIONS/trec-car/paragraphCorpus.v2.0.tar.gz` (6.9G)

### Y3

  * Raw collections
    - [KILT Wikipedia](https://github.com/facebookresearch/KILT/) => `$COLLECTIONS/kilt/kilt_knowledgesource.json` (35G)
    - [MS MARCO (Documents)](https://github.com/microsoft/MSMARCO-Document-Ranking) => `$COLLECTIONS/msmarco-doc/msmarco-docs.tsv.gz` (7.9G)
    - [Washington Post 2020](https://trec.nist.gov/data/wapost/) => `$COLLECTIONS/wapo/WashingtonPost.v4.tar.gz` (2.4G)
  * Pre-processed collections (in TREC Web format) provided by the organizers => `$COLLECTIONS/trec-cast-y3`
    - KILT `kilt_knowledgesource.trecweb` (18G)
    - MS MARCO `msmarco-docs.trecweb` (21G)
    - WaPo `TREC_Washington_Post_collection.v4.trecweb` (3.4G)

## Indices

Elasticsearch server in use is `gustav1.ux.uis.no:9204`

### Y2

  * `ms_marco_trec_car` => Basic inverted index without any preprocessing.
  * `ms_marco_trec_car_clean` => Inverted index with stopword removal and KStemming.

### Y3
  * `ms_marco_kilt_wapo_clean` => Inverted index with stopword removal and KStemming.
    - document ID: `[MARCO|KILT|WAPO]_document_id-passage_index`
    - fields: 
      - title: Document title.
      - body: Passage text.
      - catch_all: Concatenation of title and body.

## Topics

  * [Y1 topics](topics/2019)
    - [train_topics_v1.0.json](topics/2019/train_topics_v1.0.json): Training topics (30).
    - [evaluation_topics_v1.0.json](data/topics/2019/evaluation_topics_v1.0.json): Original test topics by the organizers (50).
    - [2019_manual_evaluation_topics_v1.0.json](topics/2019/2019_manual_evaluation_topics_v1.0.json): Test topics enriched with manual query rewrites (given by the organizers in a separate [TSV file](data/topics/2019/evaluation_topics_annotated_resolved_v1.0.tsv)) to follow the Y2/Y3 format (generated using [this script](treccast/core/util/topics/create_2019_topics_file.py)).
  * [Y2 topics](topics/2020)
    - [2020_manual_evaluation_topics_v1.0.json](topics/2020/2020_manual_evaluation_topics_v1.0.json): Manual query rewrites.
    - [2020_automatic_evaluation_topics_v1.0.json](topics/2020/2020_automatic_evaluation_topics_v1.0.json): Automatic query rewrites.
    - [automatic_evaluation_topics_annotated_v1.1.json](topics/2020/automatic_evaluation_topics_annotated_v1.1.json): Turns annotated with query/results dependences. Unlike what the filename suggests, these are actually manual rewrites. Also, the turns don't match those in the v1.0 files.
  * [Y3 topics](topics/2021)
    - [2021_manual_evaluation_topics_v1.0.json](topics/2021/2021_manual_evaluation_topics_v1.0.json): Manual query rewrites.
    - [2021_automatic_evaluation_topics_v1.0.json](topics/2021/2021_automatic_evaluation_topics_v1.0.json): Automatic query rewrites.

## Qrels

  * [qrels/2019_train.txt](qrels/2019_train.txt): 2019 training topics; only 5 are judged on a three-point scale (2 very relevant, 1 relevant, and 0 not relevant).
  * [qrels/2019.txt](qrels/2019.txt): 2019 test topics, judged on a five-point scale. Only 20 of the 50 test topics have judgments.
  * [qrels/2020.txt](qrels/2020.txt): 2020 test topics. Note: Turns with fewer than three relevant documents do not appear in the judgment file.

## Rewrites

  * Under `rewrites/2019` and `rewrites/2020` for Y1 and Y2, respectively.
