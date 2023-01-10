# Data

*General rule*: small files (<1 MB) on git, large files on server (with their documentation under git!). 

*Specifically*: 
  * On git: topic files, qrels, config files, compressed runfiles, and rewrites.
  * On server: collections, indices, first-pass rankings, and fine-tuned models.
    - `$DATA` refers to the [shared folder](https://gustav1.ux.uis.no/downloads/ecir2023-reproducibility/) on the server. The access to the subfolder with preprocessed collections can be granted upon request.

## Datasets

### 2020

  * [MS MARCO Passage Ranking collection](https://github.com/microsoft/MSMARCO-Passage-Ranking) => `$DATA/collections/collection.tar.gz` (988M)  
  * [TREC CAR paragraph collection v2.0](http://trec-car.cs.unh.edu/datareleases/)
    - preprocessed TREC CAR paragraph collection => `$DATA/collections/dedup.articles-paragraphs.cbor` (6.9G)

### 2021

  * Raw collections
    - [KILT Wikipedia](https://github.com/facebookresearch/KILT/)
    - [MS MARCO (Documents)](https://github.com/microsoft/MSMARCO-Document-Ranking)
    - [Washington Post 2020](https://trec.nist.gov/data/wapost/)
  * Pre-processed collections (in TREC Web format) provided by the organizers
    - KILT `$DATA/collections/kilt_knowledgesource.trecweb` (18G)
    - MS MARCO `$DATA/collections/msmarco-docs.trecweb` (21G)
    - WaPo `$DATA/collections/TREC_Washington_Post_collection.v4.trecweb` (3.4G)

## Indices

Elasticsearch servers in use is `localhost:9204`.

### 2020

Elasticsearch index used in experiments => available on `$DATA/es_indices/2020/`:
  * `ms_marco_trec_car_clean` => Inverted index with stopword removal and KStemming.

ANCE index used in experiments => available on `$DATA/retrieval/ance/2020/`:
  - MS MARCO (passages) collection provided by [ir_dataset](https://ir-datasets.com/msmarco-passage.html#msmarco-passage)
  - TREC CAR paragraph collection v2.0 provided by [ir_dataset](https://ir-datasets.com/car.html#car/v2.0)

### 2021

Elasticsearch index used in experiments => available on `$DATA/es_indices/2021/`:
  * `ms_marco_kilt_wapo_clean` => Inverted index with stopword removal and KStemming.
    - document ID: `[MARCO|KILT|WAPO]_document_id-passage_index`
    - fields: 
      - title: Document title.
      - body: Passage text.
      - catch_all: Concatenation of title and body.

ANCE index used in experiments => available on `$DATA/retrieval/ance/2021/`:
  - our own generator for MS MARCO (documents) 
  - our own generator for KILT collection 
  - our own generator for WaPo 2020 collection
All generators are using the 2021 pre-processed collections (in TREC Web format) provided by the organizers

## Topics

  * [2020 topics](topics/2020)
    - [2020_manual_evaluation_topics_v1.0.json](topics/2020/2020_manual_evaluation_topics_v1.0.json): Manual query rewrites.
    - [2020_automatic_evaluation_topics_v1.0.json](topics/2020/2020_automatic_evaluation_topics_v1.0.json): Automatic query rewrites.
    - [automatic_evaluation_topics_annotated_v1.1.json](topics/2020/automatic_evaluation_topics_annotated_v1.1.json): Turns annotated with query/results dependences. Unlike what the filename suggests, these are actually manual rewrites. Also, the turns don't match those in the v1.0 files.
  * [2021 topics](topics/2021)
    - [2021_manual_evaluation_topics_v1.0.json](topics/2021/2021_manual_evaluation_topics_v1.0.json): Manual query rewrites.
    - [2021_automatic_evaluation_topics_v1.0.json](topics/2021/2021_automatic_evaluation_topics_v1.0.json): Automatic query rewrites.
## Qrels

  * [qrels/2020.txt](qrels/2020.txt): 2020 test topics. Note: Turns with fewer than three relevant documents do not appear in the judgment file.
  * [qrels/2021.txt](qrels/2021.txt): 2021 test topics.


## Rewrites

  * Rewrites under `data/rewrites/{year}`, further documented [here](rewrites/README.md)  

## Fine-tuning data and models
 
  * Generated data for fine-tuning under `data/fine_tuning` => documentation available [here](fine_tuning/README.md)  
  * Trained models under `$DATA/models`

## Runs

  * Sparse first-pass retrieval results under `$DATA/first_pass_rankings/{year}`, further documented [here](first_pass/README.md)
  * Runfiles under `data/runs/{year}`, further documented for [2020](runs/2020/README.md) and [2021](runs/2021/README.md)
