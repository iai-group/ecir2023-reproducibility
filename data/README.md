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
    - [Washington Post 2020](https://trec.nist.gov/data/wapost/) => **TO DOWNLOAD**
  * Pre-processed collections (in TREC Web format) provided by the organizers => `$COLLECTIONS/trec-cast-y3`
    - KILT `kilt_knowledgesource.trecweb` (18G)
    - MS MARCO `msmarco-docs.trecweb` (21G)
    - WaPo `TREC_Washington_Post_collection.v4.trecweb` (3.4G)

## Qrels

  * `qrels/2019.txt`
  * `qrels/2020.txt` (Note: Turns with fewer than three relevant documents do not appear in the judgment file)
