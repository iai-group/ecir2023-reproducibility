# TREC CAsT 2021

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the IAI group's participation at the TREC 2021 Conversational Assistance Track (CAsT).

  * [TREC CAsT website](http://www.treccast.ai/)
  * [Y3 guidelines](https://docs.google.com/document/d/1Eo0IqQedYc_rfTw-YxbvUGTpoYSmejU0iDlUzQWj3_w/edit?usp=sharing)

## Experiments 2020

| *Method* | *Script* | *MAP* | *MRR* | *NDCG* | *NDCG@5* |
| -- | -- | -- | -- | -- | -- |
| BM25 ES MS-MARCO only | `scripts/cast_bm25.sh` | 0.0228 | 0.1572 | 0.0718 | 0.0532 |
| *Reference* |||||
| Best @TREC2020 || 0.302 | 0.593 | 0.526 | |

Running first-pass BM25 ranker on 2020 data:
```
  scripts/$ sh 
```


