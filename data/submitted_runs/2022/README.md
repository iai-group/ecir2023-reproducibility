# Submitted files

## Main-task run 'uis_duoboat'

  * What topic data does this run use?
    - [x] System uses automatic topic files
  
  * How is conversation understanding (NLP/rewriting) performed in this run (check all that apply)?
    - [x] method performs generative query rewriting (CQR), including models like BART/T5
  
  * What data is used for conversational query understanding in this run (check all that apply)?
    - [x] method uses other external data (please specify in the external resources field below)
  
  * How is ranking performed in this run (check all that apply)?
    - [x] method uses traditional unsupervised sparse retrieval (e.g. QL, BM25, etc.)
    - [x] method performs re-ranking with a pre-trained neural language model (BERT, Roberta, T5, etc.) (please describe specifics in the description field below)
  
  * What data is used to develop the ranking method in this run (check all that apply)?
    - [x] method is trained with previous CAsT datasets
    - [x] method is trained with TREC Deep Learning Track and/or MS MARCO dataset
  
  * Please specify all the methods used to handle feedback or clarification responses from the user.
    - [x] method does not treat them specially
  
  * Please describe the method used to generate the final conversational responses from one or more retrieved passages.
    - [x] method uses single source (single passage)
    - [x] method does not perform summarization (i.e. uses passages as-is)

  * Please describe the external resources used by this run, if applicable.
    - HuggingFace pretrained models, CANARD dataset, MS MARCO dataset, 2020 and 2021 CAsT datasets

  * Please provide a short description of this run.
    - The first-pass retrieval using BM25 with the parameters tuned on 2020 and 2021 CAsT datasets, is followed by mono T5 reranking and duo T5 reranking, which have been fine-tuned on MS MARCO. The query rewriting is performed with a HuggingFace model fine-tuned on CANARD dataset. Previously rewritten utterances and the last canonical response are used as a context.

## Main-task run 'uis_mixedboat'

  * What topic data does this run use?
    - [x] System uses automatic topic files and mixed initiative task responses
  
  * How is conversation understanding (NLP/rewriting) performed in this run (check all that apply)?
    - [x] method performs generative query rewriting (CQR), including models like BART/T5
  
  * What data is used for conversational query understanding in this run (check all that apply)?
    - [x] method uses other external data (please specify in the external resources field below)
  
  * How is ranking performed in this run (check all that apply)?
    - [x] method uses traditional unsupervised sparse retrieval (e.g. QL, BM25, etc.)
    - [x] method performs re-ranking with a pre-trained neural language model (BERT, Roberta, T5, etc.) (please describe specifics in the description field below)
  
  * What data is used to develop the ranking method in this run (check all that apply)?
    - [x] method is trained with previous CAsT datasets
    - [x] method is trained with TREC Deep Learning Track and/or MS MARCO dataset
  
  * Please specify all the methods used to handle feedback or clarification responses from the user.
    - [x] method detects and uses feedback
    - [x] method uses results from automatic clarification responses from MI sub-task (automatic-MI run)?
  
  * Please describe the method used to generate the final conversational responses from one or more retrieved passages.
    - [x] method uses single source (single passage)
    - [x] method does not perform summarization (i.e. uses passages as-is)

  * Please describe the external resources used by this run, if applicable.
    - HuggingFace pretrained models, CANARD dataset, MS MARCO dataset, 2020 and 2021 CAsT datasets, ClariQ dataset

  * Please provide a short description of this run.
    - We first fine-tune RoBERTa to filter out faulty clarifying questions (based on ClariQ and previous CAsT editions). Then, we rank the remaining clarifying questions with MPNet in a pairwise manner given a query rewritten by T5 fine-tuned on CANARD. We classify answers into three classes: useless, useful answer, and useful question. The classifier is trained on ClariQ. If the first class predicted, we do nothing to the original query. If the second or third class is predicted, we append the answer or the question, respectively, to the query. Then, the expanded query is once again rewritten with the T5-based model. The first-pass retrieval using BM25 with the parameters tuned on 2020 and 2021 CAsT datasets with PRF, is followed by mono T5 reranking and duo T5 reranking, which have been fine-tuned on MS MARCO. The query rewriting is performed with a HuggingFace model fine-tuned on CANARD dataset. Previously rewritten utterances and the last canonical response are used as a context.

## Mixed-initiative run 'uis_clearboat'

  * What category of mixed-initiative does this run use?
    - [x] System decides when to ask and what to ask
  
  * What method is used to select what turns require interaction (when to ask a question)?
    - [x] method does not use selective interaction (i.e. suggests interaction for all turns)
  
  * Please specify the method used to select/generate the system question (check all that apply):
    - [x] method selects from the question bank
  
  * Please specify the type(s) of context used by the selection method to determine the question to ask.
    - [x] method uses context from earlier conversation history 
  
  * How is question ranking performed in this run (check all that apply)? Please describe specifics in the description field below.
    - [x] method uses unsupervised retrieval (e.g. BM25, SBERT, etc.)
    - [x] method uses supervised pre-trained neural language model (e.g. Mono/Duo BERT/T5)
  
  * What data is used in the selection and generation methods?
    - [x] method uses previous CAsT data
    - [x] method uses datasets on clarifying questions (e.g. Qulac, ClariQ, etc.)
  
  * Please describe the external resources used by this run, if applicable.
    - CAsT'20, CAsT'21, ClariQ, SentenceTransformers, SimpleTransformers
  
  * Please provide a short description of this run.
    - We first fine-tune RoBERTa to filter out faulty clarifying questions. To this end, we set clarifying questions from ClariQ as a positive class and queries from previous CAsT editions as a negative class. Then, we rank the remaining clarifying questions with MPNet in a pairwise manner given a rewritten query. The rewritten query is generated with T5 fine-tuned on CANARD.

## Mixed-initiative run 'uis_ambiguousboat'
Todo: 
