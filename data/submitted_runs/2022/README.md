# Submitted files

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

