Dear TREC participant:

Attached are the evaluation results for your run(s):
       UiS_raft

Also attached are summary tables for this task.

The 2021 CAST track received 50 runs from 15 participating teams.

We uncovered a problem with passage definition at the start of assessing, namely
that the numbering of passages produced by the track-provided script varied slightly
depending on the version of component software packages used in the script. Since
canonical passage numbering is essential for valid evaluation and there was no way
of knowing which numbering was used in which runs, we had to back off to evaluating
based on documents rather than passages.  The use of documents as the evaluation unit
has several effects:
  1.  Assessing documents is much slower than assessing passages since documents
       are much longer.  To accommodate document judging in the available judgment
       budget, pools were shallower, fewer topics were judged at all, and fewer
       turns within judged topics could be assessed as compared to passage evaluation.
       The final qrels consists of documents in depth-7 pools across all submissions
       (plus some additional baseline runs produced by the coordinators). Nineteen
       topics had at least some turns judged, most of which were judged through turn eight.
       Some topics had additional turns judged and a few have less than eight turns judged.
       One turn that was judged was dropped from the evaluation set because it
       identified only one relevant document.  The final set of topics and turns in
       the evaluation qrels is as follows:
               106: 1-8, 10
               107: 1-8
               108: 1-8
               110: 1-9
               111: 1-9
               112: 1-7, 9
               113: 1-8
               115: 1-11
               116: 1-8
               117: 1-6, 8-10 [117_7 removed due to 1 relevant doc]
               118: 1-8
               119: 1-8
               121: 1-8
               124: 1-11
               125: 1-6
               127: 1-9
               128: 1-3
               129: 1-8
               131: 1-10
   2. The assessing rubric that was developed for passages was applied to documents. The
       main effect here is that concerns regarding "extraneous information" in the rubric
       were ignored in the judging.  The rubric as given to the assessors was the same
       as in 2020:
               0: Fails to meet. The passage is not relevant to the question.
                       The passage is unrelated to the target query.
               1: Slightly meets. The passage includes some information about the turn,
                       but does not directly answer it. Users will find some useful
                       information in the passage that may lead to the correct answer,
                       perhaps after additional rounds of conversation (better than nothing).
               2: Moderately meets. The passage answers the turn, but is focused on
                       other information that is unrelated to the question. The passage
                       may contain the answer, but users will need extra effort to pick
                       the correct portion. The passage may be relevant, but it may only
                       partially answer the turn, missing a small aspect of the context.
               3: Highly meets. The passage answers the question and is focused on the
                       turn. It would be a satisfactory answer if Google Assistant or
                       Alexa returned this passage in response to the query. It may
                       contain limited extraneous information.
               4: Fully meets. The passage is a perfect answer for the turn. It includes
                       all of the information needed to fully answer the turn in the
                       conversation context. It focuses only on the subject and contains
                       little extra information.
    3.  Submissions needed to be manipulated to remove duplicate documents (i.e., documents
         from which the original submission retrieved multiple passages).  Each submission
         was processed such that the first occurrence of a document in a ranked list was
         retained and all other occurrences removed.  Removing a document moved
         subsequent documents higher in the ranking and shortened the overall list to
         fewer than the allowed maximum number retrieved (1000).  The manipulated ranked
         lists were used for pooling and were the lists evaluated.


The main measures for the track are ndcg@3, ndcg@5, ndcg@500, and AP@500.  The use of 500 rather
than 1000 was motivated by the fact that the submission files were shortened by the de-duping.
In trec_eval output, these measures are ndcg_cut_3, ndcg_cut_5, ndcg_cut_500 and map_cut_500.
The trec_eval incantation to produce the attached score report is
       trec_eval -q -c -M1000  -m ndcg_cut.3,5,10,15,20,100,500 -m all_trec qrels run
(where run is the de-duped document ranking).

Tables of per-turn max, median, and minimum values for the four target measures are attached.  There is one table for each of the three types of runs: those that used the raw utterances only, those that used the raw utterances and canonical responses, and those that used manually rewritten utterances.  These tables and the relevance judgments are also posted in the CAST track section of the Tracks page in the Active Participants' part of the TREC web site.
