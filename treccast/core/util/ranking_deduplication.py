""" Converts passage level ranking into document level ranking and de-duplicates
the documents in the ranking. """

import csv


def de_duplicate_ranking(runfile_path):
    doc_ids = set()
    with open(runfile_path, "r") as f_in, open(
        "data/runs/2021/input.clarke-cc_deduplicated", "w"
    ) as trec_dedup_out:
        reader = csv.reader(f_in, delimiter=" ")
        q_ids = set()
        for row in reader:
            q_id, _, doc_id, rank, score, run_id = row
            doc_id = doc_id.split("-")[0]
            if q_id not in q_ids:
                doc_ids = set()
            if doc_id not in doc_ids:
                q_ids.add(q_id)
                doc_ids.add(doc_id)
                trec_dedup_out.write(
                    " ".join(
                        [
                            q_id,
                            "Q0",
                            doc_id,
                            rank,
                            score,
                            run_id,
                        ]
                    )
                    + "\n"
                )


if __name__ == "__main__":
    de_duplicate_ranking("data/runs/2021/input.clarke-cc")
