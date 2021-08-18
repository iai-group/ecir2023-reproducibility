"""doc_id, score, content triple to be used in Ranking """

from dataclasses import dataclass


@dataclass
class ScoredDocument:
    doc_id: str
    score: float
    content: str = None
