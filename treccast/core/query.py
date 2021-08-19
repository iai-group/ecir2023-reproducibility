"""Basic Query containing query ID and question"""

from dataclasses import dataclass


@dataclass
class Query:
    query_id: str
    question: str
