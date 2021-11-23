"""Basic Query containing query ID and question"""

from dataclasses import dataclass


@dataclass
class Query:
    query_id: str
    question: str

    def get_topic_id(self) -> str:
        """Returns topic id under assumption that query ID consists of topic ID
        and turn number separated by an underscore."""
        return self.query_id.split("_")[0]
