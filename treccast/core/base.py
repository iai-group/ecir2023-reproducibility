"""Query and Document classes as representation of """

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Query:
    """Representation of a query. It contains query ID and question."""

    query_id: str
    question: str

    def get_topic_id(self) -> str:
        """Returns topic id under assumption that query ID consists of topic ID
        and turn number separated by an underscore."""
        return self.query_id.split("_")[0]

    def __str__(self):
        return self.question


@dataclass
class SparseQuery(Query):
    """Representation of a sparse query containing a dict of weighted terms."""

    weighted_terms: Dict[str, float]

    def __str__(self):
        return " ".join(
            f"{term}^{round(weight, 2)}"
            for term, weight in self.weighted_terms.items()
        )


@dataclass
class Document:
    """Representation of a document. It contains doc_id and optionally
    document content."""

    doc_id: str
    content: str = None


@dataclass
class ScoredDocument(Document):
    """Representation of a retrieved document. It contains doc_id and optionally
    document content and ranking score."""

    doc_id: str
    score: float = 0


@dataclass
class Context:
    """Represents conversation context. It is a list of previous query-document
    tuples where document is either the canonical answer or the top-ranked
    system response.
    """

    history: List[Tuple[Query, Document]] = field(default_factory=list)
