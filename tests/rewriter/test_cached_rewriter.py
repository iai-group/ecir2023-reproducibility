import pytest

from treccast.rewriter.rewriter import CachedRewriter
from treccast.core.query.sparse_query import SparseQuery

REWRITES = "data/rewrites/2020/11_Human.tsv"


@pytest.mark.parametrize(
    "qid, original, rewrite",
    [
        (
            "81_1",
            "How do you know when your garage door opener is going bad?",
            "How do you know when your garage door opener is going bad?",
        ),
        (
            "99_5",
            "What are the differences between the two fats?",
            "What are the differences between saturated fat and trans fat?",
        ),
        (
            "103_10",
            "Why did the band break up?",
            "Why did the Grateful Dead break up?",
        ),
    ],
)
def test_cached_rewriter(qid: str, original: str, rewrite: str):
    rewriter = CachedRewriter(REWRITES)
    query = SparseQuery(qid, original)

    rewritten_query = rewriter.rewrite_query(query)

    assert rewritten_query.query_id == qid
    assert rewritten_query.question == rewrite
