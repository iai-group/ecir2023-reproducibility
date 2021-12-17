from treccast.rewriter.t5_rewriter import T5Rewriter
from treccast.core.base import Query, Context


def test_rewrite_query():
    rewriter = T5Rewriter()
    context = Context()
    prev_query = Query(
        1, "How do you know when your garage door opener is going bad?"
    )
    context.history = [(prev_query, None)]
    rewrite = rewriter.rewrite_query(
        Query(2, "Now it stopped working. Why?"), context
    )

    assert (
        rewrite.question == "Now the garage door opener stopped working. Why?"
    )
