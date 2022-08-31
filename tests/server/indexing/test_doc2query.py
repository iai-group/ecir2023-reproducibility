from treccast.indexer.docT5query import DocT5Query


def test_generate_queries(document_1):
    docT5query = DocT5Query(7)
    predicted = docT5query.generate_queries(document_1)
    assert len(predicted) == 7


def test_batch_generate_queries(document_1, document_2, document_3):
    docT5query = DocT5Query(5)
    predicted = docT5query.batch_generate_queries(
        [document_1, document_2, document_3]
    )
    assert len(predicted) == 3
    assert len(predicted[-1]) == 5
