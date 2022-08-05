import pandas as pd
from treccast.rewriter.simpletransformers_rewriter_finetuning import (
    SimpleTransformersRewriterFinetuning,
)


def test_construct_input_data():
    context = [
        "What can you tell me about Gary Cherone?",
        "Gary Francis Caine Cherone is an American rock singer and songwriter.",
    ]
    question = "Did Gary sing well?"

    input_data = SimpleTransformersRewriterFinetuning.construct_input_data(
        context, question, "<sep>"
    )

    assert input_data == (
        "What can you tell me about Gary Cherone?<sep>Gary Francis Caine "
        "Cherone is an American rock singer and songwriter.<sep>Did Gary sing "
        "well?"
    )


def test_construct_input_data_no_context():
    context = []
    question = "Did Gary sing well?"

    input_data = SimpleTransformersRewriterFinetuning.construct_input_data(
        context, question, "<sep>"
    )

    assert input_data == "Did Gary sing well?"


def test_construct_input_data_too_long_input():
    previous_questions = 50 * [
        "Gary Francis Cherone is an American rock singer and songwriter."
    ]
    canonical_response = ["Cherone is an American rock singer and songwriter."]
    question = "Did Gary Francis Cherone sing well?"

    input_data = SimpleTransformersRewriterFinetuning.construct_input_data(
        previous_questions + canonical_response, question, "<sep>", True
    )

    assert input_data == "<sep>".join(
        previous_questions + ["Cherone is an American rock singer"] + [question]
    )


def test_construct_df_from_dataset():
    context = [
        "What can you tell me about Gary Cherone?",
        "Gary Francis Caine Cherone is an American rock singer and songwriter.",
    ]
    question = "Did Gary sing well?"
    rewrite = "Did Gary Cherone sing well?"
    dataset = {
        "Context": [context],
        "Question": [question],
        "Rewrite": [rewrite],
    }

    data_df_manual = pd.DataFrame(
        {
            "prefix": ["rewrite_question"],
            "input_text": [
                SimpleTransformersRewriterFinetuning.construct_input_data(
                    context, question, "<sep>"
                )
            ],
            "target_text": [rewrite],
        }
    )

    data_df = SimpleTransformersRewriterFinetuning.construct_df_from_dataset(
        dataset, "<sep>"
    )

    assert data_df_manual.equals(data_df)
