import pytest
import pandas as pd
from treccast.rewriter.simpletransformers_rewriter_finetuning import (
    SimpleTransformersRewriterFinetuning,
)


@pytest.fixture
def st_rewiter_finetuning() -> SimpleTransformersRewriterFinetuning:
    t5args = SimpleTransformersRewriterFinetuning.get_simpletransformers_args(
        "tests/data/st_rewiter_model"
    )
    return SimpleTransformersRewriterFinetuning(
        t5args, "tests/data/", "t5", "t5-base"
    )


def test_construct_input_data(st_rewiter_finetuning):
    context = [
        "What can you tell me about Gary Cherone?",
        "Gary Francis Caine Cherone is an American rock singer and songwriter.",
    ]
    question = "Did Gary sing well?"

    input_data = st_rewiter_finetuning.construct_input_data(
        context, question, "<sep>"
    )

    assert input_data == (
        "What can you tell me about Gary Cherone?<sep>Gary Francis Caine "
        "Cherone is an American rock singer and songwriter.<sep>Did Gary sing "
        "well?"
    )


def test_construct_input_data_no_context(st_rewiter_finetuning):
    context = []
    question = "Did Gary sing well?"

    input_data = st_rewiter_finetuning.construct_input_data(
        context, question, "<sep>"
    )

    assert input_data == "Did Gary sing well?"


def test_construct_input_data_too_long_input(st_rewiter_finetuning):
    previous_questions = 50 * [
        "Gary Francis Cherone is an American rock singer."
    ]
    canonical_response = ["Cherone is an American rock singer."]
    question = "Did Gary Francis Cherone sing well?"

    input_data = st_rewiter_finetuning.construct_input_data(
        previous_questions + canonical_response, question, "<sep>", True
    )

    assert input_data == "<sep>".join(
        previous_questions + ["Cherone is an"] + [question]
    )


def test_construct_df_from_dataset(st_rewiter_finetuning):
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
                st_rewiter_finetuning.construct_input_data(
                    context, question, "<sep>"
                )
            ],
            "target_text": [rewrite],
        }
    )

    data_df = st_rewiter_finetuning.construct_df_from_dataset(dataset, "<sep>")

    assert data_df_manual.equals(data_df)
