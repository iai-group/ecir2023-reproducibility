import pytest
import treccast.rewriter.simpletransformers_rewriter_finetuning as srf
from treccast.rewriter.simpletransformers_rewriter_finetuning import (
    SimpleTransformersRewriterFinetuning,
)


@pytest.fixture
def simpletransformers_rewriter_finetuning() -> SimpleTransformersRewriterFinetuning:
    model_args = (
        SimpleTransformersRewriterFinetuning.get_simpletransformers_args(
            srf._MODEL_LOCATION
        )
    )

    return SimpleTransformersRewriterFinetuning(
        train_args=model_args,
        dataset="tests/data/",
        base_model_name=srf._BASE_MODEL_NAME,
        model_type=srf._MODEL_TYPE,
    )


def test_load_dataset(
    simpletransformers_rewriter_finetuning: SimpleTransformersRewriterFinetuning,
):
    assert (
        simpletransformers_rewriter_finetuning._dataset[
            "train"
        ].column_names.sort()
        == [
            "Context",
            "Question",
            "Rewrite",
            "Answer",
            "Answer_URL",
            "Conversation_no",
            "Turn_no",
            "Conversation_source",
        ].sort()
    )


def test_partition_dataset(
    simpletransformers_rewriter_finetuning: SimpleTransformersRewriterFinetuning,
):
    data = simpletransformers_rewriter_finetuning._dataset["train"]
    (
        train_dataset,
        val_dataset,
        _,
    ) = simpletransformers_rewriter_finetuning.partition_dataset("<sep>")

    assert len(data["Rewrite"]) == (
        len(train_dataset["input_text"]) + len(val_dataset["input_text"])
    )
