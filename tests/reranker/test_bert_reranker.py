import pytest

from treccast.core.ranking import Ranking
from treccast.reranker.bert_reranker import BERTReranker
from treccast.core.query.query import Query


@pytest.fixture
def dummy_queries():
    return [
        Query(
            "qid_0",
            "How do you know when your garage door opener is going bad?",
        ),
        Query(
            "qid_1",
            "How much does it cost for someone to repair a garage door opener?",
        ),
    ]


@pytest.fixture
def dummy_rankings():
    ranking1 = Ranking("0")
    ranking1.add_doc(
        "1",
        "Many people search for âstandard garage door sizesâ on a daily "
        "basis. However there are many common size garage door widths and"
        "heights but the standard size is probably more a matter of the age"
        "of your home and what area of the town, state, or country that you "
        "live in. There are a number of standard sizes for residential garage "
        "doors in the United States.",
        50.62,
    )
    ranking1.add_doc(
        "2",
        "The presence of communication amid scientific minds was equally"
        "important to the success of the Manhattan Project as scientific"
        "intellect was. The only cloud hanging over the impressive achievement"
        " of the atomic researchers and engineers is what their success truly "
        "meant; hundreds of thousands of innocent lives obliterated.",
        1.52,
    )
    ranking1.add_doc(
        "3",
        "Garage Door Opener Problems. So, when the garage door opener decides "
        "to take a day off, it can leave you stuck outside, probably during a "
        "rain or snow storm. Though they may seem complicated, there really are"
        " several things most homeowners can do to diagnose and repair opener "
        "failures.nd, if you are careful not to damage the door or the seal on "
        "the bottom of the door, use a flat shovel or similar tool to chip away"
        "at the ice. Once you get the door open, clear any water, ice or snow "
        "from the spot on the garage floor where the door rests when closed",
        80.22,
    )

    ranking2 = Ranking("1")
    ranking2.add_doc(
        "1",
        "Typically, it will cost less to install a steel garage door without an"
        " opener than to install a custom wood door with a garage door opener. "
        "Recent innovations have also yielded high-tech doors with thick "
        "insulation and energy-efficient glaze, as well as finished interior "
        "surfaces and other significant upgrades.f your garage door has started"
        " to malfunction, you might be considering installing a new or upgraded"
        " door. Rest assured it is a smart investment. In fact, installing a "
        "new garage door yields about 84 percent in resale value, according to "
        "Remodeling Magazine",
        20.43,
    )
    ranking2.add_doc(
        "4",
        "Organize volunteer community panels, boards, or committees that meet "
        "with the offender to discuss the incident and offender obligation to "
        "repair the harm to victims and community members. Facilitate the "
        "process of apologies to victims and communities. Invite local victim "
        "advocates to provide ongoing victim-awareness training for probation "
        "staff",
        12.3,
    )
    ranking2.add_doc(
        "5",
        "Purchasing extra remotes and getting openers set up for operation will"
        " typically range from $100 to $400, which will add to the overall cost"
        " of the garage door installation. If your opener works with the new "
        "door, you won't need to have it replaced. In cases in which the new "
        "door is much heavier than the old door, however, the old garage door "
        "opener won't be able to handle the extra weight. This is something to "
        "keep in mind when you're shopping for a new garage door.",
        100,
    )
    return [ranking1, ranking2]


# TODO #47 Decide what needs to be tested in this module
# https://github.com/iai-group/trec-cast-2021/issues/47
# def test_init_bert_ranker(dummy_queries, dummy_rankings):
#     ap = BERTReranker.add_model_specific_args()
#     ap_dict = ap.parse_args().__dict__
#     # Create a pytorch-lightning trainer with all the training arguments
#     trainer = BERTReranker.get_lightning_trainer(ap)
#     # Create a BERT ranker which has a linear classification head on top of
#     # BERT
#     bert_reranker = BERTReranker(dummy_queries, dummy_rankings, ap_dict)
#     # trainer.fit trains the model by calling the train_dataloader and
#     # training_step
#     trainer.fit(bert_reranker)
#     # trainer.test test the model by calling the test_dataloader and
#     # testing_step
#     print(trainer.test())
#     # trainer.predict test the model by calling the predict_dataloader and
#     # predict_step. It returns a list of dicts containing predictions for
#     # each batch.
#     print(trainer.predict())
#     preds = trainer.predict()[0]["prediction"]
#     assert len(preds) == 6
