import pytest
from pytorch_lightning import seed_everything

from treccast.core.ranking import Ranking
from treccast.core.query import Query
from treccast.reranker.train.bert_reranker_train import BERTRerankTrainer
from treccast.reranker.bert_reranker import BERTReranker


@pytest.fixture
def train_pairs():
    train_queries = [
        Query(
            "0",
            "How do you know when your garage door opener is going bad?",
        ),
        Query(
            "1",
            "How much does it cost for someone to repair a garage door opener?",
        ),
    ]

    ranking1 = Ranking("0")
    ranking1.add_doc(
        "1",
        2,
        "Many people search for âstandard garage door sizesâ on a daily "
        "basis. However there are many common size garage door widths and"
        "heights but the standard size is probably more a matter of the age"
        "of your home and what area of the town, state, or country that you "
        "live in. There are a number of standard sizes for residential garage "
        "doors in the United States.",
    )
    ranking1.add_doc(
        "2",
        0,
        "The presence of communication amid scientific minds was equally"
        "important to the success of the Manhattan Project as scientific"
        "intellect was. The only cloud hanging over the impressive achievement"
        " of the atomic researchers and engineers is what their success truly "
        "meant; hundreds of thousands of innocent lives obliterated.",
    )
    ranking1.add_doc(
        "3",
        3,
        "Garage Door Opener Problems. So, when the garage door opener decides "
        "to take a day off, it can leave you stuck outside, probably during a "
        "rain or snow storm. Though they may seem complicated, there really are"
        " several things most homeowners can do to diagnose and repair opener "
        "failures.nd, if you are careful not to damage the door or the seal on "
        "the bottom of the door, use a flat shovel or similar tool to chip away"
        "at the ice. Once you get the door open, clear any water, ice or snow "
        "from the spot on the garage floor where the door rests when closed",
    )

    ranking2 = Ranking("1")
    ranking2.add_doc(
        "1",
        1,
        "Typically, it will cost less to install a steel garage door without an"
        " opener than to install a custom wood door with a garage door opener. "
        "Recent innovations have also yielded high-tech doors with thick "
        "insulation and energy-efficient glaze, as well as finished interior "
        "surfaces and other significant upgrades.f your garage door has started"
        " to malfunction, you might be considering installing a new or upgraded"
        " door. Rest assured it is a smart investment. In fact, installing a "
        "new garage door yields about 84 percent in resale value, according to "
        "Remodeling Magazine",
    )
    ranking2.add_doc(
        "4",
        0,
        "Organize volunteer community panels, boards, or committees that meet "
        "with the offender to discuss the incident and offender obligation to "
        "repair the harm to victims and community members. Facilitate the "
        "process of apologies to victims and communities. Invite local victim "
        "advocates to provide ongoing victim-awareness training for probation "
        "staff",
    )
    ranking2.add_doc(
        "5",
        4,
        "Purchasing extra remotes and getting openers set up for operation will"
        " typically range from $100 to $400, which will add to the overall cost"
        " of the garage door installation. If your opener works with the new "
        "door, you won't need to have it replaced. In cases in which the new "
        "door is much heavier than the old door, however, the old garage door "
        "opener won't be able to handle the extra weight. This is something to "
        "keep in mind when you're shopping for a new garage door.",
    )
    return (train_queries, [ranking1, ranking2])


@pytest.fixture
def test_pairs():
    # Creates queries with opposite scores to test the reranker.
    test_queries = [
        Query(
            "qid_2",
            "I just had a breast biopsy for cancer. "
            "What are the most common types?",
        ),
        Query(
            "qid_3",
            "How can fires help an ecosystem?",
        ),
    ]
    ranking1 = Ranking("qid_2")
    ranking1.add_doc(
        "rel_doc",
        0,
        "More research is needed. Types Breast cancer can be: Ductal carcinoma:"
        " This begins in the milk duct and is the most common type. Lobular "
        "carcinoma: This starts in the lobules. Invasive breast cancer is when "
        "the cancer cells break out from inside the lobules or ducts and invade"
        " nearby tissue, increasing the chance of spreading to other parts of "
        "the body. Non-invasive breast cancer is when the cancer is still "
        "inside its place of origin and has not broken out.",
    )
    ranking1.add_doc(
        "unrel_doc",
        4,
        "The presence of communication amid scientific minds was equally"
        "important to the success of the Manhattan Project as scientific"
        "intellect was. The only cloud hanging over the impressive achievement"
        " of the atomic researchers and engineers is what their success truly "
        "meant; hundreds of thousands of innocent lives obliterated.",
    )

    ranking2 = Ranking("qid_3")
    ranking2.add_doc(
        "rel_doc",
        0,
        "Many ecosystems, particularly prairie, savanna, chaparral and "
        "coniferous forests, have evolved with fire as an essential contributor"
        " to habitat vitality and renewal. [1] Many plant species in "
        "fire-affected environments require fire to germinate, establish, or "
        "to reproduce. Wildfire suppression not only eliminates these species, "
        "but also the animals that depend upon them. [2]Campaigns in the United"
        " States have historically molded public opinion to believe that "
        "wildfires are always harmful to nature. This view is based on the "
        "outdated belief that ecosystems progress toward an equilibrium and "
        "that any disturbance, such as fire, disrupts the harmony of nature. "
        "More recent ecological research has shown, however, that fire is an "
        "integral component in the function and biodiversity of many natural "
        "habitats, and that the organisms within these communities have adapted"
        " to withstand, and even to exploit, natural wildfire.",
    )
    ranking2.add_doc(
        "unrel_doc",
        1,
        "Organize volunteer community panels, boards, or committees that meet "
        "with the offender to discuss the incident and offender obligation to "
        "repair the harm to victims and community members. Facilitate the "
        "process of apologies to victims and communities. Invite local victim "
        "advocates to provide ongoing victim-awareness training for probation "
        "staff",
    )
    return (test_queries, [ranking1, ranking2])


# TODO #47 Decide what needs to be tested in this module
# https://github.com/iai-group/trec-cast-2021/issues/47
def test_bert_reranker_train(train_pairs, test_pairs):
    seed_everything(7)
    ap = BERTRerankTrainer.add_model_specific_args()
    # Ignore any unknown args injected by pytest.
    args_parsed, _ = ap.parse_known_args()
    ap_dict = args_parsed.__dict__
    ap_dict["bert_type"] = "nboost/pt-bert-base-uncased-msmarco"
    ap_dict["num_epochs"] = 2
    # # Removed the args injected by pytest.

    # Create a pytorch-lightning trainer with all the training arguments
    trainer = BERTRerankTrainer.get_lightning_trainer(args_parsed)
    # Create a BERT ranker which has a linear classification head
    bert_reranker = BERTRerankTrainer(ap_dict, train_pairs)
    # trainer.fit trains the model by calling the train_dataloader and
    # training_step
    trainer.fit(bert_reranker)
    folder = f'{ap_dict["bert_type"]}_fine_tuned'
    bert_reranker.save_model(folder)
    ranker = BERTReranker(folder)
    queries, rankings = test_pairs
    reranked1 = ranker.rerank(queries[0], rankings[0]).fetch_topk_docs()
    assert reranked1[0]["doc_id"] == "rel_doc"
    assert reranked1[1]["doc_id"] == "unrel_doc"
    reranked2 = ranker.rerank(queries[1], rankings[1]).fetch_topk_docs()
    assert reranked2[0]["doc_id"] == "rel_doc"
