import pytest

from treccast.core.ranking import Ranking
from treccast.core.query import Query


@pytest.fixture
def query():
    return Query(
        "qid_0",
        "How do you know when your garage door opener is going bad?",
    )


@pytest.fixture
def ranking():
    return Ranking(
        "qid_0",
        [
            {
                "doc_id": "1",
                "score": 50.62,
                "content": (
                    "Many people search for âstandard garage door sizesâ on a daily "
                    "basis. However there are many common size garage door widths and"
                    "heights but the standard size is probably more a matter of the age"
                    "of your home and what area of the town, state, or country that you "
                    "live in. There are a number of standard sizes for residential garage "
                    "doors in the United States."
                ),
            },
            {
                "doc_id": "2",
                "score": 1.52,
                "content": (
                    "The presence of communication amid scientific minds was equally"
                    "important to the success of the Manhattan Project as scientific"
                    "intellect was. The only cloud hanging over the impressive achievement"
                    " of the atomic researchers and engineers is what their success truly "
                    "meant; hundreds of thousands of innocent lives obliterated."
                ),
            },
            {
                "doc_id": "3",
                "score": 80.22,
                "content": (
                    "Garage Door Opener Problems. So, when the garage door opener decides "
                    "to take a day off, it can leave you stuck outside, probably during a "
                    "rain or snow storm. Though they may seem complicated, there really are"
                    " several things most homeowners can do to diagnose and repair opener "
                    "failures.nd, if you are careful not to damage the door or the seal on "
                    "the bottom of the door, use a flat shovel or similar tool to chip away"
                    "at the ice. Once you get the door open, clear any water, ice or snow "
                    "from the spot on the garage floor where the door rests when closed"
                ),
            },
        ],
    )


# QUERY2 = Query(
#     "qid_1",
#     "How much does it cost for someone to repair a garage door opener?",
# )


# SCORED_DOCS2 = [
#     {
#         "doc_id": "1",
#         "score": 20.43,
#         "content": "Typically, it will cost less to install a steel garage door without an"
#         " opener than to install a custom wood door with a garage door opener. "
#         "Recent innovations have also yielded high-tech doors with thick "
#         "insulation and energy-efficient glaze, as well as finished interior "
#         "surfaces and other significant upgrades.f your garage door has started"
#         " to malfunction, you might be considering installing a new or upgraded"
#         " door. Rest assured it is a smart investment. In fact, installing a "
#         "new garage door yields about 84 percent in resale value, according to "
#         "Remodeling Magazine",
#     },
#     {
#         "doc_id": "4",
#         "score": 12.3,
#         "content": "Organize volunteer community panels, boards, or committees that meet "
#         "with the offender to discuss the incident and offender obligation to "
#         "repair the harm to victims and community members. Facilitate the "
#         "process of apologies to victims and communities. Invite local victim "
#         "advocates to provide ongoing victim-awareness training for probation "
#         "staff",
#     },
#     {
#         "doc_id": "5",
#         "score": 100,
#         "content": "Purchasing extra remotes and getting openers set up for operation will"
#         " typically range from $100 to $400, which will add to the overall cost"
#         " of the garage door installation. If your opener works with the new "
#         "door, you won't need to have it replaced. In cases in which the new "
#         "door is much heavier than the old door, however, the old garage door "
#         "opener won't be able to handle the extra weight. This is something to "
#         "keep in mind when you're shopping for a new garage door.",
#     },
# ]
