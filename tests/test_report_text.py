"""Tests for src/report_text.py: the rule-based scouting report.

Covers the wording helpers, each sentence, the low-evidence and no-PA cases,
the three-to-five sentence rule, neutral wording, and that the text never
mentions skills the data does not cover. The check on the real tables skips
itself if they have not been built yet.
"""

import re

import pandas as pd
import pytest

from src import config
from src.ability import PA_STATUS_EXPERIMENTAL, PA_STATUS_TOO_YOUNG
from src.report_text import (
    PlayerFacts,
    count_of,
    facts_for,
    ordinal,
    report_sentences,
    scouting_report,
    strongest_and_weakest,
)

HYPE_WORDS = re.compile(r"\b(elite|world[- ]class|star|superstar|wonderkid|generational|sensational|brilliant|"
                        r"exceptional|outstanding|phenomenal)\b", re.IGNORECASE)
UNCOVERED_SKILLS = re.compile(r"\b(pass\w*|dribbl\w*|aerial\w*|press\w*|progress\w*|heading)\b", re.IGNORECASE)
OUTFIELD = {"Finishing": 6, "Shot Threat": 9, "Chance Creation": 15, "Defending": 17, "Ball Recovery": 12}


def starter(**changes) -> PlayerFacts:
    """A regular U21 starter with every field filled in; keyword arguments override fields."""
    values = dict(
        name="Alex Example", short_name="Example", position="DEF", club="Testford", age=20, minutes=2176,
        attributes=dict(OUTFIELD), ca=13.2, ca_rank=22, ca_pool=147, pa_status=PA_STATUS_EXPERIMENTAL, pa=17,
        pa_low_evidence=False, pa_range_text="Projected to play about 30% to 85% of league minutes at 24 to 26",
        career_minutes=6000, closest_match="Sam Other", closest_match_club="Otherton", closest_similarity=0.874,
    )
    values.update(changes)
    return PlayerFacts(**values)


def low_evidence_player() -> PlayerFacts:
    """A U21 with one minute: no CA, no attributes, no comparison, PA flagged as low evidence."""
    return starter(minutes=1, attributes={}, ca=None, ca_rank=None, ca_pool=None, pa=11, pa_low_evidence=True,
                   career_minutes=1, closest_match=None, closest_match_club=None, closest_similarity=None,
                   pa_range_text="Range: from not playing regularly in the PL to about 45% of league minutes at 24 to 26")


def under_17_debutant() -> PlayerFacts:
    """A 15-year-old debutant: no PA, because every season so far is before 17."""
    return starter(age=15, minutes=152, attributes={}, ca=None, ca_rank=None, ca_pool=None,
                   pa_status=PA_STATUS_TOO_YOUNG, pa=None, pa_low_evidence=True, pa_range_text=None,
                   career_minutes=152, closest_match=None, closest_match_club=None, closest_similarity=None)


def senior_goalkeeper() -> PlayerFacts:
    """A player outside the U21 cohort, with the one goalkeeping attribute."""
    return starter(position="GK", age=29, attributes={"Goalkeeping": 7}, ca=7.0, ca_rank=23, ca_pool=28,
                   pa_status=None, pa=None, pa_range_text=None, career_minutes=None, closest_match=None)


ALL_CASES = {
    "starter": starter(),
    "low evidence": low_evidence_player(),
    "under 17": under_17_debutant(),
    "senior goalkeeper": senior_goalkeeper(),
}


# ---------------------------------------------------------------------------
# Wording helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("number, expected", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"),
                                              (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"),
                                              (101, "101st"), (111, "111th")])
def test_ordinal(number, expected):
    assert ordinal(number) == expected


def test_count_of_is_singular_only_for_one():
    assert count_of(1, "minute") == "1 minute"
    assert count_of(0, "minute") == "0 minutes"
    assert count_of(2176, "minute") == "2,176 minutes"


def test_strongest_and_weakest_picks_by_value():
    strongest, weakest = strongest_and_weakest(OUTFIELD)
    assert strongest == [("Defending", 17), ("Chance Creation", 15)]
    assert weakest == [("Finishing", 6)]


def test_ties_follow_the_fixed_attribute_order_not_dictionary_order():
    tied = {"Ball Recovery": 10, "Defending": 10, "Finishing": 10, "Shot Threat": 4, "Chance Creation": 4}
    reordered = dict(reversed(list(tied.items())))
    assert strongest_and_weakest(tied) == strongest_and_weakest(reordered)
    strongest, weakest = strongest_and_weakest(tied)
    assert strongest == [("Finishing", 10), ("Defending", 10)]
    assert weakest == [("Shot Threat", 4), ("Chance Creation", 4)]


def test_every_attribute_tied_for_weakest_is_named():
    tied = dict(OUTFIELD, **{"Shot Threat": 6})  # Finishing and Shot Threat both 6
    sentence = report_sentences(starter(attributes=tied))[2]
    assert sentence.endswith("the weakest are Finishing and Shot Threat (both 6).")


def test_the_weakest_is_never_one_of_the_strongest():
    strongest, weakest = strongest_and_weakest({"Finishing": 3, "Shot Threat": 3, "Chance Creation": 3})
    assert not set(weakest) & set(strongest)


def test_fewer_than_three_attributes_have_no_separate_weakest():
    assert strongest_and_weakest({"Goalkeeping": 7}) == ([("Goalkeeping", 7)], [])


# ---------------------------------------------------------------------------
# The sentences
# ---------------------------------------------------------------------------

class TestStarter:
    def test_identity(self):
        first = report_sentences(starter())[0]
        assert first == ("Alex Example (Testford) is a defender, aged 20 on 1 September 2025, "
                         "with 2,176 Premier League minutes in 2025/26.")

    def test_ca_and_rank_within_position(self):
        assert "Current Ability is 13.2 on the 1 to 20 scale, 22nd of 147 defenders with at least 450 minutes." \
            in report_sentences(starter())

    def test_two_strongest_and_the_weakest_attribute(self):
        assert ("The strongest attributes are Defending (17) and Chance Creation (15); the weakest is Finishing (6)."
                in report_sentences(starter()))

    def test_pa_rank_status_and_range(self):
        sentence = report_sentences(starter())[3]
        assert "Potential Ability (experimental) is 17 on a 1 to 20 rank among 2025/26 U21 prospects" in sentence
        assert "projected to play about 30% to 85% of league minutes at 24 to 26" in sentence
        assert "low-evidence" not in sentence

    def test_closest_comparison(self):
        assert report_sentences(starter())[-1] == ("The closest current statistical profile at the same position "
                                                   "is Sam Other (Otherton, similarity 0.87).")

    def test_five_sentences(self):
        assert len(report_sentences(starter())) == 5


class TestLowEvidence:
    def test_says_low_evidence_and_the_minutes_behind_it(self):
        text = scouting_report(low_evidence_player())
        assert "low-evidence figure based on only 1 career PL minute," in text
        assert "Potential Ability (experimental) is 11" in text

    def test_keeps_the_range_text(self):
        assert "range: from not playing regularly in the PL to about 45% of league minutes at 24 to 26." \
            in scouting_report(low_evidence_player())

    def test_no_ca_sentence_and_singular_minute(self):
        sentences = report_sentences(low_evidence_player())
        assert "with 1 Premier League minute in 2025/26." in sentences[0]
        assert sentences[1] == "With fewer than 450 minutes this season, Example has no per-90 attributes or " \
                               "Current Ability yet."

    def test_leaves_out_attributes_and_comparison(self):
        text = scouting_report(low_evidence_player())
        assert "strongest" not in text and "closest" not in text
        assert len(report_sentences(low_evidence_player())) == 3


class TestNoPa:
    def test_under_17_says_there_is_not_enough_data(self):
        text = scouting_report(under_17_debutant())
        assert ("There is not enough data to project Example: every Premier League season so far came before "
                "age 17, the youngest age the model uses.") in text

    def test_under_17_has_no_rank_or_range(self):
        text = scouting_report(under_17_debutant())
        assert "Potential Ability (experimental)" not in text
        assert "%" not in text

    def test_a_missing_pa_value_is_treated_as_no_pa_even_if_marked_experimental(self):
        text = scouting_report(starter(pa=None))
        assert "not enough data to project" in text

    def test_player_outside_the_u21_cohort(self):
        assert "Potential Ability is only projected for players under 21 on 1 September 2025." \
            in report_sentences(senior_goalkeeper())


def test_goalkeeper_has_one_attribute_sentence():
    assert "Goalkeeping (7) is the only attribute the data supports for goalkeepers." \
        in report_sentences(senior_goalkeeper())


def test_level_attributes_are_described_as_level():
    level = {name: 10 for name in OUTFIELD}
    assert "All 5 attributes are level at 10." in report_sentences(starter(attributes=level))


def test_missing_club_and_age_are_left_out():
    first = report_sentences(starter(club=None, age=None))[0]
    assert first == "Alex Example is a defender with 2,176 Premier League minutes in 2025/26."


@pytest.mark.parametrize("case", sorted(ALL_CASES))
def test_three_to_five_sentences(case):
    assert 3 <= len(report_sentences(ALL_CASES[case])) <= 5


@pytest.mark.parametrize("case", sorted(ALL_CASES))
def test_neutral_wording(case):
    assert not HYPE_WORDS.search(scouting_report(ALL_CASES[case]))


@pytest.mark.parametrize("case", sorted(ALL_CASES))
def test_never_mentions_skills_the_data_does_not_cover(case):
    assert not UNCOVERED_SKILLS.search(scouting_report(ALL_CASES[case]))


@pytest.mark.parametrize("case", sorted(ALL_CASES))
def test_no_gendered_pronouns(case):
    assert not re.search(r"\b(he|his|him|she|her)\b", scouting_report(ALL_CASES[case]), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Building the facts from tables
# ---------------------------------------------------------------------------

def toy_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Small versions of the four processed tables: one U21 starter (code 1) and one other defender (code 2)."""
    player_season = pd.DataFrame({
        "fpl_code": [1, 2], "name": ["Alex Example", "Sam Other"], "web_name": ["Example", "Other"],
        "position": ["DEF", "DEF"], "club": ["Testford", "Otherton"], "age_on_cutoff": [20, 27],
        "minutes": [2176, 3000], "finishing": [6, 10], "shot_threat": [9, 10], "chance_creation": [15, 10],
        "defending": [17, 10], "ball_recovery": [12, 10], "goalkeeping": [None, None],
    })
    ca = pd.DataFrame({"fpl_code": [1, 2], "position": ["DEF", "DEF"], "ca": [13.2, 10.0],
                       "ca_rank_in_position": [1, 2]})
    pa = pd.DataFrame({"fpl_code": [1], "pa": [17], "pa_status": [PA_STATUS_EXPERIMENTAL], "pa_low_evidence": [False],
                       "pa_range_text": ["Projected to play about 30% to 85% of league minutes at 24 to 26"],
                       "pl_minutes_to_date": [6000]})
    similar = pd.DataFrame({"fpl_code": [1, 1], "comparison": ["current", "current"], "rank": [2, 1],
                            "match_name": ["Second Best", "Sam Other"], "match_club": ["Elsewhere", "Otherton"],
                            "similarity": [0.5, 0.874]})
    return player_season, ca, pa, similar


def test_facts_for_reads_every_table():
    facts = facts_for(1, *toy_tables())
    assert facts == starter(ca_rank=1, ca_pool=2)


def test_facts_for_a_player_outside_the_cohort():
    facts = facts_for(2, *toy_tables())
    assert facts.pa_status is None and facts.pa is None and facts.closest_match is None
    assert facts.ca == 10.0 and facts.ca_rank == 2 and facts.ca_pool == 2


def test_facts_for_an_unknown_code_raises():
    with pytest.raises(KeyError):
        facts_for(99, *toy_tables())


REAL_TABLES = (config.PLAYER_SEASON_PATH, config.CA_PATH, config.PA_PROJECTIONS_PATH, config.SIMILAR_PLAYERS_PATH)


@pytest.mark.skipif(not all(path.exists() for path in REAL_TABLES), reason="tables not built (python run_pipeline.py)")
def test_every_real_player_with_minutes_gets_a_valid_report():
    player_season, ca, pa, similar = (pd.read_parquet(path) for path in REAL_TABLES)
    played = player_season[player_season["minutes"] > 0]
    for code in played["fpl_code"]:
        facts = facts_for(int(code), player_season, ca, pa, similar)
        sentences = report_sentences(facts)
        text = " ".join(sentences)
        assert 3 <= len(sentences) <= 5, code
        assert not HYPE_WORDS.search(text) and not UNCOVERED_SKILLS.search(text), code
        assert set(facts.attributes) <= {"Finishing", "Shot Threat", "Chance Creation", "Defending",
                                         "Ball Recovery", "Goalkeeping"}
        if facts.pa_status == PA_STATUS_TOO_YOUNG:
            assert "not enough data to project" in text, code
        if facts.pa_low_evidence and facts.pa is not None:
            assert "low-evidence" in text, code
