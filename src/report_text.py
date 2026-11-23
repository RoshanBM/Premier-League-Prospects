"""Short scouting reports in plain English, built only from the data (no language model).

Each report is three to five neutral sentences for a coach:

1. Who the player is: position, club, age on 1 September 2025 and Premier
   League minutes in 2025/26.
2. Current Ability (CA) and its rank among Premier League players at the same
   position with at least 450 minutes, or a note that there is no CA yet.
3. The two strongest attributes and the weakest one, by name and value.
   Only attributes in the data are named (features.ATTRIBUTES); the text
   never comments on passing, dribbling, aerial ability or pressing, because
   no open source covers them.
4. Potential Ability (PA): the rank among 2025/26 U21 prospects, marked as
   experimental and, where flagged, as low evidence, with the range in
   words; or a note that there is not enough data to project the player.
5. The closest current statistical comparison at the same position.

Rules: every number comes from the processed tables; the wording is fixed and
neutral, with no hype words; the text uses the player's name rather than
pronouns. Sentences 3 and 5 are left out when there is nothing to say (no
attributes, or no comparison).

Used by the Streamlit app. To print reports from the command line:
    python -m src.report_text Hall Oriola Dowman
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import pandas as pd

from src import config
from src.ability import PA_STATUS_EXPERIMENTAL
from src.clean import normalise_name
from src.features import ATTRIBUTES

POSITION_WORDS: dict[str, tuple[str, str]] = {
    "GK": ("goalkeeper", "goalkeepers"),
    "DEF": ("defender", "defenders"),
    "MID": ("midfielder", "midfielders"),
    "FWD": ("forward", "forwards"),
}
# Attribute column -> display name, in the fixed order used to break ties.
ATTRIBUTE_NAMES: dict[str, str] = {attribute.column: attribute.name for attribute in ATTRIBUTES}
SEASON_TEXT: str = config.CURRENT_SEASON.replace("-", "/")  # "2025/26"
CUTOFF_TEXT: str = f"{config.U21_CUTOFF_DATE.day} {config.U21_CUTOFF_DATE:%B %Y}"  # "1 September 2025"


@dataclass(frozen=True)
class PlayerFacts:
    """Everything a report may mention about one player. Missing values are None.

    Attributes:
        name: full name, used once at the start.
        short_name: the FPL web name, used after that.
        position: "GK", "DEF", "MID" or "FWD".
        club: club name.
        age: completed age on the U21 cut-off date (1 September 2025).
        minutes: PL minutes in 2025/26.
        attributes: display name -> 1 to 20 value, only for attributes the
            player has (450+ minutes and applicable to the position).
        ca: Current Ability (1 to 20).
        ca_rank: rank of CA among players at the position with 450+ minutes (1 = highest).
        ca_pool: how many players at the position have a CA.
        pa_status: "experimental", "insufficient data: debut before 17", or
            None for a player outside the U21 cohort.
        pa: the PA rank among 2025/26 U21 prospects (1 to 20).
        pa_low_evidence: True if the PA rests on fewer than 450 career PL minutes.
        pa_range_text: the card's range in words.
        career_minutes: PL minutes from 2016-17 to 2025-26.
        closest_match: name of the most similar current player at the same position.
        closest_match_club: that player's club.
        closest_similarity: cosine similarity with that player (-1 to 1).
    """

    name: str
    short_name: str
    position: str
    club: str | None = None
    age: int | None = None
    minutes: int = 0
    attributes: dict[str, int] = field(default_factory=dict)
    ca: float | None = None
    ca_rank: int | None = None
    ca_pool: int | None = None
    pa_status: str | None = None
    pa: int | None = None
    pa_low_evidence: bool = False
    pa_range_text: str | None = None
    career_minutes: int | None = None
    closest_match: str | None = None
    closest_match_club: str | None = None
    closest_similarity: float | None = None


# ---------------------------------------------------------------------------
# Small wording helpers
# ---------------------------------------------------------------------------

def ordinal(number: int) -> str:
    """1 -> "1st", 2 -> "2nd", 11 -> "11th", 23 -> "23rd"."""
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def count_of(number: int, noun: str) -> str:
    """A count with its noun, singular for exactly 1: (1, "minute") -> "1 minute", (2176, "minute") -> "2,176 minutes"."""
    return f"{number:,} {noun}" + ("" if number == 1 else "s")


def lower_first(text: str) -> str:
    """The text with its first letter in lower case, so it can follow a semicolon."""
    return text[:1].lower() + text[1:]


def strongest_and_weakest(attributes: dict[str, int]) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """The two highest attributes, and every attribute tied for the lowest value among the rest.

    Ties for the top two keep the fixed attribute order (features.ATTRIBUTES),
    so the result does not depend on dictionary order. Ties for the lowest
    are all returned, in that order, so the text never names one of two
    equally weak attributes at random. With fewer than three attributes there
    is no separate weakest one, and the second list is empty.
    """
    order = {name: position for position, name in enumerate(ATTRIBUTE_NAMES.values())}
    ranked = sorted(attributes.items(), key=lambda item: (-item[1], order.get(item[0], len(order))))
    if len(ranked) < 3:
        return ranked, []
    rest = ranked[2:]
    lowest = min(value for _, value in rest)
    weakest = sorted((item for item in rest if item[1] == lowest), key=lambda item: order.get(item[0], len(order)))
    return ranked[:2], weakest


def join_names(names: list[str]) -> str:
    """["A"] -> "A"; ["A", "B"] -> "A and B"; ["A", "B", "C"] -> "A, B and C"."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


# ---------------------------------------------------------------------------
# The sentences
# ---------------------------------------------------------------------------

def identity_sentence(facts: PlayerFacts) -> str:
    """Position, club, age and minutes."""
    position = POSITION_WORDS.get(facts.position, (facts.position, facts.position))[0]
    article = "an" if position[:1] in "aeiou" else "a"
    club = f" ({facts.club})" if facts.club else ""
    age = f", aged {facts.age} on {CUTOFF_TEXT}," if facts.age is not None else ""
    minutes = count_of(facts.minutes, "Premier League minute")
    return f"{facts.name}{club} is {article} {position}{age} with {minutes} in {SEASON_TEXT}."


def ability_sentence(facts: PlayerFacts) -> str:
    """CA and its rank within the position, or why there is none."""
    if facts.ca is None:
        return (f"With fewer than {config.MIN_MINUTES_FOR_PER90} minutes this season, {facts.short_name} has no "
                f"per-90 attributes or Current Ability yet.")
    plural = POSITION_WORDS.get(facts.position, (facts.position, facts.position))[1]
    rank = ""
    if facts.ca_rank is not None and facts.ca_pool is not None:
        rank = (f", {ordinal(facts.ca_rank)} of {facts.ca_pool} {plural} with at least "
                f"{config.MIN_MINUTES_FOR_PER90} minutes")
    return f"Current Ability is {facts.ca:.1f} on the 1 to 20 scale{rank}."


def attribute_sentence(facts: PlayerFacts) -> str | None:
    """The strongest two attributes and the weakest, or None if the player has no attributes."""
    if not facts.attributes:
        return None
    strongest, weakest = strongest_and_weakest(facts.attributes)
    values = set(facts.attributes.values())
    if len(facts.attributes) >= 3 and len(values) == 1:
        return f"All {len(facts.attributes)} attributes are level at {values.pop()}."
    if not weakest:
        plural = POSITION_WORDS.get(facts.position, (facts.position, facts.position))[1]
        listed = " and ".join(f"{name} ({value})" for name, value in strongest)
        verb = "is the only attribute" if len(strongest) == 1 else "are the only attributes"
        return f"{listed} {verb} the data supports for {plural}."
    (first, first_value), (second, second_value) = strongest
    if len(weakest) == 1:
        name, value = weakest[0]
        lowest = f"the weakest is {name} ({value})"
    else:
        both = "both" if len(weakest) == 2 else "all"
        lowest = f"the weakest are {join_names([name for name, _ in weakest])} ({both} {weakest[0][1]})"
    return f"The strongest attributes are {first} ({first_value}) and {second} ({second_value}); {lowest}."


def potential_sentence(facts: PlayerFacts) -> str:
    """PA rank with its status and range, or why there is no PA."""
    if facts.pa_status is None:
        return f"Potential Ability is only projected for players under {config.U21_AGE_LIMIT} on {CUTOFF_TEXT}."
    if facts.pa_status != PA_STATUS_EXPERIMENTAL or facts.pa is None:
        return (f"There is not enough data to project {facts.short_name}: every Premier League season so far came "
                f"before age {config.MODEL_AGE_RANGE[0]}, the youngest age the model uses.")
    evidence = ""
    if facts.pa_low_evidence:
        minutes = (count_of(facts.career_minutes, "career PL minute") if facts.career_minutes is not None
                   else "a few career PL minutes")
        evidence = (f", a low-evidence figure based on only {minutes}, so it mostly reflects the "
                    f"average for the position")
    range_part = f"; {lower_first(facts.pa_range_text)}" if facts.pa_range_text else ""
    return (f"Potential Ability (experimental) is {facts.pa} on a 1 to 20 rank among {SEASON_TEXT} U21 prospects"
            f"{evidence}{range_part}.")


def comparison_sentence(facts: PlayerFacts) -> str | None:
    """The closest current comparison at the same position, or None if there is none."""
    if not facts.closest_match:
        return None
    club = f"{facts.closest_match_club}, " if facts.closest_match_club else ""
    similarity = f"similarity {facts.closest_similarity:.2f}" if facts.closest_similarity is not None else ""
    detail = f" ({club}{similarity})" if (club or similarity) else ""
    return f"The closest current statistical profile at the same position is {facts.closest_match}{detail}."


def report_sentences(facts: PlayerFacts) -> list[str]:
    """The report as a list of three to five sentences."""
    sentences = [
        identity_sentence(facts),
        ability_sentence(facts),
        attribute_sentence(facts),
        potential_sentence(facts),
        comparison_sentence(facts),
    ]
    return [sentence for sentence in sentences if sentence]


def scouting_report(facts: PlayerFacts) -> str:
    """The report as one paragraph."""
    return " ".join(report_sentences(facts))


# ---------------------------------------------------------------------------
# Building the facts from the processed tables
# ---------------------------------------------------------------------------

def _value(row: pd.Series | None, column: str) -> object | None:
    """A cell as a plain Python value, or None if the row or the value is missing."""
    if row is None or column not in row.index:
        return None
    value = row[column]
    return None if pd.isna(value) else value


def player_attributes(row: pd.Series) -> dict[str, int]:
    """Display name -> value for the attributes this player has (missing ones left out)."""
    found = {}
    for column, name in ATTRIBUTE_NAMES.items():
        value = _value(row, column)
        if value is not None:
            found[name] = int(value)
    return found


def facts_for(code: int, player_season: pd.DataFrame, ca: pd.DataFrame, pa: pd.DataFrame,
              similar: pd.DataFrame) -> PlayerFacts:
    """Collect the facts for one player from the processed tables.

    Args:
        code: the player's FPL code.
        player_season: player_season_2025_26.parquet.
        ca: ca_2025_26.parquet.
        pa: pa_projections_2025_26.parquet.
        similar: similar_players_2025_26.parquet.

    Raises:
        KeyError: if the code is not in player_season.
    """
    rows = player_season[player_season["fpl_code"] == code]
    if rows.empty:
        raise KeyError(f"no {config.CURRENT_SEASON} player with FPL code {code}")
    player = rows.iloc[0]

    def first(table: pd.DataFrame) -> pd.Series | None:
        hits = table[table["fpl_code"] == code]
        return None if hits.empty else hits.iloc[0]

    ca_row, pa_row = first(ca), first(pa)
    closest = similar[(similar["fpl_code"] == code) & (similar["comparison"] == "current")].sort_values("rank")
    match = closest.iloc[0] if not closest.empty else None
    ca_value = _value(ca_row, "ca")
    pool = int(ca.loc[ca["position"] == player["position"], "ca"].notna().sum()) if ca_value is not None else None
    age = _value(player, "age_on_cutoff")
    pa_value = _value(pa_row, "pa")
    rank = _value(ca_row, "ca_rank_in_position")
    career = _value(pa_row, "pl_minutes_to_date")
    similarity = _value(match, "similarity")
    return PlayerFacts(
        name=str(player["name"]),
        short_name=str(player["web_name"]),
        position=str(player["position"]),
        club=_value(player, "club"),
        age=int(age) if age is not None else None,
        minutes=int(player["minutes"]),
        attributes=player_attributes(player),
        ca=float(ca_value) if ca_value is not None else None,
        ca_rank=int(rank) if rank is not None else None,
        ca_pool=pool,
        pa_status=_value(pa_row, "pa_status"),
        pa=int(pa_value) if pa_value is not None else None,
        pa_low_evidence=bool(_value(pa_row, "pa_low_evidence") or False),
        pa_range_text=_value(pa_row, "pa_range_text"),
        career_minutes=int(career) if career is not None else None,
        closest_match=_value(match, "match_name"),
        closest_match_club=_value(match, "match_club"),
        closest_similarity=float(similarity) if similarity is not None else None,
    )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Print the report for each named 2025/26 player (FPL web name, accents and case ignored)."""
    parser = argparse.ArgumentParser(description="Print scouting reports for 2025/26 players.")
    parser.add_argument("players", nargs="+", metavar="WEB_NAME")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    player_season = pd.read_parquet(config.PLAYER_SEASON_PATH)
    ca = pd.read_parquet(config.CA_PATH)
    pa = pd.read_parquet(config.PA_PROJECTIONS_PATH)
    similar = pd.read_parquet(config.SIMILAR_PLAYERS_PATH)
    names = player_season["web_name"].map(normalise_name)
    for query in args.players:
        hits = player_season[(names == normalise_name(query)) & (player_season["minutes"] > 0)]
        if hits.empty:
            print(f"\n{query}: no {config.CURRENT_SEASON} player with PL minutes has this web name.")
        for code in hits["fpl_code"]:
            print(f"\n{scouting_report(facts_for(int(code), player_season, ca, pa, similar))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
