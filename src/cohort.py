"""The 2025/26 U21 cohort.

U21 means aged under config.U21_AGE_LIMIT on config.U21_CUTOFF_DATE (1 Sep
2025), with age in completed years. A player born on 1 Sep 2004 turns 21 that
day, so is not U21; one born on 2 Sep 2004 is. The cohort is every U21 player
with Premier League minutes in config.CURRENT_SEASON. Players with at least
config.MIN_MINUTES_FOR_PER90 minutes get attributes in the next step; the rest
are listed without them.

Birth dates come from FPL where it has one (the official source, complete for
everyone who played in 2025-26), otherwise from the matched Transfermarkt
player.

Writes data/processed/u21_cohort_2025_26.csv and prints the counts.

Run from the repository root, after python -m src.matching:
    python -m src.cohort
"""

from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd

from src import config
from src.clean import age_on, exact_age, fpl_birth_dates, fpl_player_seasons, fpl_team_names, latest_team_names
from src.load_data import read_fpl_gameweeks, read_fpl_players
from src.matching import read_matches

# How many players on each side of the cut-off birthday to list, to show how
# the boundary is handled.
BOUNDARY_PLAYERS: int = 3
COHORT_COLUMNS = [
    "fpl_code", "name", "web_name", "position", "club", "birth_date", "birth_date_source",
    "age_on_cutoff", "exact_age_on_cutoff", "minutes", "starts", "has_attributes", "tm_player_id",
]


def is_u21(birth_dates: pd.Series, cutoff: date = config.U21_CUTOFF_DATE) -> pd.Series:
    """True where a player is under the U21 age limit on the cut-off date; missing if the birth date is."""
    return age_on(birth_dates, cutoff) < config.U21_AGE_LIMIT


def current_season_players() -> pd.DataFrame:
    """Every player in the current season's players_raw.csv, with club, minutes and the best birth date."""
    season = config.CURRENT_SEASON
    players = {s: read_fpl_players(s) for s in config.FPL_SEASONS}
    all_seasons = fpl_player_seasons(players)
    current = all_seasons[all_seasons["season"] == season].copy()
    current = current.merge(players[season][["code", "starts"]], on="code", how="left")
    clubs = latest_team_names(fpl_team_names({season: players[season]}, {season: read_fpl_gameweeks(season)}))
    matches = read_matches().set_index("fpl_code")

    current["name"] = current["first_name"] + " " + current["second_name"]
    current["club"] = current["team_code"].map(clubs)
    current["fpl_birth_date"] = current["code"].map(fpl_birth_dates(all_seasons))
    current["tm_birth_date"] = current["code"].map(matches["tm_birth_date"])
    current["tm_player_id"] = current["code"].map(matches["tm_player_id"]).astype("Int64")
    current["birth_date"] = current["fpl_birth_date"].fillna(current["tm_birth_date"])
    current["birth_date_source"] = np.select(
        [current["fpl_birth_date"].notna(), current["tm_birth_date"].notna()],
        ["FPL", "Transfermarkt"],
        default="unknown",
    )
    current["age_on_cutoff"] = age_on(current["birth_date"], config.U21_CUTOFF_DATE)
    current["exact_age_on_cutoff"] = exact_age(current["birth_date"], config.U21_CUTOFF_DATE).round(2)
    current["u21"] = is_u21(current["birth_date"])
    current["has_attributes"] = current["minutes"] >= config.MIN_MINUTES_FOR_PER90
    return current.rename(columns={"code": "fpl_code"})


def build_cohort(players: pd.DataFrame) -> pd.DataFrame:
    """The U21 players with any PL minutes: attribute-eligible first, then by minutes."""
    cohort = players[players["u21"].fillna(False).astype(bool) & (players["minutes"] > 0)]
    return cohort.sort_values(["has_attributes", "minutes"], ascending=False, ignore_index=True)[COHORT_COLUMNS]


# ---------------------------------------------------------------------------
# Counts and checks
# ---------------------------------------------------------------------------

def funnel(players: pd.DataFrame) -> pd.DataFrame:
    """From everyone listed down to the cohort, step by step."""
    u21 = players["u21"].fillna(False).astype(bool)
    played = players["minutes"] > 0
    full = players["has_attributes"]
    threshold = config.MIN_MINUTES_FOR_PER90
    steps = [
        (f"Players in FPL {config.CURRENT_SEASON}", len(players)),
        ("  with a known birth date", int(players["birth_date"].notna().sum())),
        (f"  U21 (under {config.U21_AGE_LIMIT} on {config.U21_CUTOFF_DATE:%d %b %Y})", int(u21.sum())),
        ("  U21 with any PL minutes: the cohort", int((u21 & played).sum())),
        (f"    with {threshold}+ minutes: get attributes", int((u21 & full).sum())),
        (f"    with 1 to {threshold - 1} minutes: listed without attributes", int((u21 & played & ~full).sum())),
    ]
    return pd.DataFrame(steps, columns=["group", "players"])


def breakdown(cohort: pd.DataFrame, column: str, order: list | None = None) -> pd.DataFrame:
    """Cohort players and attribute-eligible players, grouped by one column."""
    table = cohort.groupby(column).agg(
        cohort=("fpl_code", "size"),
        with_attributes=("has_attributes", "sum"),
        minutes=("minutes", "sum"),
    )
    if order is not None:
        table = table.reindex(order, fill_value=0)
    return table.reset_index()


def cutoff_birthday() -> pd.Timestamp:
    """The cut-off date moved back by the age limit (1 Sep 2004): born on or before it means too old."""
    cutoff = config.U21_CUTOFF_DATE
    return pd.Timestamp(cutoff.replace(year=cutoff.year - config.U21_AGE_LIMIT))


def boundary_check(players: pd.DataFrame) -> pd.DataFrame:
    """The players born closest to the cut-off birthday on each side, with their U21 status."""
    birthday = cutoff_birthday()
    dated = players.dropna(subset=["birth_date"]).sort_values("birth_date")
    too_old = dated[dated["birth_date"] <= birthday].tail(BOUNDARY_PLAYERS)
    young_enough = dated[dated["birth_date"] > birthday].head(BOUNDARY_PLAYERS)
    return pd.concat([too_old, young_enough])[["fpl_code", "name", "birth_date", "age_on_cutoff", "u21", "minutes"]]


def birth_date_disagreements(players: pd.DataFrame) -> pd.DataFrame:
    """Players whose FPL and Transfermarkt birth dates differ, and whether U21 status would change."""
    both = players.dropna(subset=["fpl_birth_date", "tm_birth_date"])
    differ = both[both["fpl_birth_date"] != both["tm_birth_date"]].copy()
    differ["u21_with_fpl_date"] = is_u21(differ["fpl_birth_date"])
    differ["u21_with_tm_date"] = is_u21(differ["tm_birth_date"])
    return differ[["fpl_code", "name", "fpl_birth_date", "tm_birth_date", "u21_with_fpl_date", "u21_with_tm_date", "minutes"]]


def main() -> int:
    """Build the cohort, write it and print the counts."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # player names may not fit the console's encoding
    config.ensure_directories()
    players = current_season_players()
    cohort = build_cohort(players)
    cohort.to_csv(config.U21_COHORT_PATH, index=False, encoding="utf-8-sig")

    print(funnel(players).to_string(index=False))
    print("\nCohort by position:")
    print(breakdown(cohort, "position", list(config.FPL_POSITIONS.values())).to_string(index=False))
    print("\nCohort by age on the cut-off date:")
    print(breakdown(cohort, "age_on_cutoff").to_string(index=False))
    print("\nCohort by club:")
    print(breakdown(cohort, "club").sort_values(["cohort", "minutes"], ascending=False).to_string(index=False))
    print(f"\nPlayers born closest to the cut-off birthday ({cutoff_birthday():%d %b %Y}), on each side:")
    print(boundary_check(players).to_string(index=False))
    differ = birth_date_disagreements(players)
    changes = int((differ["u21_with_fpl_date"] != differ["u21_with_tm_date"]).sum())
    print(f"\nFPL and Transfermarkt birth dates differ for {len(differ)} players; U21 status changes for {changes}:")
    print(differ.to_string(index=False))
    unknown = players[players["birth_date"].isna()]
    print(f"\nBirth date unknown: {len(unknown)} players, with {int(unknown['minutes'].sum())} PL minutes between them.")
    print(f"\nCohort written to {config.U21_COHORT_PATH} ({len(cohort)} players)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
