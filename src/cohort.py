"""Who is in the sample: the U21 cohort of a Premier League season.

U21 means aged under config.U21_AGE_LIMIT (21) on a cut-off date, with age in
completed years. By default the cut-off is 1 September of the season's first
year (config.u21_cutoff), so 1 Sep 2025 for 2025-26: a player born on 1 Sep
2004 turns 21 that day and is not U21; one born on 2 Sep 2004 is. The cohort
is every U21 player with Premier League minutes in the season. Players with at
least config.MIN_MINUTES_FOR_PER90 minutes are flagged as having enough
minutes for per-90 stats; how they are measured is up to features.py.

The season and the cut-off date are parameters, so the backtest (validate.py)
can build the cohort of an earlier season, such as U21 on 1 Sep 2018 for
2018-19, with this same code.

Only data available at the time:
- Minutes, starts, position and team come from that season's players_raw.csv
  alone, so nothing that happened after the season is used.
- Birth dates come from FPL where it has one, otherwise from the matched
  Transfermarkt player. FPL only has birth dates from 2024-25, so for earlier
  seasons they are looked up in later files. That is safe because a birth date
  never changes.
- The gameweek files only name clubs from 2020-21, so an earlier season's
  clubs are named from a later season. Club names are labels only. Clubs that
  have not been in the league since 2019-20 stay unnamed.

Writes data/processed/u21_cohort_{season}.csv and prints the counts.

Run from the repository root, after python -m src.matching:
    python -m src.cohort                                     # 2025-26, U21 on 1 Sep 2025
    python -m src.cohort --season 2018-19                    # U21 on 1 Sep 2018
    python -m src.cohort --season 2018-19 --cutoff 2019-01-01
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path

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
    "age_on_cutoff", "exact_age_on_cutoff", "minutes", "starts", "enough_minutes", "tm_player_id",
]


def is_u21(birth_dates: pd.Series, cutoff: date = config.U21_CUTOFF_DATE) -> pd.Series:
    """True where a player is under the U21 age limit on the cut-off date; missing if the birth date is."""
    return age_on(birth_dates, cutoff) < config.U21_AGE_LIMIT


def club_names(season: str, players_by_season: Mapping[str, pd.DataFrame]) -> dict[int, str]:
    """Club name for each team_code in a season.

    The season's own gameweek file names its clubs from 2020-21 on. Before
    that, a club takes its name from the latest later season that names it.
    """
    own = fpl_team_names({season: players_by_season[season]}, {season: read_fpl_gameweeks(season)})
    names = dict(zip(own["team_code"], own["club"]))
    unnamed = set(players_by_season[season]["team_code"]) - set(names)
    if unnamed:
        later = [s for s in config.FPL_SEASONS if s > season]
        later_names = latest_team_names(
            fpl_team_names({s: players_by_season[s] for s in later}, {s: read_fpl_gameweeks(s) for s in later})
        )
        names.update({code: later_names[code] for code in unnamed if code in later_names})
    return names


def season_players(season: str = config.CURRENT_SEASON, cutoff: date | None = None) -> pd.DataFrame:
    """Every player in a season's players_raw.csv, with club, minutes, the best birth date and U21 status.

    Args:
        season: an FPL season in config.FPL_SEASONS, such as "2018-19".
        cutoff: the date on which age is measured. By default 1 September of
            the season's first year (config.u21_cutoff).

    Raises:
        ValueError: if the season is not one of config.FPL_SEASONS.
    """
    if season not in config.FPL_SEASONS:
        raise ValueError(f"Unknown season {season!r}; expected one of {config.describe_seasons(config.FPL_SEASONS)}")
    cutoff = config.u21_cutoff(season) if cutoff is None else cutoff
    players = {s: read_fpl_players(s) for s in config.FPL_SEASONS}
    all_seasons = fpl_player_seasons(players)
    table = all_seasons[all_seasons["season"] == season].copy()
    if "starts" in players[season].columns:
        table = table.merge(players[season][["code", "starts"]], on="code", how="left")
    else:
        table["starts"] = pd.Series(pd.NA, index=table.index, dtype="Int64")  # FPL has starts from 2022-23
    clubs = club_names(season, players)
    matches = read_matches().set_index("fpl_code")

    table["name"] = table["first_name"] + " " + table["second_name"]
    table["club"] = table["team_code"].map(clubs)
    table["fpl_birth_date"] = table["code"].map(fpl_birth_dates(all_seasons))
    table["tm_birth_date"] = table["code"].map(matches["tm_birth_date"])
    table["tm_player_id"] = table["code"].map(matches["tm_player_id"]).astype("Int64")
    table["birth_date"] = table["fpl_birth_date"].fillna(table["tm_birth_date"])
    table["birth_date_source"] = np.select(
        [table["fpl_birth_date"].notna(), table["tm_birth_date"].notna()],
        ["FPL", "Transfermarkt"],
        default="unknown",
    )
    table["age_on_cutoff"] = age_on(table["birth_date"], cutoff)
    table["exact_age_on_cutoff"] = exact_age(table["birth_date"], cutoff).round(2)
    table["u21"] = is_u21(table["birth_date"], cutoff)
    table["enough_minutes"] = table["minutes"] >= config.MIN_MINUTES_FOR_PER90
    return table.rename(columns={"code": "fpl_code"})


def in_cohort(players: pd.DataFrame) -> pd.Series:
    """True for cohort members: U21 on the cut-off date and any PL minutes. An unknown birth date means out."""
    return players["u21"].fillna(False).astype(bool) & (players["minutes"] > 0)


def build_cohort(players: pd.DataFrame) -> pd.DataFrame:
    """The cohort, with players who have enough minutes for per-90 stats first, then by minutes."""
    cohort = players[in_cohort(players)]
    return cohort.sort_values(["enough_minutes", "minutes"], ascending=False, ignore_index=True)[COHORT_COLUMNS]


# ---------------------------------------------------------------------------
# Counts and checks
# ---------------------------------------------------------------------------

def funnel(players: pd.DataFrame, season: str, cutoff: date) -> pd.DataFrame:
    """From everyone listed down to the cohort, step by step."""
    u21 = players["u21"].fillna(False).astype(bool)
    played = players["minutes"] > 0
    enough = players["enough_minutes"]
    threshold = config.MIN_MINUTES_FOR_PER90
    steps = [
        (f"Players in FPL {season}", len(players)),
        ("  with a known birth date", int(players["birth_date"].notna().sum())),
        (f"  U21 (under {config.U21_AGE_LIMIT} on {cutoff:%d %b %Y})", int(u21.sum())),
        ("  U21 with any PL minutes: the cohort", int((u21 & played).sum())),
        (f"    with {threshold}+ minutes: enough for per-90 stats", int((u21 & enough).sum())),
        (f"    with 1 to {threshold - 1} minutes: listed without per-90 stats", int((u21 & played & ~enough).sum())),
    ]
    return pd.DataFrame(steps, columns=["group", "players"])


def breakdown(cohort: pd.DataFrame, column: str, order: list | None = None) -> pd.DataFrame:
    """Cohort players, and those with enough minutes for per-90 stats, grouped by one column."""
    table = cohort.groupby(column).agg(
        cohort=("fpl_code", "size"),
        enough_minutes=("enough_minutes", "sum"),
        minutes=("minutes", "sum"),
    )
    if order is not None:
        table = table.reindex(order, fill_value=0)
    return table.reset_index()


def cutoff_birthday(cutoff: date) -> pd.Timestamp:
    """The latest birth date that is too old: the cut-off date moved back by the age limit.

    For 1 Sep 2025 it is 1 Sep 2004. For a 29 February cut-off it is
    28 February, because someone born then has had their birthday by the cut-off.
    """
    return pd.Timestamp(cutoff) - pd.DateOffset(years=config.U21_AGE_LIMIT)


def boundary_check(players: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """The players born closest to the cut-off birthday on each side, with their U21 status."""
    birthday = cutoff_birthday(cutoff)
    dated = players.dropna(subset=["birth_date"]).sort_values("birth_date")
    too_old = dated[dated["birth_date"] <= birthday].tail(BOUNDARY_PLAYERS)
    young_enough = dated[dated["birth_date"] > birthday].head(BOUNDARY_PLAYERS)
    return pd.concat([too_old, young_enough])[["fpl_code", "name", "birth_date", "age_on_cutoff", "u21", "minutes"]]


def birth_date_disagreements(players: pd.DataFrame, cutoff: date) -> pd.DataFrame:
    """Players whose FPL and Transfermarkt birth dates differ, and whether U21 status would change."""
    both = players.dropna(subset=["fpl_birth_date", "tm_birth_date"])
    differ = both[both["fpl_birth_date"] != both["tm_birth_date"]].copy()
    differ["u21_with_fpl_date"] = is_u21(differ["fpl_birth_date"], cutoff)
    differ["u21_with_tm_date"] = is_u21(differ["tm_birth_date"], cutoff)
    return differ[["fpl_code", "name", "fpl_birth_date", "tm_birth_date", "u21_with_fpl_date", "u21_with_tm_date", "minutes"]]


# ---------------------------------------------------------------------------
# Running from the command line
# ---------------------------------------------------------------------------

def output_path(season: str, cutoff: date) -> Path:
    """Where the cohort is written. A non-default cut-off goes in the file name, so it never overwrites the default."""
    path = config.u21_cohort_path(season)
    if cutoff == config.u21_cutoff(season):
        return path
    return path.with_name(f"{path.stem}_cutoff_{cutoff:%Y_%m_%d}{path.suffix}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Build the U21 cohort of a Premier League season.")
    parser.add_argument("--season", choices=config.FPL_SEASONS, default=config.CURRENT_SEASON,
                        help=f"FPL season (default {config.CURRENT_SEASON})")
    parser.add_argument("--cutoff", type=date.fromisoformat, metavar="YYYY-MM-DD",
                        help="date on which age is measured (default 1 September of the season's first year)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the cohort, write it and print the counts."""
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # player names may not fit the console's encoding
    config.ensure_directories()
    season = args.season
    cutoff = config.u21_cutoff(season) if args.cutoff is None else args.cutoff
    players = season_players(season, cutoff)
    cohort = build_cohort(players)
    path = output_path(season, cutoff)
    cohort.to_csv(path, index=False, encoding="utf-8-sig")

    print(funnel(players, season, cutoff).to_string(index=False))
    print("\nCohort by position:")
    print(breakdown(cohort, "position", list(config.FPL_POSITIONS.values())).to_string(index=False))
    print("\nCohort by age on the cut-off date:")
    print(breakdown(cohort, "age_on_cutoff").to_string(index=False))
    print("\nCohort by club:")
    by_club = breakdown(cohort.fillna({"club": "(unnamed)"}), "club")
    print(by_club.sort_values(["cohort", "minutes"], ascending=False).to_string(index=False))
    print(f"\nPlayers born closest to the cut-off birthday ({cutoff_birthday(cutoff):%d %b %Y}), on each side:")
    print(boundary_check(players, cutoff).to_string(index=False))
    differ = birth_date_disagreements(players, cutoff)
    changes = int((differ["u21_with_fpl_date"] != differ["u21_with_tm_date"]).sum())
    print(f"\nFPL and Transfermarkt birth dates differ for {len(differ)} players; U21 status changes for {changes}:")
    print(differ.to_string(index=False))
    unknown = players[players["birth_date"].isna()]
    print(f"\nBirth date unknown: {len(unknown)} players, with {int(unknown['minutes'].sum())} PL minutes between them.")
    print(f"\nCohort written to {path} ({len(cohort)} players)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
