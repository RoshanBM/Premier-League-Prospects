"""Shared cleaning steps for the raw tables.

Turns the raw files (read unchanged by src/load_data.py) into tidy tables that
later steps share: normalised names, one row per FPL player-season, club names
for FPL team codes, and Transfermarkt players and minutes per season.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import date

import pandas as pd
from unidecode import unidecode

from src import config


# ---------------------------------------------------------------------------
# Names and dates
# ---------------------------------------------------------------------------

def normalise_name(text: object) -> str:
    """Return a name in lower-case ASCII words: "Martin Ødegaard" becomes "martin odegaard".

    Accents are removed and punctuation (hyphens, apostrophes, dots) becomes a
    space. A missing value becomes an empty string.
    """
    if text is None or pd.isna(text):
        return ""
    plain = unidecode(str(text)).lower()
    return " ".join(part for part in re.split(r"[^a-z0-9]+", plain) if part)


def to_date(values: pd.Series) -> pd.Series:
    """Parse ISO dates to whole-second datetimes, so dates from both sources compare equal."""
    return pd.to_datetime(values, errors="coerce", format="ISO8601").astype("datetime64[s]")


def age_on(birth_dates: pd.Series, on: date) -> pd.Series:
    """Age in completed years on a given date; a missing birth date gives a missing age.

    Someone born on 1 Sep 2004 is 21 on 1 Sep 2025 and 20 the day before.
    Months and days are compared directly, so 29 February birthdays need no
    special case.
    """
    born = pd.to_datetime(birth_dates)
    had_birthday = (born.dt.month < on.month) | ((born.dt.month == on.month) & (born.dt.day <= on.day))
    years = on.year - born.dt.year - (~had_birthday).astype(int)
    return years.where(born.notna()).astype("Int64")


def exact_age(birth_dates: pd.Series, on: date) -> pd.Series:
    """Age in years with decimals on a given date (days divided by 365.25)."""
    return (pd.Timestamp(on) - pd.to_datetime(birth_dates)).dt.days / 365.25


# ---------------------------------------------------------------------------
# FPL
# ---------------------------------------------------------------------------

def fpl_player_seasons(players_by_season: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack players_raw.csv into one row per player per season, leaving out managers.

    Columns: season, code, id, first_name, second_name, web_name, known_name,
    position (GK, DEF, MID or FWD), team_code, minutes and birth_date. FPL only
    has birth_date from 2024-25 and known_name from 2025-26; before that they
    are missing. Rows whose element_type is not a player position (the 2024-25
    managers) are dropped.
    """
    frames = []
    for season, players in players_by_season.items():
        frame = pd.DataFrame(
            {
                "season": season,
                "code": players["code"],
                "id": players["id"],
                "first_name": players["first_name"],
                "second_name": players["second_name"],
                "web_name": players["web_name"],
                "known_name": players["known_name"] if "known_name" in players.columns else None,
                "position": players["element_type"].map(config.FPL_POSITIONS),
                "team_code": players["team_code"],
                "minutes": players["minutes"],
                "birth_date": (
                    to_date(players["birth_date"]) if "birth_date" in players.columns
                    else pd.Series(pd.NaT, index=players.index, dtype="datetime64[s]")
                ),
            }
        )
        frames.append(frame[frame["position"].notna()])
    return pd.concat(frames, ignore_index=True)


def fpl_birth_dates(player_seasons: pd.DataFrame) -> pd.Series:
    """One birth date per FPL code, from whichever season has it.

    The audit found no code whose birth date differs between seasons; this
    raises an error if that ever changes, rather than picking one silently.
    """
    dated = player_seasons.dropna(subset=["birth_date"])
    dates_per_code = dated.groupby("code")["birth_date"].nunique()
    conflicting = dates_per_code[dates_per_code > 1]
    if not conflicting.empty:
        raise ValueError(f"FPL codes with conflicting birth dates: {conflicting.index.tolist()}")
    return dated.groupby("code")["birth_date"].first()


def fpl_team_names(
    players_by_season: Mapping[str, pd.DataFrame],
    gameweeks_by_season: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Name each (season, team_code) from the club names in the gameweek files.

    From 2020-21, merged_gw.csv has a club name on every row. A team_code's name
    is the most common name across its players' rows, so a player who changed
    club mid-season cannot mislabel a team. Earlier seasons have no names.
    Returns the columns season, team_code and club.
    """
    rows = []
    for season, gameweeks in gameweeks_by_season.items():
        if "team" not in gameweeks.columns:
            continue
        codes = players_by_season[season][["id", "team_code"]].rename(columns={"id": "element"})
        joined = gameweeks[["element", "team"]].merge(codes, on="element", how="inner")
        for team_code, names in joined.groupby("team_code")["team"]:
            rows.append({"season": season, "team_code": int(team_code), "club": names.mode().iat[0]})
    return pd.DataFrame(rows, columns=["season", "team_code", "club"])


def latest_team_names(team_names: pd.DataFrame) -> dict[int, str]:
    """Map each team_code to its club name in the latest season it appears."""
    latest = team_names.sort_values("season").drop_duplicates("team_code", keep="last")
    return dict(zip(latest["team_code"], latest["club"]))


# ---------------------------------------------------------------------------
# Transfermarkt
# ---------------------------------------------------------------------------

def transfermarkt_players(players: pd.DataFrame) -> pd.DataFrame:
    """Transfermarkt players with a parsed birth date.

    The text 'Missing' in `position` (586 players in the audit) becomes a real
    missing value.
    """
    tidy = players[
        ["player_id", "name", "first_name", "last_name", "date_of_birth", "position",
         "sub_position", "current_club_name"]
    ].copy()
    tidy["birth_date"] = to_date(tidy.pop("date_of_birth"))
    tidy["position"] = tidy["position"].mask(tidy["position"] == "Missing")
    return tidy


def transfermarkt_season_minutes(
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    competition_ids: Iterable[str],
) -> pd.DataFrame:
    """Minutes per Transfermarkt player, season and competition, with the clubs played for.

    `season` is Transfermarkt's start year (2016 means 2016/17), taken from the
    game rather than the date. `club_ids` is a sorted tuple of the clubs the
    player appeared for in that competition and season.
    """
    wanted = appearances[appearances["competition_id"].isin(list(competition_ids))]
    with_season = wanted.merge(games[["game_id", "season"]], on="game_id", how="left")
    return with_season.groupby(["player_id", "season", "competition_id"], as_index=False).agg(
        minutes=("minutes_played", "sum"),
        games=("game_id", "nunique"),
        club_ids=("player_club_id", lambda ids: tuple(sorted(set(ids)))),
    )
