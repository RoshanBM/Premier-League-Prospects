"""Multi-season player panel: one row per player per Premier League season, 2016-17 to 2025-26.

This is the data the trajectory model (src/trajectory.py) is fitted on.

How it is built:
1. Players are linked across seasons by FPL ``code``, which the Day 1 audit
   found to be stable (it is the Opta player ID; the one exception, Kaine
   Kesler-Hayden, has 5 PL minutes in total).
2. Only players with PL minutes in at least one season are kept. Every season
   in which such a player is listed in FPL is an observed row, including
   seasons with 0 minutes (an "observed zero").
3. Once a player has played, a later season in which he is missing from FPL is
   added as a "filled zero": minutes share 0, flagged as filled. Leaving the
   league is a real outcome for a young player, and dropping those seasons
   would make the model look only at players who stayed. Filling stops at the
   current season or at age config.ZERO_FILL_MAX_AGE, whichever comes first.
   Seasons before a player's first minutes are never filled: before his debut
   there is no way to tell whether he was at a PL club.
4. Birth date comes from FPL where it has one (2024-25 on), otherwise from the
   matched Transfermarkt player, the same rule as src/cohort.py.
5. Position is FPL's position in that season. A filled season carries forward
   the position of the player's last listed season before it, which was known
   at the time. The matching table's single "latest position" is not used,
   because in the backtest it would leak a later season's position into an
   earlier forecast.
6. Age is the decimal age on 1 September of the season's first year (days /
   365.25). ``age_completed`` is the same age in completed years, worked out
   exactly as src/clean.age_on does, so it agrees with the U21 rule.
7. minutes_share = minutes / (38 x 90).
8. ``first_seen_age`` is the player's completed age in his first panel season
   with PL minutes (his debut season). The trajectory model is fitted only on
   players first seen at config.FIRST_SEEN_YOUNG_MAX_AGE (20) or under, because
   the panel starts in 2016-17: a player first seen older is only in it because
   he was still in the league (left truncation). See first_seen_young.

Writes data/processed/player_season_panel.parquet and prints the counts. Also
writes data/processed/truncation_check.csv: panel players whose matched
Transfermarkt record shows PL minutes in 2012/13 to 2015/16, before the panel
starts, as a check of how much left truncation remains.

Run from the repository root, after python -m src.matching:
    python -m src.panel
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src import config
from src.clean import fpl_birth_dates, fpl_player_seasons, fpl_team_names, latest_team_names, transfermarkt_season_minutes
from src.load_data import read_fpl_gameweeks, read_fpl_players, read_transfermarkt
from src.matching import PREMIER_LEAGUE, read_matches

PANEL_COLUMNS: list[str] = [
    "fpl_code", "season", "season_year", "name", "web_name", "position", "club", "team_code",
    "birth_date", "birth_date_source", "age", "age_completed", "age_c", "minutes", "minutes_share",
    "filled", "zero_source", "debut_season", "first_seen_age", "tm_player_id",
]
OBSERVED_COLUMNS: list[str] = ["fpl_code", "season", "name", "web_name", "position", "team_code", "minutes"]


# ---------------------------------------------------------------------------
# Ages
# ---------------------------------------------------------------------------

def season_cutoffs(season_years: pd.Series) -> pd.Series:
    """1 September of each season's first year, as datetimes."""
    parts = pd.DataFrame({"year": season_years, "month": config.U21_CUTOFF_MONTH, "day": config.U21_CUTOFF_DAY})
    return pd.to_datetime(parts)


def decimal_age(birth_dates: pd.Series, season_years: pd.Series) -> pd.Series:
    """Age in years with decimals on 1 September of each season's first year (days / 365.25).

    Missing birth dates give missing ages. Same arithmetic as clean.exact_age.
    """
    return (season_cutoffs(season_years) - pd.to_datetime(birth_dates)).dt.days / 365.25


def completed_age(birth_dates: pd.Series, season_years: pd.Series) -> pd.Series:
    """Age in completed years on 1 September of each season's first year.

    Uses the same birthday rule as clean.age_on: someone born on 1 Sep 2004 is
    21 on 1 Sep 2025. Taking the floor of decimal_age would get that case
    wrong (7,670 days / 365.25 is 20.999).
    """
    born = pd.to_datetime(birth_dates)
    month, day = config.U21_CUTOFF_MONTH, config.U21_CUTOFF_DAY
    had_birthday = (born.dt.month < month) | ((born.dt.month == month) & (born.dt.day <= day))
    years = season_years - born.dt.year - (~had_birthday).astype(int)
    return years.where(born.notna()).astype("Int64")


def centre_age(age: pd.Series | np.ndarray | float, centre: float = config.MODEL_AGE_CENTRE):
    """Age minus the centre (21 by default), so 0 means 21 and -2 means 19."""
    return age - centre


# ---------------------------------------------------------------------------
# Building the panel
# ---------------------------------------------------------------------------

def lookup(values: pd.Series, keys: pd.Series) -> pd.Series:
    """values[key] for each key, missing where a key is absent, keeping the keys' index.

    Used instead of keys.map(values) because pandas 3.1 cannot map through an
    empty datetime Series (it tries to cast it to float).
    """
    return pd.Series(values.reindex(keys.to_numpy()).to_numpy(), index=keys.index)


def first_played_year(observed: pd.DataFrame) -> pd.Series:
    """For each fpl_code with PL minutes, the first season (start year) in which he played."""
    played = observed[observed["minutes"] > 0]
    return played.groupby("fpl_code")["season_year"].min()


def filled_seasons(
    observed: pd.DataFrame,
    birth_dates: pd.Series,
    season_years: Iterable[int],
    max_fill_age: int = config.ZERO_FILL_MAX_AGE,
) -> pd.DataFrame:
    """The (fpl_code, season_year) pairs to add as filled zeros.

    A pair is filled when the player has already played in an earlier (or the
    same, impossible here) season, is not listed in FPL that season, and is at
    most max_fill_age in completed years on 1 September. A player without a
    known birth date gets no filled seasons, because his age cannot be checked.
    """
    debut = first_played_year(observed)
    years = sorted(season_years)
    grid = pd.DataFrame(
        [(code, year) for code, first in debut.items() for year in years if year > first],
        columns=["fpl_code", "season_year"],
    )
    listed = observed[["fpl_code", "season_year"]].assign(listed=True)
    grid = grid.merge(listed, on=["fpl_code", "season_year"], how="left")
    missing = grid[grid["listed"].isna()].drop(columns="listed")
    ages = completed_age(lookup(birth_dates, missing["fpl_code"]), missing["season_year"])
    return missing[(ages <= max_fill_age).fillna(False).to_numpy()].reset_index(drop=True)


def build_panel(
    observed: pd.DataFrame,
    birth_dates: pd.Series,
    seasons: Iterable[str] = config.FPL_SEASONS,
    max_fill_age: int = config.ZERO_FILL_MAX_AGE,
) -> pd.DataFrame:
    """One row per player per season: observed seasons plus filled zeros.

    Args:
        observed: one row per FPL code per season it is listed, with the
            columns in OBSERVED_COLUMNS (minutes may be 0).
        birth_dates: birth date per fpl_code (index). Codes missing from it
            keep their observed rows, with missing ages and no filled seasons.
        seasons: the FPL seasons the panel covers.
        max_fill_age: last age (completed years) at which a season is filled.

    Returns:
        A table with the columns in PANEL_COLUMNS, except club and tm_player_id,
        which build_full_panel adds. Players who never played are left out.
    """
    labels = {config.season_start_year(s): s for s in seasons}
    obs = observed[observed["season"].isin(list(labels.values()))].copy()
    obs["season_year"] = obs["season"].map(config.season_start_year)
    if obs.duplicated(["fpl_code", "season_year"]).any():
        raise ValueError("observed has more than one row for a player in a season")
    debut = first_played_year(obs)
    obs = obs[obs["fpl_code"].isin(debut.index)]
    obs["filled"] = False

    fills = filled_seasons(obs, birth_dates, labels.keys(), max_fill_age)
    fills["season"] = fills["season_year"].map(labels)
    fills["minutes"] = 0
    fills["filled"] = True

    panel = pd.concat([obs, fills], ignore_index=True).sort_values(["fpl_code", "season_year"], ignore_index=True)
    # A filled season takes the position of the last listed season before it.
    panel["position"] = panel.groupby("fpl_code")["position"].ffill()
    # Names are player-level labels: the latest listed version.
    latest = obs.sort_values("season_year").drop_duplicates("fpl_code", keep="last").set_index("fpl_code")
    panel["name"] = panel["fpl_code"].map(latest["name"])
    panel["web_name"] = panel["fpl_code"].map(latest["web_name"])
    panel["team_code"] = panel["team_code"].astype("Int64")
    panel["minutes"] = panel["minutes"].astype("int64")

    panel["birth_date"] = pd.to_datetime(lookup(birth_dates, panel["fpl_code"]))
    panel["age"] = decimal_age(panel["birth_date"], panel["season_year"])
    panel["age_completed"] = completed_age(panel["birth_date"], panel["season_year"])
    panel["age_c"] = centre_age(panel["age"])
    panel["minutes_share"] = panel["minutes"] / config.MAX_SEASON_MINUTES
    panel["zero_source"] = pd.Series(
        np.select([panel["filled"], panel["minutes"] == 0], ["filled", "observed"], default=""),
        index=panel.index,
    ).replace("", pd.NA)
    panel["debut_season"] = panel["fpl_code"].map(debut).map(labels)
    # How old he was when first seen: completed age in his first season with minutes.
    panel["first_seen_age"] = completed_age(panel["birth_date"], panel["fpl_code"].map(debut))
    return panel


def first_seen_young(frame: pd.DataFrame, max_age: int = config.FIRST_SEEN_YOUNG_MAX_AGE) -> pd.Series:
    """True for rows of players first seen at max_age or under.

    A player with no first_seen_age (no birth date) counts as False, so he is
    left out of anything restricted to players first seen young.
    """
    return (frame["first_seen_age"] <= max_age).fillna(False).astype(bool)


def observed_player_seasons() -> pd.DataFrame:
    """Every FPL player-season, 2016-17 to 2025-26, in the shape build_panel expects (managers left out)."""
    raw = fpl_player_seasons({season: read_fpl_players(season) for season in config.FPL_SEASONS})
    raw = raw.rename(columns={"code": "fpl_code"})
    raw["name"] = raw["first_name"] + " " + raw["second_name"]
    return raw


def player_birth_dates(player_seasons: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Birth date per fpl_code (FPL first, then Transfermarkt) and where it came from."""
    fpl = fpl_birth_dates(player_seasons.rename(columns={"fpl_code": "code"}))
    tm = matches.set_index("fpl_code")["tm_birth_date"]
    combined = fpl.combine_first(tm)
    source = pd.Series(np.where(combined.index.isin(fpl.index), "FPL", "Transfermarkt"), index=combined.index)
    return combined, source


def club_lookup() -> dict[int, str]:
    """Club name per FPL team_code, from the latest season whose gameweek file names it (2020-21 on)."""
    players = {season: read_fpl_players(season) for season in config.FPL_SEASONS}
    gameweeks = {season: read_fpl_gameweeks(season) for season in config.FPL_SEASONS}
    return latest_team_names(fpl_team_names(players, gameweeks))


def build_full_panel() -> pd.DataFrame:
    """The panel from the real files, with club names and Transfermarkt ids attached."""
    raw = observed_player_seasons()
    matches = read_matches()
    birth_dates, source = player_birth_dates(raw, matches)
    panel = build_panel(raw[OBSERVED_COLUMNS], birth_dates)
    panel["birth_date_source"] = panel["fpl_code"].map(source).where(panel["birth_date"].notna())
    panel["club"] = panel["team_code"].map(club_lookup())
    panel["tm_player_id"] = panel["fpl_code"].map(matches.set_index("fpl_code")["tm_player_id"]).astype("Int64")
    return panel[PANEL_COLUMNS]


def read_panel() -> pd.DataFrame:
    """Read the panel this module writes.

    Raises FileNotFoundError with the command to run if it does not exist yet.
    """
    if not config.PANEL_PATH.exists():
        raise FileNotFoundError(f"{config.PANEL_PATH} not found. Run: python -m src.panel")
    return pd.read_parquet(config.PANEL_PATH)


# ---------------------------------------------------------------------------
# Checks printed after a run
# ---------------------------------------------------------------------------

def transfermarkt_pl_minutes() -> pd.DataFrame:
    """Premier League minutes per Transfermarkt player and season (start year), seasons with minutes only."""
    minutes = transfermarkt_season_minutes(read_transfermarkt("appearances"), read_transfermarkt("games"), [PREMIER_LEAGUE])
    return minutes[minutes["minutes"] > 0].rename(columns={"player_id": "tm_player_id", "season": "season_year"})


def transfermarkt_conflicts(panel: pd.DataFrame, tm_minutes: pd.DataFrame) -> pd.DataFrame:
    """Filled zeros for which Transfermarkt does record Premier League minutes.

    FPL is the source of truth for PL minutes, but a filled zero rests on a
    player's absence from FPL. If Transfermarkt shows PL minutes in such a
    season, the fill would be wrong. Transfermarkt has gaps of its own (see
    matching_report.md), so an empty result is a partial check, not proof.
    """
    filled = panel[panel["filled"]].dropna(subset=["tm_player_id"]).astype({"tm_player_id": "int64"})
    return filled.merge(tm_minutes[["tm_player_id", "season_year", "minutes"]], on=["tm_player_id", "season_year"],
                        suffixes=("", "_transfermarkt"))


def pre_panel_minutes(
    panel: pd.DataFrame,
    tm_minutes: pd.DataFrame,
    seasons: Iterable[int] = config.PRE_PANEL_SEASON_STARTS,
    young: int = config.FIRST_SEEN_YOUNG_MAX_AGE,
) -> pd.DataFrame:
    """Panel players whose matched Transfermarkt record has PL minutes before the panel starts.

    The panel starts in 2016-17, so a player who played in the PL earlier was
    really first seen younger than first_seen_age says. Two cases matter:
    - first seen young in the panel: still young, but his history in the
      panel starts after his real debut (residual truncation);
    - first seen older in the panel, but young by Transfermarkt: he is left
      out of the model although he would belong in it.
    Report only; the model is not changed. Only matched players with a
    first_seen_age can be checked, and Transfermarkt misses some
    player-seasons, so counts are lower bounds.

    Args:
        panel: the player-season panel (needs fpl_code, name, tm_player_id,
            birth_date, first_seen_age and debut_season).
        tm_minutes: Transfermarkt PL minutes per tm_player_id and season_year
            (start year), as transfermarkt_pl_minutes returns.
        seasons: the pre-panel seasons to look at (start years).
        young: the oldest age that counts as first seen young.

    Returns:
        One row per player with pre-panel PL minutes: the panel columns above,
        first_seen_young, earliest_tm_season (start year),
        tm_seasons_before_panel, tm_minutes_before_panel, tm_first_seen_age
        (completed age on 1 September of the earliest of those seasons) and
        young_by_tm.
    """
    players = panel.drop_duplicates("fpl_code")[["fpl_code", "name", "tm_player_id", "birth_date", "first_seen_age",
                                                 "debut_season"]]
    # Players with no first_seen_age (no birth date) are in neither group, so they are left out.
    players = players.dropna(subset=["tm_player_id", "first_seen_age"]).astype({"tm_player_id": "int64"})
    early = tm_minutes[tm_minutes["season_year"].isin(list(seasons))]
    early = early.groupby("tm_player_id").agg(
        earliest_tm_season=("season_year", "min"),
        tm_seasons_before_panel=("season_year", "nunique"),
        tm_minutes_before_panel=("minutes", "sum"),
    ).reset_index()
    hits = players.merge(early, on="tm_player_id")
    hits["first_seen_young"] = first_seen_young(hits, young)
    hits["tm_first_seen_age"] = completed_age(hits["birth_date"], hits["earliest_tm_season"])
    hits["young_by_tm"] = (hits["tm_first_seen_age"] <= young).fillna(False).astype(bool)
    return hits.sort_values(["first_seen_young", "fpl_code"], ascending=[False, True], ignore_index=True)


def truncation_counts(panel: pd.DataFrame, hits: pd.DataFrame) -> list[tuple[str, int]]:
    """Counts for the residual truncation check (see pre_panel_minutes), as (label, count) pairs in print order."""
    players = panel.drop_duplicates("fpl_code")
    young = first_seen_young(players)
    older = ~young & players["first_seen_age"].notna()
    matched = players["tm_player_id"].notna()
    early_young = hits["first_seen_young"]
    return [
        ("players first seen young (20 or under)", int(young.sum())),
        ("  matched to Transfermarkt", int((young & matched).sum())),
        ("  with Transfermarkt PL minutes in 2012/13 to 2015/16", int(early_young.sum())),
        ("players first seen older (21 or over)", int(older.sum())),
        ("  matched to Transfermarkt", int((older & matched).sum())),
        ("  with Transfermarkt PL minutes in 2012/13 to 2015/16", int((~early_young).sum())),
        ("  of those, 20 or under in their first such season", int((~early_young & hits["young_by_tm"]).sum())),
        ("players with no first_seen_age (no birth date)", int(players["first_seen_age"].isna().sum())),
    ]


def counts(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows per season: listed with minutes, observed zeros and filled zeros."""
    table = panel.assign(
        played=panel["minutes"] > 0,
        observed_zero=panel["zero_source"].eq("observed").fillna(False),
        filled_zero=panel["filled"],
    ).groupby("season").agg(
        rows=("fpl_code", "size"),
        played=("played", "sum"),
        observed_zero=("observed_zero", "sum"),
        filled_zero=("filled_zero", "sum"),
    )
    total = table.sum().to_frame("all seasons").T
    return pd.concat([table, total]).reset_index(names="season")


def main() -> int:
    """Build the panel, write it and print the counts."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # player names may not fit the console's encoding
    config.ensure_directories()
    panel = build_full_panel()
    panel.to_parquet(config.PANEL_PATH, index=False)

    low, high = config.MODEL_AGE_RANGE
    in_range = panel["age_completed"].between(low, high).fillna(False)
    no_birth_date = panel[panel["birth_date"].isna()]
    print(f"Panel: {len(panel)} rows, {panel['fpl_code'].nunique()} players, "
          f"seasons {panel['season'].min()} to {panel['season'].max()}")
    print(f"Age (decimal, on 1 Sep): {panel['age'].min():.2f} to {panel['age'].max():.2f}")
    print(f"Rows at ages {low} to {high} (model range): {int(in_range.sum())}, "
          f"players {panel.loc[in_range, 'fpl_code'].nunique()}")
    print(f"Rows with no birth date: {len(no_birth_date)} ({no_birth_date['fpl_code'].nunique()} players: "
          f"{', '.join(no_birth_date['name'].unique())})")
    print("\nRows per season:")
    print(counts(panel).to_string(index=False))
    print("\nRows by position:")
    print(panel.groupby("position").agg(rows=("fpl_code", "size"), players=("fpl_code", "nunique")).to_string())
    print("\nRows by age (completed years):")
    print(panel.groupby("age_completed").agg(rows=("fpl_code", "size"), filled=("filled", "sum"),
                                             mean_share=("minutes_share", "mean")).round(3).to_string())
    players = panel.drop_duplicates("fpl_code")
    print("\nPlayers by age when first seen (completed age in the first season with PL minutes):")
    print(players["first_seen_age"].value_counts(dropna=False).sort_index().to_string())

    tm_minutes = transfermarkt_pl_minutes()
    conflicts = transfermarkt_conflicts(panel, tm_minutes)
    print(f"\nFilled zeros where Transfermarkt records PL minutes: {len(conflicts)}")
    if not conflicts.empty:
        print(conflicts[["fpl_code", "name", "season", "minutes_transfermarkt"]].to_string(index=False))

    hits = pre_panel_minutes(panel, tm_minutes)
    hits.to_csv(config.TRUNCATION_CHECK_PATH, index=False, encoding="utf-8-sig")
    print("\nResidual truncation check (Transfermarkt PL minutes before the panel starts; report only):")
    for label, value in truncation_counts(panel, hits):
        print(f"  {label}: {value}")
    print(f"\nPanel written to {config.PANEL_PATH}")
    print(f"Truncation check written to {config.TRUNCATION_CHECK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
