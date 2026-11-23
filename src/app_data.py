"""Data for the Streamlit app: which files it needs, loading them, and the tables each page shows.

Everything here is plain pandas, with no Streamlit, so it can be tested on its
own. The app (app/streamlit_app.py) reads only from the processed-data folder:
data/processed/ by default, or the folder named by the PROSPECTS_DATA_DIR
environment variable (used by the tests to point the app at an empty folder).

The main table, `players`, has one row per 2025/26 Premier League player with
any minutes, joining:
- player_season_2025_26.parquet: identity, minutes, attributes and per-90 inputs;
- ca_2025_26.parquet: Current Ability, its contributions and rank in position;
- pa_projections_2025_26.parquet: Potential Ability for the U21 cohort.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.ability import PA_STATUS_EXPERIMENTAL
from src.features import ATTRIBUTES

DATA_DIR_VARIABLE: str = "PROSPECTS_DATA_DIR"

# The files the app reads, by short name. All are written by run_pipeline.py.
REQUIRED_FILES: dict[str, str] = {
    "player_season": config.PLAYER_SEASON_PATH.name,
    "ca": config.CA_PATH.name,
    "pa": config.PA_PROJECTIONS_PATH.name,
    "similar": config.SIMILAR_PLAYERS_PATH.name,
    "panel": config.PANEL_PATH.name,
    "backtest": config.BACKTEST_RESULTS_PATH.name,
    "model": config.TRAJECTORY_PARAMS_PATH.name,
}

# Columns the app uses from each table. A test checks them against the real files.
EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "player_season": ("fpl_code", "name", "web_name", "position", "club", "age_on_cutoff", "exact_age_on_cutoff",
                      "u21", "minutes", "enough_minutes", *(a.column for a in ATTRIBUTES)),
    "ca": ("fpl_code", "position", "ca", "ca_rank_in_position", *(f"contrib_{a.column}" for a in ATTRIBUTES)),
    "pa": ("fpl_code", "pa", "pa_pct", "pa_label", "pa_status", "pa_low_evidence", "pa_display", "pa_range_text",
           "persistence_share", "peak_median_display", "pl_minutes_to_date", "history_seasons", "card_interval"),
    "similar": ("fpl_code", "comparison", "rank", "match_name", "match_season", "match_club", "match_age",
                "match_minutes", "similarity", "match_next_season_share", "match_peak_share", "match_peak_seasons"),
    "panel": ("fpl_code", "season", "season_year", "age", "age_completed", "minutes", "minutes_share", "filled",
              "position", "name", "club"),
    "backtest": ("origin", "subset", "method", "players", "mae", "rmse", "spearman", "mean_predicted",
                 "mean_actual", "bias", "coverage_80"),
}

# PA status as the shortlist shows it.
STATUS_EXPERIMENTAL: str = "experimental"
STATUS_LOW_EVIDENCE: str = "experimental, low evidence"
STATUS_TOO_YOUNG: str = "not enough data to project"
STATUS_NOT_U21: str = "not projected (not U21)"

# Backtest methods (as written by validate.py) and the names the app and README use.
METHOD_MODEL: str = "new model (first seen young)"
METHOD_PERSISTENCE: str = "persistence (mean share to date)"
METHOD_POPULATION: str = "population curve (new model's fixed effects)"
METHOD_DAY2: str = "Day 2 model (all players)"
METHOD_NAMES: dict[str, str] = {
    METHOD_MODEL: "Model (fitted on players first seen young)",
    METHOD_PERSISTENCE: "Minutes so far (persistence)",
    METHOD_POPULATION: "Average age curve only",
    METHOD_DAY2: "First model (fitted on all players)",
}
POOL_IDENTIFIED: str = "pooled, peak identified"
POOL_ALL: str = "pooled, all origins"


class MissingDataError(FileNotFoundError):
    """Raised when processed files the app needs are not there."""

    def __init__(self, missing: list[Path]):
        self.missing = missing
        super().__init__(f"{len(missing)} processed file(s) missing, for example {missing[0]}")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def data_dir() -> Path:
    """The processed-data folder: PROSPECTS_DATA_DIR if set, else data/processed/."""
    override = os.environ.get(DATA_DIR_VARIABLE)
    return Path(override) if override else config.PROCESSED_DIR


def file_path(name: str, directory: Path | None = None) -> Path:
    """Where one required file is, by its short name (a key of REQUIRED_FILES)."""
    return (directory or data_dir()) / REQUIRED_FILES[name]


def missing_files(directory: Path | None = None) -> list[Path]:
    """The required files that do not exist, in REQUIRED_FILES order."""
    return [file_path(name, directory) for name in REQUIRED_FILES if not file_path(name, directory).exists()]


def missing_columns(tables: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """For each table, the expected columns it lacks (only tables with gaps are listed)."""
    gaps = {name: [c for c in EXPECTED_COLUMNS[name] if c not in table.columns]
            for name, table in tables.items() if name in EXPECTED_COLUMNS}
    return {name: columns for name, columns in gaps.items() if columns}


@dataclass(frozen=True)
class AppData:
    """Every table the app shows, loaded once."""

    players: pd.DataFrame
    player_season: pd.DataFrame
    ca: pd.DataFrame
    pa: pd.DataFrame
    similar: pd.DataFrame
    panel: pd.DataFrame
    backtest: pd.DataFrame


def read_tables(directory: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read every required table (not the model file).

    Raises:
        MissingDataError: if any required file is missing.
        ValueError: if a table lacks a column the app uses.
    """
    missing = missing_files(directory)
    if missing:
        raise MissingDataError(missing)
    tables = {}
    for name in REQUIRED_FILES:
        path = file_path(name, directory)
        if path.suffix == ".parquet":
            tables[name] = pd.read_parquet(path)
        elif path.suffix == ".csv":
            tables[name] = pd.read_csv(path)
    gaps = missing_columns(tables)
    if gaps:
        raise ValueError(f"processed files are missing columns the app uses: {gaps}; run python run_pipeline.py")
    return tables


def load_app_data(directory: Path | None = None) -> AppData:
    """Read the tables and build the joined player table."""
    tables = read_tables(directory)
    return AppData(players=build_player_table(tables["player_season"], tables["ca"], tables["pa"]), **tables)


# ---------------------------------------------------------------------------
# The player table
# ---------------------------------------------------------------------------

def pa_status_label(pa: pd.DataFrame) -> pd.Series:
    """PA status as the shortlist shows it, for rows of the joined table (missing pa_status = not U21)."""
    status = pa["pa_status"]
    labels = np.select(
        [status.isna(), status != PA_STATUS_EXPERIMENTAL, pa["pa_low_evidence"].fillna(False).astype(bool)],
        [STATUS_NOT_U21, STATUS_TOO_YOUNG, STATUS_LOW_EVIDENCE],
        default=STATUS_EXPERIMENTAL,
    )
    return pd.Series(labels, index=pa.index, dtype="str")


def build_player_table(player_season: pd.DataFrame, ca: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    """One row per 2025/26 player with PL minutes: identity, attributes, CA and PA fields.

    `in_cohort` marks the 2025/26 U21 cohort (the players with a PA row).
    `ca_pool` is the number of players at the position with a CA (450+ minutes).
    """
    played = player_season[player_season["minutes"] > 0]
    identity = ["fpl_code", "name", "web_name", "position", "club", "age_on_cutoff", "exact_age_on_cutoff", "u21",
                "minutes", "enough_minutes", *(a.column for a in ATTRIBUTES)]
    ca_columns = ["fpl_code", "ca", "ca_rank_in_position", *(f"contrib_{a.column}" for a in ATTRIBUTES)]
    pa_columns = [c for c in EXPECTED_COLUMNS["pa"]]
    table = (played[identity]
             .merge(ca[ca_columns], on="fpl_code", how="left", validate="one_to_one")
             .merge(pa[pa_columns], on="fpl_code", how="left", validate="one_to_one"))
    table["in_cohort"] = table["fpl_code"].isin(pa["fpl_code"])
    table["ca_pool"] = table["position"].map(ca.groupby("position")["ca"].count())
    table["pa_status_label"] = pa_status_label(table)
    return table.reset_index(drop=True)


PLAYER_QUERY_KEY: str = "player"


def player_from_query(value: str | list[str] | None, valid_codes) -> int | None:
    """The FPL code in a URL's player parameter (for example ?player=487838).

    Returns None if the parameter is missing, is not a whole number, or is not
    a player the app knows, so a bad or old link falls back to the default player.
    """
    if isinstance(value, list):
        value = value[-1] if value else None
    if value is None:
        return None
    try:
        code = int(str(value).strip())
    except ValueError:
        return None
    return code if code in valid_codes else None


def player_label(row: pd.Series) -> str:
    """How a player appears in the player picker: "Lewis Hall (Newcastle, DEF, 20)"."""
    club = row["club"] if isinstance(row["club"], str) and row["club"] else "no club listed"
    age = f", {int(row['age_on_cutoff'])}" if pd.notna(row["age_on_cutoff"]) else ""
    return f"{row['name']} ({club}, {row['position']}{age})"


# ---------------------------------------------------------------------------
# The shortlist
# ---------------------------------------------------------------------------

SORT_CA: str = "Current Ability"
SORT_PA: str = "Potential Ability rank"
SORT_MINUTES: str = "Minutes"
SORT_OPTIONS: tuple[str, ...] = (SORT_CA, SORT_PA, SORT_MINUTES)


@dataclass(frozen=True)
class ShortlistFilters:
    """The shortlist's filters. Ranges are inclusive.

    A CA range narrower than the full 1 to 20 drops players with no CA (under
    450 minutes); the full range keeps them.
    """

    ages: tuple[int, int] = (15, 40)
    positions: tuple[str, ...] = tuple(config.FPL_POSITIONS.values())
    min_minutes: int = 0
    ca_range: tuple[float, float] = (float(config.ATTRIBUTE_MIN), float(config.ATTRIBUTE_MAX))
    u21_only: bool = True
    hide_low_evidence: bool = False
    sort_by: str = SORT_CA


def filter_players(players: pd.DataFrame, filters: ShortlistFilters) -> pd.DataFrame:
    """The players that pass the filters, sorted (highest first, missing values last)."""
    keep = players["position"].isin(filters.positions) & (players["minutes"] >= filters.min_minutes)
    keep &= players["age_on_cutoff"].between(*filters.ages).fillna(False).astype(bool)
    if filters.u21_only:
        keep &= players["in_cohort"]
    if filters.hide_low_evidence:
        keep &= ~players["pa_low_evidence"].fillna(False).astype(bool)
    full_ca_range = filters.ca_range == (float(config.ATTRIBUTE_MIN), float(config.ATTRIBUTE_MAX))
    if not full_ca_range:
        keep &= players["ca"].between(*filters.ca_range).fillna(False).astype(bool)
    chosen = players[keep]
    order = {
        SORT_CA: ["ca", "minutes"],
        SORT_PA: ["pa", "peak_median_display", "ca"],
        SORT_MINUTES: ["minutes", "ca"],
    }[filters.sort_by]
    return chosen.sort_values(order, ascending=False, na_position="last", kind="mergesort").reset_index(drop=True)


SHORTLIST_COLUMNS: dict[str, str] = {
    "name": "Name",
    "club": "Club",
    "position": "Position",
    "age_on_cutoff": "Age",
    "minutes": "Minutes 2025/26",
    "ca": "Current Ability",
    "pa": "PA rank",
    "persistence_pct": "Minutes share so far",
    "pa_status_label": "PA status",
}


def shortlist_table(chosen: pd.DataFrame) -> pd.DataFrame:
    """The shortlist columns, renamed for display; minutes share as a percentage."""
    table = chosen.assign(persistence_pct=chosen["persistence_share"] * 100)
    return table[list(SHORTLIST_COLUMNS)].rename(columns=SHORTLIST_COLUMNS)


# ---------------------------------------------------------------------------
# Player card pieces
# ---------------------------------------------------------------------------

MEASURE_COLUMNS: dict[str, str] = {
    "goals": "goals_per90", "goals_minus_xg": "goals_minus_xg_per90", "xg": "xg_per90", "xa": "xa_per90",
    "assists": "assists_per90", "defensive_actions": "defensive_actions_per90", "recoveries": "recoveries_per90",
    "saves": "saves_per90", "goals_prevented": "goals_prevented_per90",
}


def two_decimals(value: float) -> str:
    """A number to two decimals, without a minus sign on zero: -0.001 -> "0.00"."""
    return f"{round(float(value), 2) + 0.0:.2f}"


def attribute_rows(player: pd.Series, player_season: pd.DataFrame) -> pd.DataFrame:
    """The player's attributes with the per-90 numbers behind each, for the card's table.

    Only attributes the player has are listed (none under 450 minutes).
    """
    from src.features import MEASURE_LABELS

    season_row = player_season[player_season["fpl_code"] == player["fpl_code"]].iloc[0]
    rows = []
    for attribute in ATTRIBUTES:
        value = season_row.get(attribute.column)
        if pd.isna(value):
            continue
        inputs = "; ".join(f"{MEASURE_LABELS[m]} {two_decimals(season_row[MEASURE_COLUMNS[m]])}"
                           for m in attribute.inputs)
        rows.append({"Attribute": attribute.name, "Rating (1 to 20)": int(value), "Based on (per 90)": inputs})
    return pd.DataFrame(rows, columns=["Attribute", "Rating (1 to 20)", "Based on (per 90)"])


def ca_contributions(player: pd.Series) -> pd.DataFrame:
    """Each weighted attribute's weight, rating and contribution to CA, largest first."""
    from src.ability import CA_WEIGHTS

    rows = []
    for attribute in ATTRIBUTES:
        weight = CA_WEIGHTS.get(player["position"], {}).get(attribute.column)
        if weight is None or pd.isna(player.get(attribute.column)):
            continue
        rows.append({"attribute": attribute.name, "weight": weight.value, "rating": int(player[attribute.column]),
                     "contribution": float(player[f"contrib_{attribute.column}"]), "reason": weight.reason})
    table = pd.DataFrame(rows, columns=["attribute", "weight", "rating", "contribution", "reason"])
    return table.sort_values("contribution", ascending=False, kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Similar players
# ---------------------------------------------------------------------------

def current_matches(similar: pd.DataFrame, code: int) -> pd.DataFrame:
    """The closest current players for one U21, for display."""
    rows = similar[(similar["fpl_code"] == code) & (similar["comparison"] == "current")].sort_values("rank")
    return pd.DataFrame({
        "Rank": rows["rank"].astype(int),
        "Player": rows["match_name"],
        "Club": rows["match_club"].fillna(""),
        "Age": rows["match_age"].astype(float).round(1),
        "Minutes 2025/26": rows["match_minutes"].astype(int),
        "Similarity": rows["similarity"].astype(float),
    }).reset_index(drop=True)


def peak_text(share: float, seasons: int) -> str:
    """What a historical match went on to play at 24 to 26, in words."""
    if seasons <= 0 or pd.isna(share):
        return "not reached by 2025/26"
    noun = "season" if seasons == 1 else "seasons"
    return f"{share:.0%} of league minutes ({seasons} {noun})"


def historical_matches(similar: pd.DataFrame, code: int) -> pd.DataFrame:
    """The closest earlier players at the same age for one U21, with what they went on to play."""
    rows = similar[(similar["fpl_code"] == code) & (similar["comparison"] == "historical")].sort_values("rank")
    next_share = rows["match_next_season_share"].astype(float)
    return pd.DataFrame({
        "Rank": rows["rank"].astype(int),
        "Player": rows["match_name"],
        "Season": rows["match_season"],
        "Club then": rows["match_club"].fillna(""),
        "Age then": rows["match_age"].astype(float).round(1),
        "Minutes then": rows["match_minutes"].astype(int),
        "Similarity": rows["similarity"].astype(float),
        "Next season": [f"{s:.0%} of league minutes" if pd.notna(s) else "not in the data" for s in next_share],
        "At 24 to 26": [peak_text(s, int(n)) for s, n in zip(rows["match_peak_share"], rows["match_peak_seasons"])],
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# The backtest, for the About page
# ---------------------------------------------------------------------------

def backtest_row(backtest: pd.DataFrame, pool: str, method: str, subset: str = "all") -> pd.Series:
    """One row of backtest_results.csv.

    Raises:
        KeyError: if there is no such row.
    """
    hits = backtest[(backtest["origin"] == pool) & (backtest["method"] == method) & (backtest["subset"] == subset)]
    if len(hits) != 1:
        raise KeyError(f"expected one backtest row for {pool!r}, {method!r}, {subset!r}; found {len(hits)}")
    return hits.iloc[0]


def backtest_table(backtest: pd.DataFrame) -> pd.DataFrame:
    """The headline results: two pools by four methods, for display."""
    rows = []
    for pool, pool_name in ((POOL_IDENTIFIED, "Peak identified (2020-21 origin)"), (POOL_ALL, "All four origins")):
        for method, name in METHOD_NAMES.items():
            row = backtest_row(backtest, pool, method)
            rows.append({"Test set": pool_name, "Method": name, "Players": int(row["players"]),
                         "Average error (MAE)": row["mae"], "Ranking (Spearman)": row["spearman"],
                         "Bias": row["bias"]})
    return pd.DataFrame(rows)
