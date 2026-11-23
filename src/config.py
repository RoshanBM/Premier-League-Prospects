"""Project configuration: paths, seasons, thresholds and source URLs.

Every value that might need changing (a season range, a minutes threshold, a
download URL) lives here, so the rest of the code never hard-codes it.

Run ``python -m src.config`` from the repository root to print a summary.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------

def season_label(start_year: int) -> str:
    """Return the FPL-style label for the season that starts in ``start_year``.

    Example: ``season_label(2025)`` returns ``"2025-26"``.
    """
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_start_year(label: str) -> int:
    """Return the start year of an FPL-style season label.

    Example: ``season_start_year("2025-26")`` returns ``2025``. Useful when
    joining to a source that stores a season as its start year.

    Raises:
        ValueError: if the label is not of the form ``YYYY-YY`` with
            consecutive years.
    """
    error_message = f"Not a season label of the form YYYY-YY: {label!r}"
    try:
        start_text, end_text = label.split("-")
        start_year = int(start_text)
        end_two_digits = int(end_text)
    except ValueError as error:
        raise ValueError(error_message) from error
    if (
        len(start_text) != 4
        or len(end_text) != 2
        or (start_year + 1) % 100 != end_two_digits
    ):
        raise ValueError(error_message)
    return start_year


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------

# The "current" season: the last season that is complete in every source.
CURRENT_SEASON_START: int = 2025
CURRENT_SEASON: str = season_label(CURRENT_SEASON_START)  # "2025-26"

# FPL seasons used to build multi-season player histories. The archive also
# has 2026-27, but only up to gameweek 6, so it is left out.
FPL_FIRST_SEASON_START: int = 2016
FPL_SEASONS: tuple[str, ...] = tuple(
    season_label(year)
    for year in range(FPL_FIRST_SEASON_START, CURRENT_SEASON_START + 1)
)


def describe_seasons(seasons: Iterable[str]) -> str:
    """Describe FPL seasons as runs, e.g. "2016-17 to 2023-24, 2025-26" ("none" if empty)."""
    present = set(seasons)
    runs: list[list[str]] = []
    previous_present = False
    for season in FPL_SEASONS:
        if season in present:
            if previous_present:
                runs[-1][1] = season
            else:
                runs.append([season, season])
        previous_present = season in present
    return ", ".join(a if a == b else f"{a} to {b}" for a, b in runs) or "none"


# FPL element_type codes for players. 2024-25 also has element_type 5 for
# managers, who are left out everywhere (see the audit report).
FPL_POSITIONS: dict[int, str] = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# The repository root is the folder that contains src/.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
FPL_RAW_DIR: Path = RAW_DIR / "fpl"
TRANSFERMARKT_RAW_DIR: Path = RAW_DIR / "transfermarkt"
STATSBOMB_RAW_DIR: Path = RAW_DIR / "statsbomb"
PROCESSED_DIR: Path = DATA_DIR / "processed"

# One entry per downloaded file: source URL, download time, size and SHA-256.
MANIFEST_PATH: Path = RAW_DIR / "manifest.json"

# Hand-made fixes for FPL to Transfermarkt player matching. This is the only
# file under data/ that is committed to git (see .gitignore).
MANUAL_MATCHES_PATH: Path = DATA_DIR / "manual_matches.csv"

# Pipeline outputs (all gitignored). File names carry the season, e.g. 2025_26.
_SEASON_TAG: str = CURRENT_SEASON.replace("-", "_")


def u21_cohort_path(season: str) -> Path:
    """Where the U21 cohort of a season is written, e.g. u21_cohort_2018_19.csv."""
    return PROCESSED_DIR / f"u21_cohort_{season.replace('-', '_')}.csv"


AUDIT_REPORT_PATH: Path = PROCESSED_DIR / "audit_report.md"
MATCHES_PATH: Path = PROCESSED_DIR / "fpl_tm_matches.csv"
UNMATCHED_REPORT_PATH: Path = PROCESSED_DIR / "unmatched_report.csv"
MATCHING_REPORT_PATH: Path = PROCESSED_DIR / "matching_report.md"
U21_COHORT_PATH: Path = u21_cohort_path(CURRENT_SEASON)
PLAYER_SEASON_PATH: Path = PROCESSED_DIR / f"player_season_{_SEASON_TAG}.parquet"
U21_CARDS_PATH: Path = PROCESSED_DIR / f"u21_cards_{_SEASON_TAG}.parquet"

# Trajectories, ability, backtest and similar players.
PANEL_PATH: Path = PROCESSED_DIR / "player_season_panel.parquet"
TRAJECTORY_SUMMARY_PATH: Path = PROCESSED_DIR / "trajectory_model_summary.txt"
# The fitted model's numbers (fixed effects, covariances), so later steps can
# project without fitting again.
TRAJECTORY_PARAMS_PATH: Path = PROCESSED_DIR / "trajectory_model_params.json"
PA_PROJECTIONS_PATH: Path = PROCESSED_DIR / f"pa_projections_{_SEASON_TAG}.parquet"
CA_PATH: Path = PROCESSED_DIR / f"ca_{_SEASON_TAG}.parquet"
BACKTEST_RESULTS_PATH: Path = PROCESSED_DIR / "backtest_results.csv"
BACKTEST_PLAYERS_PATH: Path = PROCESSED_DIR / "backtest_players.csv"
BACKTEST_SUMMARY_PATH: Path = PROCESSED_DIR / "backtest_summary.md"
SIMILAR_PLAYERS_PATH: Path = PROCESSED_DIR / f"similar_players_{_SEASON_TAG}.parquet"
FIGURES_DIR: Path = PROCESSED_DIR / "figures"
# The Day 2 model (fitted on every player, before the survivorship fix), kept
# for comparison with the main model.
TRAJECTORY_ALL_PLAYERS_PARAMS_PATH: Path = PROCESSED_DIR / "trajectory_model_params_all_players.json"
# Scale factors that widen the projection intervals, estimated on the backtest.
CALIBRATION_PATH: Path = PROCESSED_DIR / "interval_calibration.json"
# Players whose Transfermarkt record shows PL minutes before the panel starts.
TRUNCATION_CHECK_PATH: Path = PROCESSED_DIR / "truncation_check.csv"


# ---------------------------------------------------------------------------
# Cohort and modelling thresholds
# ---------------------------------------------------------------------------

# Per-90 stats, and so attributes, only for player-seasons with at least this
# many PL minutes. Players below it are listed separately without attributes.
MIN_MINUTES_FOR_PER90: int = 450

# U21 means aged under 21 on the cut-off date, which is 1 September of the
# season's first year. A player born on 1 Sep 2004 turns 21 on the 2025-26
# cut-off date itself, so they are not U21; a player born on 2 Sep 2004 is.
U21_AGE_LIMIT: int = 21
U21_CUTOFF_MONTH: int = 9
U21_CUTOFF_DAY: int = 1


def u21_cutoff(season: str) -> date:
    """The U21 cut-off date of a season: 1 September of its first year.

    Example: ``u21_cutoff("2018-19")`` returns ``date(2018, 9, 1)``.
    """
    return date(season_start_year(season), U21_CUTOFF_MONTH, U21_CUTOFF_DAY)


U21_CUTOFF_DATE: date = u21_cutoff(CURRENT_SEASON)  # 1 Sep 2025

# Primary trajectory outcome: minutes share = PL minutes / (38 x 90).
MATCHES_PER_SEASON: int = 38
MINUTES_PER_MATCH: int = 90
MAX_SEASON_MINUTES: int = MATCHES_PER_SEASON * MINUTES_PER_MATCH  # 3420

# Football Manager-style attribute scale.
ATTRIBUTE_MIN: int = 1
ATTRIBUTE_MAX: int = 20

# Ages (inclusive) over which the projected peak is summarised.
PEAK_AGE_RANGE: tuple[int, int] = (24, 26)

# Seed for anything random, so results can be reproduced.
RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Trajectory model, projections and backtest
# ---------------------------------------------------------------------------

# "Age" in every model and window is age on 1 September of the season's first
# year. Windows such as PEAK_AGE_RANGE use completed years (so "age 24" means
# 24.0 to 24.99), which gives each player exactly one season per age.

# Once a player has PL minutes, a later season in which he is missing from FPL
# counts as a minutes share of 0, up to the current season or this age in
# completed years, whichever comes first.
ZERO_FILL_MAX_AGE: int = 27

# Age is centred at 21 in the model (age_c = age - 21), so the intercept is the
# expected minutes share of a 21-year-old.
MODEL_AGE_CENTRE: float = 21.0
# Ages (completed years, inclusive) of the seasons the model is fitted on.
MODEL_AGE_RANGE: tuple[int, int] = (17, 30)

# Monte Carlo draws for projection intervals, and the interval's coverage.
SIMULATION_DRAWS: int = 2000
INTERVAL_LEVEL: float = 0.80

# Backtest: forecast origins and the ages (completed years, inclusive) whose
# mean minutes share is predicted.
BACKTEST_ORIGINS: tuple[str, ...] = ("2017-18", "2018-19", "2019-20", "2020-21")
BACKTEST_TARGET_AGES: tuple[int, int] = (23, 25)

# Survivorship fix. The panel starts in 2016-17, so a player first seen in his
# twenties is only in it because he was still in the league (left truncation).
# The model is therefore fitted only on players "first seen young": completed
# age on 1 September of their first PL season with minutes at most this. It is
# the oldest U21 age, so the model learns from the same kind of player it
# projects.
FIRST_SEEN_YOUNG_MAX_AGE: int = U21_AGE_LIMIT - 1  # 20
# A backtest origin counts as "peak identified" only if the restricted training
# data has at least this many player-seasons at BACKTEST_TARGET_AGES. With
# fewer, the curve at those ages is an extrapolation (see docs/day2_log.md).
PEAK_IDENTIFIED_MIN_SEASONS: int = 50
# Transfermarkt Premier League seasons (start years) before the panel starts,
# used only to check how much left truncation remains.
PRE_PANEL_SEASON_STARTS: tuple[int, ...] = tuple(range(2012, FPL_FIRST_SEASON_START))

# Interval calibration: if the backtest's pooled coverage of the 80% interval
# is below this, intervals are widened by a factor estimated from past errors.
CALIBRATION_TRIGGER_COVERAGE: float = 0.70

# Which 80% interval the PA cards show: "raw" (as simulated) or "calibrated"
# (widened by the backtest factor). Both are always written. "raw" was chosen
# after seeing the backtest (a post-hoc decision, see docs/day2_log.md): the
# factor was learnt mostly from origins where the peak was not identified, and
# the one origin like 2025/26 (2020-21) already had 81% raw coverage.
CARD_INTERVAL: str = "raw"

# A PA resting on fewer career PL minutes than this is flagged as low evidence
# and shown greyed out (the same 450 minutes as the per-90 threshold).
PA_LOW_EVIDENCE_MINUTES: int = MIN_MINUTES_FOR_PER90


# ---------------------------------------------------------------------------
# Similar players (used by src/similarity.py)
# ---------------------------------------------------------------------------

SIMILAR_PLAYERS_TOP_N: int = 5
# Historical comparisons: earlier player-seasons within this many years of age.
HISTORICAL_AGE_WINDOW: float = 1.0


# ---------------------------------------------------------------------------
# FPL to Transfermarkt matching (used by src/matching.py)
# ---------------------------------------------------------------------------

# Name scores run from 0 to 100 (see matching.name_score).
# Rule 1: the birth dates are equal, so the name only has to agree.
MATCH_NAME_SCORE_WITH_BIRTH_DATE: int = 70
# Rule 2: no usable birth date, so the name must be close and the Premier
# League minutes, season by season, must agree (0 = no overlap, 1 = identical).
MATCH_NAME_SCORE_WITHOUT_BIRTH_DATE: int = 85
MATCH_MIN_MINUTES_AGREEMENT: float = 0.8
# Rule 3: a full name that only one Transfermarkt player has. No other player
# may score this high, and the player must be this old (years) in their first
# Premier League season.
MATCH_UNIQUE_NAME_RUNNER_UP: int = 90
MATCH_PLAUSIBLE_AGE: tuple[int, int] = (15, 40)
# When several candidates pass, a tie is only broken by a clear margin.
MATCH_DECISIVE_NAME_MARGIN: int = 5
MATCH_DECISIVE_MINUTES_MARGIN: float = 0.2
# A loose name score, used only to shortlist rule-2 candidates quickly.
MATCH_SHORTLIST_SCORE: int = 70


# ---------------------------------------------------------------------------
# Source URLs
# ---------------------------------------------------------------------------

# Fantasy Premier League archive (vaastav/Fantasy-Premier-League on GitHub).
FPL_ARCHIVE_BASE_URL: str = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
)
# Files downloaded for every season, as paths relative to the season folder.
FPL_SEASON_FILES: tuple[str, ...] = ("players_raw.csv", "gws/merged_gw.csv")

# Transfermarkt dataset (dcaribou/transfermarkt-datasets). The R2 address is
# the download location given in the dataset's README; Kaggle is the fallback.
TRANSFERMARKT_BASE_URL: str = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TRANSFERMARKT_KAGGLE_URL: str = "https://www.kaggle.com/datasets/davidcariboo/player-scores"
TRANSFERMARKT_TABLES: tuple[str, ...] = (
    "players",
    "appearances",
    "games",
    "competitions",
    "player_valuations",
    "clubs",
)


def fpl_url(season: str, relative_path: str) -> str:
    """Return the archive URL of one FPL file, e.g. ``players_raw.csv``."""
    return f"{FPL_ARCHIVE_BASE_URL}/{season}/{relative_path}"


def fpl_raw_path(season: str, relative_path: str) -> Path:
    """Return where an FPL file is stored locally.

    The local layout mirrors the archive, so ``gws/merged_gw.csv`` for 2025-26
    is saved as ``data/raw/fpl/2025-26/gws/merged_gw.csv``.
    """
    return FPL_RAW_DIR / season / relative_path


def transfermarkt_url(table: str) -> str:
    """Return the download URL of one Transfermarkt table."""
    return f"{TRANSFERMARKT_BASE_URL}/{table}.csv.gz"


def transfermarkt_raw_path(table: str) -> Path:
    """Return where a downloaded Transfermarkt table is stored (``{table}.csv.gz``)."""
    return TRANSFERMARKT_RAW_DIR / f"{table}.csv.gz"


def transfermarkt_manual_path(table: str) -> Path:
    """Return the accepted name for a Transfermarkt table copied in by hand.

    A manual download from Kaggle may give plain ``{table}.csv`` files rather
    than ``{table}.csv.gz``. Both are accepted.
    """
    return TRANSFERMARKT_RAW_DIR / f"{table}.csv"


# ---------------------------------------------------------------------------
# Download settings (used by src/load_data.py)
# ---------------------------------------------------------------------------

# Be polite to the hosts: one request at a time, a pause between files, a
# timeout, and up to three retries with exponential backoff (2, 4, 8 seconds).
REQUEST_TIMEOUT_SECONDS: tuple[float, float] = (10.0, 60.0)  # (connect, read)
REQUEST_MAX_RETRIES: int = 3
REQUEST_BACKOFF_SECONDS: float = 2.0
REQUEST_PAUSE_SECONDS: float = 1.0
# A generic client name with no personal details.
USER_AGENT: str = "premier-league-prospects/0.1 (non-commercial portfolio project)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """Create the data folders if they do not exist yet. Safe to call repeatedly."""
    for directory in (FPL_RAW_DIR, TRANSFERMARKT_RAW_DIR, STATSBOMB_RAW_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    """Return a short, readable summary of the main settings."""
    lines = [
        f"Project root:         {PROJECT_ROOT}",
        f"Current season:       {CURRENT_SEASON}",
        f"FPL seasons:          {FPL_SEASONS[0]} to {FPL_SEASONS[-1]} ({len(FPL_SEASONS)} seasons)",
        f"Per-90 threshold:     {MIN_MINUTES_FOR_PER90} PL minutes",
        f"U21 rule:             under {U21_AGE_LIMIT} on {U21_CUTOFF_DATE.isoformat()}",
        f"Minutes share basis:  {MAX_SEASON_MINUTES} minutes ({MATCHES_PER_SEASON} x {MINUTES_PER_MATCH})",
        f"Attribute scale:      {ATTRIBUTE_MIN} to {ATTRIBUTE_MAX}",
        f"Peak age range:       {PEAK_AGE_RANGE[0]} to {PEAK_AGE_RANGE[1]}",
        f"Model ages:           {MODEL_AGE_RANGE[0]} to {MODEL_AGE_RANGE[1]}, centred at {MODEL_AGE_CENTRE:g}",
        f"Zero-fill up to age:  {ZERO_FILL_MAX_AGE}",
        f"Simulation:           {SIMULATION_DRAWS} draws, {INTERVAL_LEVEL:.0%} intervals",
        f"Backtest origins:     {', '.join(BACKTEST_ORIGINS)} (target ages "
        f"{BACKTEST_TARGET_AGES[0]} to {BACKTEST_TARGET_AGES[1]})",
        f"Model fitted on:      players first seen at {FIRST_SEEN_YOUNG_MAX_AGE} or under",
        f"Peak identified if:   {PEAK_IDENTIFIED_MIN_SEASONS}+ training seasons at the target ages",
        f"Calibrate if:         pooled 80% coverage below {CALIBRATION_TRIGGER_COVERAGE:.0%}",
        f"PA cards show:        the {CARD_INTERVAL} 80% interval; low evidence under {PA_LOW_EVIDENCE_MINUTES} PL minutes",
        f"FPL files per season: {', '.join(FPL_SEASON_FILES)}",
        f"Transfermarkt tables: {', '.join(TRANSFERMARKT_TABLES)}",
        f"Example FPL URL:      {fpl_url(CURRENT_SEASON, FPL_SEASON_FILES[1])}",
        f"Example FPL path:     {fpl_raw_path(CURRENT_SEASON, FPL_SEASON_FILES[1])}",
        f"Example TM URL:       {transfermarkt_url(TRANSFERMARKT_TABLES[0])}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
