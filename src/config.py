"""Project configuration: paths, seasons, thresholds and source URLs.

Every value that might need changing (a season range, a minutes threshold, a
download URL) lives here, so the rest of the code never hard-codes it.

Run ``python -m src.config`` from the repository root to print a summary.
"""

from __future__ import annotations

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
AUDIT_REPORT_PATH: Path = PROCESSED_DIR / "audit_report.md"
UNMATCHED_REPORT_PATH: Path = PROCESSED_DIR / "unmatched_report.csv"
PLAYER_SEASON_PATH: Path = PROCESSED_DIR / f"player_season_{_SEASON_TAG}.parquet"
U21_CARDS_PATH: Path = PROCESSED_DIR / f"u21_cards_{_SEASON_TAG}.parquet"


# ---------------------------------------------------------------------------
# Cohort and modelling thresholds
# ---------------------------------------------------------------------------

# Per-90 stats, and so attributes, only for player-seasons with at least this
# many PL minutes. Players below it are listed separately without attributes.
MIN_MINUTES_FOR_PER90: int = 450

# U21 means aged under 21 on the cut-off date. A player born on 1 Sep 2004
# turns 21 on the cut-off date itself, so they are not U21; a player born on
# 2 Sep 2004 is.
U21_CUTOFF_DATE: date = date(2025, 9, 1)
U21_AGE_LIMIT: int = 21

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
    """Return where a Transfermarkt table is stored locally."""
    return TRANSFERMARKT_RAW_DIR / f"{table}.csv.gz"


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
        f"FPL files per season: {', '.join(FPL_SEASON_FILES)}",
        f"Transfermarkt tables: {', '.join(TRANSFERMARKT_TABLES)}",
        f"Example FPL URL:      {fpl_url(CURRENT_SEASON, FPL_SEASON_FILES[1])}",
        f"Example FPL path:     {fpl_raw_path(CURRENT_SEASON, FPL_SEASON_FILES[1])}",
        f"Example TM URL:       {transfermarkt_url(TRANSFERMARKT_TABLES[0])}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
