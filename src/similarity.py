"""Similar players: who a 2025/26 U21 resembles now, and who he resembles from earlier seasons at the same age.

Both kinds of comparison work the same way:
1. Take per-90 stats (and, for the historical kind, minutes share).
2. Standardise each stat within position (z-scores: minus the mean, divided
   by the standard deviation), so a stat measured in big numbers does not
   swamp one measured in small numbers, and so a midfielder is measured
   against midfielders.
3. Compare players by cosine similarity of their standardised stats: 1 means
   the same profile (the same mix of strengths and weaknesses relative to the
   position), 0 unrelated, -1 the opposite. Only players at the same position
   are compared, and only on the same stat set.

Current comparisons (2025/26 only):
    Every 2025/26 player with at least 450 minutes is a candidate. Outfield
    players are compared on goals, xG, assists, xA, tackles, clearances /
    blocks / interceptions and recoveries per 90 (the raw counts behind the
    attributes, without "goals minus xG", which would count goals and xG
    twice). Goalkeepers are compared on saves, goals conceded and xG
    conceded per 90.

Historical comparisons ("at the same age"):
    Candidates are player-seasons from 2016-17 to 2024-25 with at least 450
    minutes, at the same position and within one year of the U21's age (both
    on 1 September). The stats are the only ones every season has: minutes
    share and goals, assists, influence, creativity and threat per 90.
    Influence, creativity and threat are FPL composites with an unpublished
    formula. These stats are standardised within position *and season*, so
    a change in how a season was scored does not make players look alike.
    The U21's own earlier seasons are left out, and each comparison player
    appears once (his best-matching season). For each one, the panel shows
    what happened next: his minutes share the following season, and his mean
    share at ages 24 to 26 where those seasons exist by 2025-26.

Writes data/processed/similar_players_2025_26.parquet, one row per U21
player per comparison.

Run from the repository root, after python -m src.features and python -m src.panel:
    python -m src.similarity
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src import config
from src.load_data import read_fpl_players
from src.panel import read_panel

CURRENT_OUTFIELD_STATS: tuple[str, ...] = (
    "goals_scored", "expected_goals", "assists", "expected_assists", "tackles",
    "clearances_blocks_interceptions", "recoveries",
)
CURRENT_GOALKEEPER_STATS: tuple[str, ...] = ("saves", "goals_conceded", "expected_goals_conceded")
HISTORICAL_COUNTS: tuple[str, ...] = ("goals_scored", "assists", "influence", "creativity", "threat")
HISTORICAL_FEATURES: tuple[str, ...] = ("minutes_share", *(f"{c}_per90" for c in HISTORICAL_COUNTS))
CURRENT = "current"
HISTORICAL = "historical"
OUTPUT_COLUMNS: list[str] = [
    "fpl_code", "name", "web_name", "position", "club", "age", "minutes", "comparison", "rank",
    "match_fpl_code", "match_name", "match_season", "match_club", "match_age", "match_minutes", "similarity",
    "match_next_season_share", "match_peak_share", "match_peak_seasons",
]


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def standardise(frame: pd.DataFrame, columns: list[str], by: list[str]) -> pd.DataFrame:
    """z-scores of each column within groups (population standard deviation).

    A stat with no spread in a group (for example goalkeepers' goals) gets 0
    for everyone in it, so it adds nothing to the comparison.
    """
    grouped = frame.groupby(by)[columns]
    mean = grouped.transform("mean")
    std = grouped.transform(lambda values: values.std(ddof=0))
    return ((frame[columns] - mean) / std.where(std > 0)).fillna(0.0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of a and every row of b; missing where a row is all zeros."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a / norm_a) @ (b / norm_b).T
    return np.clip(result, -1.0, 1.0)


def best_matches(query: pd.Series, candidates: pd.DataFrame, columns: list[str], n: int) -> pd.DataFrame:
    """The n candidates most similar to the query, one row per candidate player (his best row).

    Ties are broken by fpl_code so the result does not depend on row order.
    """
    if candidates.empty:
        return candidates.assign(similarity=pd.Series(dtype=float))
    scores = cosine_similarity(query[columns].to_numpy(dtype=float), candidates[columns].to_numpy(dtype=float))[0]
    scored = candidates.assign(similarity=scores).dropna(subset=["similarity"])
    scored = scored.sort_values(["similarity", "fpl_code"], ascending=[False, True])
    return scored.drop_duplicates("fpl_code").head(n)


def similar_players(
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    columns: list[str],
    n: int = config.SIMILAR_PLAYERS_TOP_N,
    age_window: float | None = None,
    before_season: int | None = None,
) -> pd.DataFrame:
    """For each query player, the n most similar candidates at the same position.

    Args:
        queries: one row per player to find matches for, with fpl_code,
            position, age and the (already standardised) columns.
        candidates: rows to search, with the same columns plus season_year.
        columns: the stats to compare on.
        n: how many matches per player.
        age_window: if given, only candidates within this many years of the
            query's age.
        before_season: if given, only candidates from seasons starting before
            this year.

    Returns:
        One row per match: the query's fpl_code, rank (1 = most similar),
        similarity and the candidate's columns prefixed with "match_".
    """
    rows = []
    for _, query in queries.iterrows():
        pool = candidates[(candidates["position"] == query["position"]) & (candidates["fpl_code"] != query["fpl_code"])]
        if age_window is not None:
            pool = pool[(pool["age"] - query["age"]).abs() <= age_window]
        if before_season is not None:
            pool = pool[pool["season_year"] < before_season]
        matches = best_matches(query, pool, columns, n)
        renamed = matches.add_prefix("match_").rename(columns={"match_similarity": "similarity"})
        renamed.insert(0, "fpl_code", query["fpl_code"])
        renamed.insert(1, "rank", range(1, len(renamed) + 1))
        rows.append(renamed)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Current comparisons
# ---------------------------------------------------------------------------

def current_features(player_season: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """2025/26 players with enough minutes, with per-90 stats standardised within position.

    Returns the table and the stat columns used for goalkeepers and for outfield players.
    """
    pool = player_season[player_season["enough_minutes"]].copy()
    pool["age"] = pool["exact_age_on_cutoff"]
    pool["season_year"] = config.CURRENT_SEASON_START
    stat_sets = {"GK": list(CURRENT_GOALKEEPER_STATS), "outfield": list(CURRENT_OUTFIELD_STATS)}
    for stat in {*CURRENT_GOALKEEPER_STATS, *CURRENT_OUTFIELD_STATS}:
        pool[f"{stat}_per90"] = pool[stat] / pool["minutes"] * 90
    columns = {kind: [f"z_{s}" for s in stats] for kind, stats in stat_sets.items()}
    for kind, stats in stat_sets.items():
        rows = (pool["position"] == "GK") if kind == "GK" else (pool["position"] != "GK")
        z = standardise(pool.loc[rows], [f"{s}_per90" for s in stats], ["position"])
        pool.loc[rows, columns[kind]] = z.to_numpy()
    return pool, columns


def current_comparisons(player_season: pd.DataFrame, n: int = config.SIMILAR_PLAYERS_TOP_N) -> pd.DataFrame:
    """Top n 2025/26 players at the same position for each U21 with enough minutes."""
    pool, columns = current_features(player_season)
    queries = pool[pool["u21"].fillna(False).astype(bool)]
    parts = []
    for kind, stat_columns in columns.items():
        is_gk = kind == "GK"
        chosen = queries[(queries["position"] == "GK") == is_gk]
        candidates = pool[(pool["position"] == "GK") == is_gk]
        parts.append(similar_players(chosen, candidates, stat_columns, n))
    matches = pd.concat(parts, ignore_index=True)
    matches["comparison"] = CURRENT
    matches["match_season"] = config.CURRENT_SEASON
    return matches


# ---------------------------------------------------------------------------
# Historical comparisons
# ---------------------------------------------------------------------------

def historical_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Every listed player-season with enough minutes, with stats every season has, standardised.

    Standardised within position and season.
    """
    raw = []
    for season in config.FPL_SEASONS:
        players = read_fpl_players(season)[["code", *HISTORICAL_COUNTS]].rename(columns={"code": "fpl_code"})
        raw.append(players.assign(season=season))
    stats = pd.concat(raw, ignore_index=True)
    observed = panel[~panel["filled"]].merge(stats, on=["fpl_code", "season"], how="left", validate="one_to_one")
    pool = observed[(observed["minutes"] >= config.MIN_MINUTES_FOR_PER90) & observed["age"].notna()].copy()
    for count in HISTORICAL_COUNTS:
        pool[f"{count}_per90"] = pool[count] / pool["minutes"] * 90
    z_columns = [f"z_{c}" for c in HISTORICAL_FEATURES]
    pool[z_columns] = standardise(pool, list(HISTORICAL_FEATURES), ["position", "season"]).to_numpy()
    return pool.reset_index(drop=True)


def later_outcomes(panel: pd.DataFrame, ages: tuple[int, int] = config.PEAK_AGE_RANGE) -> pd.DataFrame:
    """For each player-season: the player's share the next season and his mean share at peak ages.

    Uses the panel, so a season out of the league counts as 0. Missing where
    the next season, or every peak-age season, is after 2025-26 (or before
    the panel starts). match_peak_seasons says how many peak-age seasons the
    mean is over.
    """
    low, high = ages
    share = panel.set_index(["fpl_code", "season_year"])["minutes_share"]
    peak = panel[panel["age_completed"].between(low, high).fillna(False).astype(bool)]
    peak = peak.groupby("fpl_code").agg(peak_share=("minutes_share", "mean"), peak_seasons=("season", "size"))
    out = panel[["fpl_code", "season_year"]].copy()
    next_keys = pd.MultiIndex.from_arrays([out["fpl_code"], out["season_year"] + 1])
    out["next_season_share"] = share.reindex(next_keys).to_numpy()
    out = out.join(peak, on="fpl_code")
    out["peak_seasons"] = out["peak_seasons"].fillna(0).astype(int)
    return out


def historical_comparisons(panel: pd.DataFrame, n: int = config.SIMILAR_PLAYERS_TOP_N) -> pd.DataFrame:
    """Top n earlier player-seasons at the same position and within a year of age, for each U21 with enough minutes."""
    pool = historical_features(panel)
    queries = pool[(pool["season"] == config.CURRENT_SEASON) & (pool["age_completed"] < config.U21_AGE_LIMIT)]
    columns = [f"z_{c}" for c in HISTORICAL_FEATURES]
    matches = similar_players(queries, pool, columns, n, age_window=config.HISTORICAL_AGE_WINDOW,
                              before_season=config.CURRENT_SEASON_START)
    outcomes = later_outcomes(panel).rename(columns=lambda c: f"match_{c}")
    matches = matches.merge(outcomes, on=["match_fpl_code", "match_season_year"], how="left", validate="many_to_one")
    matches["comparison"] = HISTORICAL
    return matches


# ---------------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------------

def build_similar_players(player_season: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Both kinds of comparison for every 2025/26 U21 with enough minutes, in one table."""
    combined = pd.concat([current_comparisons(player_season), historical_comparisons(panel)], ignore_index=True)
    identity = player_season.set_index("fpl_code")[["name", "web_name", "position", "club", "exact_age_on_cutoff", "minutes"]]
    combined = combined.join(identity.rename(columns={"exact_age_on_cutoff": "age"}), on="fpl_code")
    for column in OUTPUT_COLUMNS:
        if column not in combined.columns:
            combined[column] = np.nan
    combined = combined[OUTPUT_COLUMNS]
    if not (combined["position"] == combined["fpl_code"].map(identity["position"])).all():
        raise ValueError("a comparison crossed positions")
    return combined.sort_values(["fpl_code", "comparison", "rank"], ignore_index=True)


def main() -> int:
    """Build the comparisons, write them and print a few."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    config.ensure_directories()
    pd.set_option("display.width", 250)
    player_season = pd.read_parquet(config.PLAYER_SEASON_PATH)
    panel = read_panel()
    table = build_similar_players(player_season, panel)
    table.to_parquet(config.SIMILAR_PLAYERS_PATH, index=False)

    for kind in (CURRENT, HISTORICAL):
        part = table[table["comparison"] == kind]
        print(f"{kind}: {part['fpl_code'].nunique()} U21 players, {len(part)} matches; similarity "
              f"median {part['similarity'].median():.3f}, range {part['similarity'].min():.3f} to {part['similarity'].max():.3f}")
    historical = table[table["comparison"] == HISTORICAL]
    print(f"historical matches with a peak-age outcome: {int((historical['match_peak_seasons'] > 0).sum())} of {len(historical)}")
    shown = ["web_name", "comparison", "rank", "match_name", "match_season", "match_age", "similarity",
             "match_next_season_share", "match_peak_share", "match_peak_seasons"]
    for web_name in table["web_name"].drop_duplicates().head(3):
        print()
        print(table[table["web_name"] == web_name][shown].round(3).to_string(index=False))
    print(f"\nSimilar players written to {config.SIMILAR_PLAYERS_PATH} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
