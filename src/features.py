"""Per-90 stats, within-position percentiles and Football Manager-style 1 to 20 attributes.

For the current season (config.CURRENT_SEASON):
1. Season totals come from FPL's players_raw.csv. They match the summed
   gameweek rows for every input used here, once exact duplicate gameweek
   rows are removed.
2. Per-90 stats are worked out only for players with at least
   config.MIN_MINUTES_FOR_PER90 minutes; below that they are left missing.
3. Each per-90 stat becomes a percentile within the player's FPL position
   (GK, DEF, MID or FWD), against every Premier League player at that
   position who reached the threshold, not only U21s, so ratings are absolute
   as in Football Manager. See mid_rank_percentile for how ties are treated.
4. attribute = min(20, 1 + floor(percentile x 20)). An attribute with two
   inputs uses the average of their two percentiles.

The attributes are the only ones the open data supports for 2025/26
(CLAUDE.md section 5.2): Finishing, Shot Threat, Chance Creation, Defending
and Ball Recovery for outfield players, and Goalkeeping for goalkeepers.

Writes:
- data/processed/player_season_2025_26.parquet: every player in the season.
- data/processed/u21_cards_2025_26.parquet: the U21 cohort, with attributes
  for those with enough minutes.

Run from the repository root, after python -m src.matching:
    python -m src.features
    python -m src.features --show Diouf Estêvão Kroupi   # also print these players' cards
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.clean import normalise_name
from src.cohort import current_season_players
from src.load_data import read_fpl_players

# FPL season totals kept in the output, under their FPL names.
RAW_STATS: tuple[str, ...] = (
    "goals_scored", "assists", "expected_goals", "expected_assists", "tackles",
    "clearances_blocks_interceptions", "recoveries", "saves", "goals_conceded",
    "expected_goals_conceded",
)
# Added before rounding down, so a percentile that is exactly on a boundary
# (say 0.5, which should give 11) is not pushed to the value below by
# floating-point error (0.49999999999999994). Real differences between
# percentiles here are far larger than this.
FLOAT_TOLERANCE: float = 1e-9
IDENTITY_COLUMNS: list[str] = [
    "fpl_code", "name", "web_name", "position", "club", "birth_date", "birth_date_source",
    "age_on_cutoff", "exact_age_on_cutoff", "u21", "tm_player_id", "minutes", "starts", "has_attributes",
]


@dataclass(frozen=True)
class Attribute:
    """One attribute: its inputs (per-90 measures) and which players it applies to."""

    name: str
    inputs: tuple[str, ...]
    goalkeepers_only: bool = False

    @property
    def column(self) -> str:
        """Column name, e.g. "chance_creation"."""
        return self.name.lower().replace(" ", "_")

    def applies_to(self, positions: pd.Series) -> pd.Series:
        """True for the players this attribute is given to (goalkeepers, or outfield players)."""
        is_goalkeeper = positions == "GK"
        return is_goalkeeper if self.goalkeepers_only else ~is_goalkeeper


MEASURE_LABELS: dict[str, str] = {
    "goals": "goals",
    "goals_minus_xg": "goals minus xG",
    "xg": "xG",
    "xa": "xA",
    "assists": "assists",
    "defensive_actions": "tackles + clearances/blocks/interceptions",
    "recoveries": "recoveries",
    "saves": "saves",
    "goals_prevented": "xGC minus goals conceded",
}

ATTRIBUTES: tuple[Attribute, ...] = (
    Attribute("Finishing", ("goals", "goals_minus_xg")),
    Attribute("Shot Threat", ("xg",)),
    Attribute("Chance Creation", ("xa", "assists")),
    Attribute("Defending", ("defensive_actions",)),
    Attribute("Ball Recovery", ("recoveries",)),
    Attribute("Goalkeeping", ("saves", "goals_prevented"), goalkeepers_only=True),
)


# ---------------------------------------------------------------------------
# The calculations
# ---------------------------------------------------------------------------

def measure_totals(players: pd.DataFrame) -> pd.DataFrame:
    """Season totals of every attribute input, from FPL's column names.

    goals_prevented is expected goals conceded minus goals conceded, so a
    positive value means a goalkeeper let in fewer goals than expected and
    higher is better, like every other input.
    """
    return pd.DataFrame(
        {
            "goals": players["goals_scored"],
            "goals_minus_xg": players["goals_scored"] - players["expected_goals"],
            "xg": players["expected_goals"],
            "xa": players["expected_assists"],
            "assists": players["assists"],
            "defensive_actions": players["tackles"] + players["clearances_blocks_interceptions"],
            "recoveries": players["recoveries"],
            "saves": players["saves"],
            "goals_prevented": players["expected_goals_conceded"] - players["goals_conceded"],
        },
        index=players.index,
    )


def per_90(totals: pd.Series, minutes: pd.Series, min_minutes: int = config.MIN_MINUTES_FOR_PER90) -> pd.Series:
    """Totals per 90 minutes; missing for players with fewer than min_minutes minutes."""
    return (totals / minutes * 90).where(minutes >= min_minutes)


def mid_rank_percentile(values: pd.Series) -> pd.Series:
    """Percentile of each value among the non-missing values, from 0 to 1.

    It is the share of values that are lower, plus half the share that are
    equal (the mid-rank). Tied players get the same percentile, the median
    player gets 0.5, the best gets just under 1 and the worst just over 0.
    Missing values stay missing.
    """
    ranks = values.rank(method="average")
    return (ranks - 0.5) / values.notna().sum()


def attribute_score(percentiles: pd.Series) -> pd.Series:
    """Map percentiles (0 to 1) to the 1 to 20 scale: min(20, 1 + floor(percentile x 20)).

    Raises ValueError if a percentile is outside 0 to 1. Missing stays missing.
    """
    if ((percentiles < 0) | (percentiles > 1)).any():
        raise ValueError("percentiles must lie between 0 and 1")
    steps = config.ATTRIBUTE_MAX - config.ATTRIBUTE_MIN + 1
    scores = np.minimum(
        config.ATTRIBUTE_MAX, config.ATTRIBUTE_MIN + np.floor(percentiles * steps + FLOAT_TOLERANCE)
    )
    return pd.Series(scores, index=percentiles.index).astype("Int64")


def add_features(players: pd.DataFrame) -> pd.DataFrame:
    """Add per-90 values, percentiles and attributes to one row per player."""
    out = players.copy()
    totals = measure_totals(out)
    measures_for = {}  # measure name -> the attribute rule deciding who gets a percentile for it
    for attribute in ATTRIBUTES:
        for measure in attribute.inputs:
            measures_for[measure] = attribute
    for measure, attribute in measures_for.items():
        rate = per_90(totals[measure], out["minutes"])
        out[f"{measure}_per90"] = rate
        eligible = rate.where(attribute.applies_to(out["position"]))
        out[f"{measure}_pct"] = eligible.groupby(out["position"]).transform(mid_rank_percentile)
    for attribute in ATTRIBUTES:
        averaged = out[[f"{measure}_pct" for measure in attribute.inputs]].mean(axis=1, skipna=False)
        out[attribute.column] = attribute_score(averaged)
    return out


# ---------------------------------------------------------------------------
# Building the tables
# ---------------------------------------------------------------------------

def build_player_season() -> pd.DataFrame:
    """Every player in the current season: identity, totals, per-90 values, percentiles and attributes."""
    players = current_season_players()
    raw = read_fpl_players(config.CURRENT_SEASON)[["code", *RAW_STATS]].rename(columns={"code": "fpl_code"})
    players = players.merge(raw, on="fpl_code", how="left", validate="one_to_one")
    featured = add_features(players)
    measure_columns = [c for c in featured.columns if c.endswith(("_per90", "_pct"))]
    columns = [*IDENTITY_COLUMNS, *RAW_STATS, *[a.column for a in ATTRIBUTES], *measure_columns]
    return featured[columns].sort_values(["position", "minutes"], ascending=[True, False], ignore_index=True)


def build_u21_cards(player_season: pd.DataFrame) -> pd.DataFrame:
    """The U21 cohort (U21 with any PL minutes); attributes only for those with enough minutes."""
    cohort = player_season[player_season["u21"].fillna(False).astype(bool) & (player_season["minutes"] > 0)]
    return cohort.sort_values(["has_attributes", "minutes"], ascending=False, ignore_index=True)


# ---------------------------------------------------------------------------
# Checks printed after a run
# ---------------------------------------------------------------------------

def pool_sizes(player_season: pd.DataFrame) -> pd.DataFrame:
    """Players with enough minutes at each position: the comparison pools."""
    pools = player_season[player_season["has_attributes"]]
    return pools.groupby("position").size().reindex(list(config.FPL_POSITIONS.values())).rename("pool").reset_index()


def attribute_spread(player_season: pd.DataFrame) -> pd.DataFrame:
    """Within the pools: range, average and the biggest share of players on a single value.

    Percentiles spread players evenly, so each value should hold about 5% of a
    pool. A much bigger share points to ties (for example many defenders with
    no goals).
    """
    rows = []
    pools = player_season[player_season["has_attributes"]]
    for attribute in ATTRIBUTES:
        for position, group in pools[attribute.applies_to(pools["position"])].groupby("position"):
            values = group[attribute.column].dropna()
            counts = values.value_counts()
            rows.append(
                {
                    "attribute": attribute.name,
                    "position": position,
                    "players": len(values),
                    "lowest": int(values.min()),
                    "highest": int(values.max()),
                    "average": round(float(values.mean()), 1),
                    "biggest single-value share": f"{100 * counts.iloc[0] / len(values):.0f}% on {counts.index[0]}",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Printing a card
# ---------------------------------------------------------------------------

def find_players(player_season: pd.DataFrame, query: str) -> pd.DataFrame:
    """Players whose FPL web name or full name matches the query (accents and case ignored)."""
    wanted = normalise_name(query)
    web = player_season["web_name"].map(normalise_name)
    full = player_season["name"].map(normalise_name)
    words = set(wanted.split())
    hits = (web == wanted) | web.str.contains(wanted, regex=False) | full.map(lambda name: words <= set(name.split()))
    return player_season[hits]


def format_card(player: pd.Series, player_season: pd.DataFrame) -> str:
    """A player's card as text: attributes, and the per-90 numbers behind each one."""
    position = player["position"]
    pool = player_season[player_season["has_attributes"] & (player_season["position"] == position)]
    born = f"{player['birth_date']:%d %b %Y}" if pd.notna(player["birth_date"]) else "unknown"
    lines = [
        f"{player['name']} ({player['web_name']}), {position}, {player['club']}",
        f"Born {born} (source: {player['birth_date_source']}), age {player['age_on_cutoff']} on "
        f"{config.U21_CUTOFF_DATE:%d %b %Y}; {'U21' if player['u21'] else 'not U21'}",
        f"{config.CURRENT_SEASON}: {player['minutes']:,} minutes, {player['starts']} starts",
    ]
    if not player["has_attributes"]:
        lines.append(f"Fewer than {config.MIN_MINUTES_FOR_PER90} minutes, so no attributes.")
        return "\n".join(lines)
    lines.append(f"Compared with the {len(pool)} {position} players with {config.MIN_MINUTES_FOR_PER90}+ minutes.")
    lines.append("")
    lines.append(f"  {'Attribute':<16}{'Score':>5}   Inputs per 90: player (median of group, player's percentile)")
    for attribute in ATTRIBUTES:
        if attribute.goalkeepers_only != (position == "GK"):
            continue
        inputs = [
            f"{MEASURE_LABELS[m]} {player[f'{m}_per90']:.2f} "
            f"({pool[f'{m}_per90'].median():.2f}, {100 * player[f'{m}_pct']:.0f})"
            for m in attribute.inputs
        ]
        lines.append(f"  {attribute.name:<16}{int(player[attribute.column]):>5}   " + "; ".join(inputs))
    if position == "GK":
        totals = (f"{player['saves']} saves, {player['goals_conceded']} goals conceded, "
                  f"{player['expected_goals_conceded']:.1f} xGC")
    else:
        totals = (
            f"{player['goals_scored']} goals, {player['assists']} assists, {player['expected_goals']:.1f} xG, "
            f"{player['expected_assists']:.1f} xA, {player['tackles']} tackles, "
            f"{player['clearances_blocks_interceptions']} clearances/blocks/interceptions, "
            f"{player['recoveries']} recoveries"
        )
    lines.append("")
    lines.append(f"  Season totals: {totals}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Build per-90 stats and 1 to 20 attributes for the current season.")
    parser.add_argument(
        "--show", nargs="+", metavar="NAME",
        help="print the card of each named player (FPL web name or full name; accents optional)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build both tables, write them, and print the checks or the requested cards."""
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # player names may not fit the console's encoding
    config.ensure_directories()
    player_season = build_player_season()
    cards = build_u21_cards(player_season)
    player_season.to_parquet(config.PLAYER_SEASON_PATH, index=False)
    cards.to_parquet(config.U21_CARDS_PATH, index=False)

    if args.show:
        for query in args.show:
            found = find_players(player_season, query)
            if found.empty:
                print(f"\nNo player in {config.CURRENT_SEASON} matches '{query}'.")
            for _, player in found.iterrows():
                print("\n" + "=" * 100 + "\n" + format_card(player, player_season))
        return 0

    pd.set_option("display.width", 220)
    print(f"Comparison pools ({config.MIN_MINUTES_FOR_PER90}+ minutes):")
    print(pool_sizes(player_season).to_string(index=False))
    print("\nAttribute spread within the pools:")
    print(attribute_spread(player_season).to_string(index=False))
    shown = ["web_name", "position", "club", "age_on_cutoff", "minutes", *[a.column for a in ATTRIBUTES]]
    print(f"\nU21 cards with attributes ({int(cards['has_attributes'].sum())} of {len(cards)}):")
    print(cards[cards["has_attributes"]][shown].to_string(index=False))
    print(f"\nPlayer-season table: {config.PLAYER_SEASON_PATH} ({len(player_season)} rows)")
    print(f"U21 cards:           {config.U21_CARDS_PATH} ({len(cards)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

