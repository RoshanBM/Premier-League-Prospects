"""Current Ability (CA) and Potential Ability (PA) for the 2025/26 cards.

They measure different things:
- **CA** is a weighted mean of the 2025/26 per-90 attributes (features.py):
  how productive a player was per 90 minutes this season, against every PL
  player at his position. It says nothing about how much he played.
- **PA** comes from the trajectory model's projection (trajectory.py) of his
  Premier League minutes share at ages 24 to 26: how much he is likely to
  play at his peak, given his age, position and minutes so far. It says
  nothing about his per-90 output.
A player can have a high CA and a low PA (productive in few minutes, at a
club where he is unlikely to break through), or the reverse.

CA
    CA = sum over attributes of weight x attribute, with hand-chosen weights per
    position that sum to 1 (CA_WEIGHTS below). Attributes run from 1 to 20,
    so CA does too. Each attribute's contribution (weight x attribute) is kept,
    so the card can show what drives the score. Only players with at least
    config.MIN_MINUTES_FOR_PER90 minutes have attributes, and so a CA.

PA
    For each player in the 2025/26 U21 cohort (cohort.py), the projected peak
    is his mean minutes share over the seasons at ages 24, 25 and 26, from the
    fixed effects plus his own random effects. The model is the one fitted on
    players first seen at 20 or under (trajectory.py). 2,000 simulated futures
    (trajectory.simulate_window_mean) give the median and an 80% interval.
    Two versions of the interval are written: the raw one, as simulated, and
    a calibrated one, widened by the backtest's factor (calibration.py,
    interval_calibration.json). The card uses config.CARD_INTERVAL ("raw";
    the calibrated version is kept as a sensitivity check). Values are clipped
    to 0 to 1 for display only; the unclipped values are kept as well.

    The card's PA (column `pa`, labelled "rank among 2025/26 U21 prospects")
    is the mid-rank percentile of the player's clipped projected median among
    every projected 2025/26 U21 (all positions together; the Day 1 rule in
    features.mid_rank_percentile), mapped to 1 to 20 with
    min(20, 1 + floor(percentile x 20)). Projections are compared only with
    projections: comparing an expected share with realised shares, half of
    which are 0 for young debutants, put any projection above 0 ahead of every
    player who dropped out. The card shows no 1 to 20 bounds; the minutes range
    text carries the uncertainty.

    Sensitivity columns only (not for the card): the median and the card
    interval's bounds scored against what players actually did at ages 24 to
    26, by position, for all players (pa_all_players_*) and for players first
    seen at 20 or under (pa_young_debutants_*).

    What the card shows alongside PA:
    - pa_status: "experimental" for every projected player, because the
      backtest did not show the model beating the persistence baseline; a
      player whose PL seasons so far all came before age 17 gets no PA and
      "insufficient data: debut before 17" (the model is not fitted below 17,
      so it would only give him the average curve);
    - pa_low_evidence and pa_display: a PA resting on fewer than
      config.PA_LOW_EVIDENCE_MINUTES (450) career PL minutes is shown greyed out;
    - persistence_share: his mean minutes share so far, the baseline the
      backtest compares against;
    - pa_range_text: the card's range in words, rounded to 5%, always naming
      the ages. A range whose lower end rounds to 0% reads "Range: from not
      playing regularly in the PL to about X% of league minutes at 24 to 26",
      because dropping out is a real outcome.

Writes data/processed/ca_2025_26.parquet and
data/processed/pa_projections_2025_26.parquet.

Run from the repository root, after python -m src.features, python -m src.trajectory
and python -m src.validate (which writes the calibration factor):
    python -m src.ability
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.calibration import Calibration, load_calibration, widen
from src.cohort import build_cohort, season_players
from src.features import ATTRIBUTES, attribute_score, mid_rank_percentile
from src.panel import first_seen_young, read_panel
from src.trajectory import TrajectoryModel, future_targets, load_model, model_rows, project

PA_STATUS_EXPERIMENTAL: str = "experimental"
PA_STATUS_TOO_YOUNG: str = f"insufficient data: debut before {config.MODEL_AGE_RANGE[0]}"
PA_LABEL: str = f"rank among {config.CURRENT_SEASON.replace('-', '/')} U21 prospects"
PA_DISPLAY_NORMAL: str = "normal"
PA_DISPLAY_GREYED: str = "greyed"
# Projection fields that are left missing for a player who gets no PA.
PROJECTION_FIELDS: tuple[str, ...] = (
    "re_intercept", "re_slope", "intercept_contribution", "slope_contribution", "peak_population", "peak_predicted",
    "peak_median", "peak_lower", "peak_upper", "peak_lower_calibrated", "peak_upper_calibrated",
)
# The two versions of the 80% interval: display (clipped) lower and upper columns.
INTERVALS: dict[str, tuple[str, str]] = {
    "raw": ("peak_lower_display", "peak_upper_display"),
    "calibrated": ("peak_lower_calibrated_display", "peak_upper_calibrated_display"),
}
# Sensitivity references for the 1 to 20 scale: name -> first_seen_max_age for peak_reference.
SENSITIVITY_REFERENCES: dict[str, int | None] = {
    "all_players": None,
    "young_debutants": config.FIRST_SEEN_YOUNG_MAX_AGE,
}
RANGE: str = "Projected to play about {span} of league minutes at {low} to {high}"
RANGE_FROM_ZERO: str = "Range: from not playing regularly in the PL to about {upper}% of league minutes at {low} to {high}"
RANGE_ALL_ZERO: str = "Range: not playing regularly in the PL at {low} to {high}"


def card_bounds(interval: str = config.CARD_INTERVAL) -> tuple[str, str]:
    """The display (clipped) lower and upper columns the card uses, for "raw" or "calibrated"."""
    if interval not in INTERVALS:
        raise ValueError(f"unknown card interval {interval!r}; expected one of {sorted(INTERVALS)}")
    return INTERVALS[interval]


# ---------------------------------------------------------------------------
# Current Ability: the weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Weight:
    """One attribute's weight in a position's CA, and the reason for it."""

    value: float
    reason: str


# Hand-chosen weights. Each position's weights sum to 1 (checked by
# check_weights). Attributes are the columns written by features.py. To change
# CA, change these numbers and their reasons, then rerun python -m src.ability.
CA_WEIGHTS: dict[str, dict[str, Weight]] = {
    "GK": {
        "goalkeeping": Weight(1.00, "the only goalkeeping attribute the data supports (saves; goals prevented)"),
    },
    "DEF": {
        "defending": Weight(0.40, "stopping attacks is a defender's first job (tackles, clearances, blocks, interceptions)"),
        "ball_recovery": Weight(0.25, "winning loose balls back matters for every defender"),
        "chance_creation": Weight(0.15, "full-backs and ball-playing centre-backs create chances"),
        "shot_threat": Weight(0.10, "set-piece threat has value but is secondary for a defender"),
        "finishing": Weight(0.10, "defenders score rarely, so goals per 90 are noisy; kept small"),
    },
    "MID": {
        "chance_creation": Weight(0.30, "creating chances is the most distinctive midfield output in the data"),
        "ball_recovery": Weight(0.25, "regaining possession is a core midfield job"),
        "defending": Weight(0.15, "tackles and interceptions matter, more for deeper midfielders"),
        "shot_threat": Weight(0.15, "getting into scoring positions matters for attacking midfielders"),
        "finishing": Weight(0.15, "goals matter but are noisier than xG, so weighted no higher than shot threat"),
    },
    "FWD": {
        "shot_threat": Weight(0.35, "xG per 90 is the steadiest sign of a forward who gets good chances"),
        "finishing": Weight(0.30, "goals are the job, but goals minus xG is noisy, so a little below shot threat"),
        "chance_creation": Weight(0.20, "forwards who also create are worth more"),
        "ball_recovery": Weight(0.10, "pressing forwards win the ball back high up"),
        "defending": Weight(0.05, "little defending is expected of a forward"),
    },
}
WEIGHT_TOLERANCE: float = 1e-9


def check_weights(weights: dict[str, dict[str, Weight]] = CA_WEIGHTS) -> None:
    """Raise ValueError unless every position's weights are positive, sum to 1 and name real attributes."""
    known = {attribute.column for attribute in ATTRIBUTES}
    goalkeeping = {attribute.column for attribute in ATTRIBUTES if attribute.goalkeepers_only}
    for position, by_attribute in weights.items():
        total = sum(weight.value for weight in by_attribute.values())
        if abs(total - 1) > WEIGHT_TOLERANCE:
            raise ValueError(f"{position} CA weights sum to {total}, not 1")
        unknown = set(by_attribute) - known
        if unknown:
            raise ValueError(f"{position} CA weights name unknown attributes: {sorted(unknown)}")
        allowed = goalkeeping if position == "GK" else known - goalkeeping
        if set(by_attribute) - allowed:
            raise ValueError(f"{position} CA weights use attributes that do not apply to it")
        if any(weight.value <= 0 for weight in by_attribute.values()):
            raise ValueError(f"{position} CA weights must be positive")


def current_ability(players: pd.DataFrame, weights: dict[str, dict[str, Weight]] = CA_WEIGHTS) -> pd.DataFrame:
    """CA and each attribute's contribution (weight x attribute) for each player.

    Args:
        players: one row per player with position and the attribute columns.
        weights: CA weights per position.

    Returns:
        A table with the same index: ca, plus contrib_<attribute> for every
        attribute (0 where an attribute does not apply to the position).
        Players without attributes (fewer than 450 minutes) get missing values.
    """
    check_weights(weights)
    contrib_columns = [f"contrib_{attribute.column}" for attribute in ATTRIBUTES]
    out = pd.DataFrame(np.nan, index=players.index, columns=contrib_columns)
    for position, by_attribute in weights.items():
        rows = players["position"] == position
        for attribute in ATTRIBUTES:
            weight = by_attribute.get(attribute.column)
            values = players.loc[rows, attribute.column].astype("float64") if weight else 0.0
            out.loc[rows, f"contrib_{attribute.column}"] = weight.value * values if weight else 0.0
    # skipna=False: a player missing any weighted attribute gets no CA, not a partial sum.
    out["ca"] = out[contrib_columns].sum(axis=1, skipna=False)
    out.loc[out["ca"].isna(), contrib_columns] = np.nan
    return out


def build_ca() -> pd.DataFrame:
    """CA for every 2025/26 player with enough minutes, with a U21 flag."""
    player_season = pd.read_parquet(config.PLAYER_SEASON_PATH)
    qualified = player_season[player_season["enough_minutes"]].reset_index(drop=True)
    ca = current_ability(qualified)
    identity = ["fpl_code", "name", "web_name", "position", "club", "u21", "age_on_cutoff", "minutes"]
    attributes = [attribute.column for attribute in ATTRIBUTES]
    table = pd.concat([qualified[identity + attributes], ca], axis=1)
    table["ca_rank_in_position"] = table.groupby("position")["ca"].rank(ascending=False, method="min").astype("Int64")
    return table.sort_values(["position", "ca"], ascending=[True, False], ignore_index=True)


# ---------------------------------------------------------------------------
# Potential Ability: the card's rank among this season's prospects
# ---------------------------------------------------------------------------

def prospect_rank(medians: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The card's PA: percentile of each projected median among all of them, and its 1 to 20 value.

    Uses the Day 1 mid-rank rule (features.mid_rank_percentile: ties share a
    percentile, missing stays missing) on the clipped medians of every
    projected player, all positions together, then the Day 1 mapping
    (features.attribute_score).

    Returns:
        (percentile from 0 to 1, PA from 1 to 20), both with the medians' index.
    """
    percentiles = mid_rank_percentile(medians)
    return percentiles, attribute_score(percentiles)


def low_evidence(career_minutes: pd.Series, threshold: int = config.PA_LOW_EVIDENCE_MINUTES) -> pd.Series:
    """True where a PA rests on fewer than `threshold` career PL minutes."""
    return (career_minutes < threshold).astype(bool)


# ---------------------------------------------------------------------------
# Potential Ability: sensitivity scores against what players actually did
# ---------------------------------------------------------------------------

def peak_reference(panel: pd.DataFrame, ages: tuple[int, int] = config.PEAK_AGE_RANGE,
                   first_seen_max_age: int | None = None) -> pd.DataFrame:
    """What players actually did at peak ages: each player's mean minutes share over his seasons at those ages.

    Uses every panel season at completed ages within `ages`, observed or
    filled (zeros included, as in the model's outcome). Position is the one in
    his latest season inside the window. One row per player.

    Args:
        panel: the player-season panel.
        ages: the peak ages (completed years, inclusive).
        first_seen_max_age: keep only players first seen at this age or under;
            None keeps every player.
    """
    low, high = ages
    in_window = panel["age_completed"].between(low, high).fillna(False).astype(bool)
    if first_seen_max_age is not None:
        in_window &= first_seen_young(panel, first_seen_max_age)
    window = panel[in_window]
    window = window.sort_values(["fpl_code", "season_year"])
    return window.groupby("fpl_code").agg(
        position=("position", "last"),
        peak_share=("minutes_share", "mean"),
        seasons=("season", "size"),
    ).reset_index()


def percentile_against(values: pd.Series, reference: np.ndarray) -> pd.Series:
    """Mid-rank percentile of each value against a reference sample (0 to 1).

    Share of the reference below the value, plus half the share equal to it,
    the same convention as features.mid_rank_percentile. Missing stays missing.
    """
    reference = np.sort(np.asarray(reference, dtype=float))
    below = np.searchsorted(reference, values.to_numpy(dtype=float), side="left")
    not_above = np.searchsorted(reference, values.to_numpy(dtype=float), side="right")
    result = (below + 0.5 * (not_above - below)) / len(reference)
    return pd.Series(result, index=values.index).where(values.notna())


def reference_scores(projections: pd.DataFrame, reference: pd.DataFrame, columns: dict[str, str],
                     prefix: str) -> pd.DataFrame:
    """1 to 20 values for clipped shares, against the same position's reference (sensitivity columns).

    Args:
        projections: one row per player, with position and the columns named in `columns`.
        reference: peak_reference output.
        columns: name -> share column to score; each gives <prefix>_<name>.
        prefix: column prefix, such as "pa_all_players".

    Missing shares give missing values.
    """
    out = pd.DataFrame(index=projections.index)
    for name, column in columns.items():
        scores = pd.Series(pd.NA, index=projections.index, dtype="Int64")
        for position, group in projections.groupby("position"):
            sample = reference.loc[reference["position"] == position, "peak_share"].to_numpy()
            scores.loc[group.index] = attribute_score(percentile_against(group[column], sample))
        out[f"{prefix}_{name}"] = scores
    return out


# ---------------------------------------------------------------------------
# Potential Ability: words and baselines for the card
# ---------------------------------------------------------------------------

def nearest_five_percent(share: float) -> int:
    """A share (0 to 1) as a whole percentage rounded to the nearest 5, halves rounded up."""
    return int(np.floor(share * 20 + 0.5) * 5)


def range_text(lower: pd.Series, upper: pd.Series, ages: tuple[int, int] = config.PEAK_AGE_RANGE) -> pd.Series:
    """The projected range in words for the card, from clipped shares rounded to 5%; every text names the ages.

    - Usual case: "Projected to play about 20% to 55% of league minutes at 24 to 26"
      ("about 30%" if both ends round to the same value).
    - Lower end rounds to 0%: "Range: from not playing regularly in the PL to
      about 55% of league minutes at 24 to 26", because dropping out is a real
      outcome and "about 0%" reads oddly to a coach.
    - Both ends round to 0%: "Range: not playing regularly in the PL at 24 to 26".
    Missing bounds give missing text.
    """
    low_age, high_age = ages

    def one(low: float, high: float) -> str | None:
        if pd.isna(low) or pd.isna(high):
            return None
        a, b = nearest_five_percent(low), nearest_five_percent(high)
        if a == 0:
            if b == 0:
                return RANGE_ALL_ZERO.format(low=low_age, high=high_age)
            return RANGE_FROM_ZERO.format(upper=b, low=low_age, high=high_age)
        span = f"{a}%" if a == b else f"{a}% to {b}%"
        return RANGE.format(span=span, low=low_age, high=high_age)

    return pd.Series([one(a, b) for a, b in zip(lower, upper)], index=lower.index, dtype="str")


def persistence_shares(panel: pd.DataFrame, codes: pd.Series, last_season: str = config.CURRENT_SEASON) -> pd.Series:
    """Each player's mean minutes share over all his panel seasons up to last_season (the persistence baseline).

    Every panel season counts, observed or filled, at any age, the same
    definition as the backtest's persistence baseline (validate.py).
    """
    upto = panel[(panel["season_year"] <= config.season_start_year(last_season)) & panel["fpl_code"].isin(codes)]
    return upto.groupby("fpl_code")["minutes_share"].mean()


# ---------------------------------------------------------------------------
# Potential Ability: building the table
# ---------------------------------------------------------------------------

def cohort_latest_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """The 2025/26 U21 cohort (cohort.py), with each player's 2025/26 panel row."""
    cohort = build_cohort(season_players(config.CURRENT_SEASON, config.U21_CUTOFF_DATE))
    current = panel[panel["season"] == config.CURRENT_SEASON]
    rows = current[current["fpl_code"].isin(cohort["fpl_code"])]
    missing = set(cohort["fpl_code"]) - set(rows["fpl_code"])
    if missing:
        raise ValueError(f"cohort players missing from the panel's {config.CURRENT_SEASON} rows: {sorted(missing)}")
    if not (rows["age_completed"] < config.U21_AGE_LIMIT).all():
        raise ValueError("a cohort player is not U21 by the panel's age")
    return rows.reset_index(drop=True)


def check_inputs(model: TrajectoryModel, calibration: Calibration) -> None:
    """Refuse to project with a stale model or a calibration made for another model."""
    if model.last_season != config.CURRENT_SEASON:
        raise ValueError(f"the saved model was fitted up to {model.last_season}, not {config.CURRENT_SEASON}; "
                         "run python -m src.trajectory")
    if model.first_seen_max_age != config.FIRST_SEEN_YOUNG_MAX_AGE:
        raise ValueError(f"the saved model was fitted on {model.training_population}, not on players first seen at "
                         f"{config.FIRST_SEEN_YOUNG_MAX_AGE} or under; run python -m src.trajectory")
    if calibration.first_seen_max_age != model.first_seen_max_age:
        raise ValueError("the calibration factor was estimated for a different model; run python -m src.validate")


def build_pa(model: TrajectoryModel, panel: pd.DataFrame, calibration: Calibration) -> pd.DataFrame:
    """PA projections for the 2025/26 U21 cohort, with both intervals, the card's rank, flags and persistence."""
    check_inputs(model, calibration)
    cohort = cohort_latest_rows(panel)
    history = model_rows(panel, model.age_range)
    history = history[history["fpl_code"].isin(cohort["fpl_code"])]
    targets = future_targets(cohort, config.PEAK_AGE_RANGE)
    projected = project(model, history, targets)
    projected = projected.rename(columns={
        "population": "peak_population", "predicted": "peak_predicted",
        "sim_median": "peak_median", "sim_lower": "peak_lower", "sim_upper": "peak_upper",
    })
    listed = panel[panel["fpl_code"].isin(cohort["fpl_code"])].groupby("fpl_code").agg(
        seasons_listed=("season", "size"), pl_minutes_to_date=("minutes", "sum"))
    table = cohort[["fpl_code", "name", "web_name", "position", "club", "birth_date", "age", "age_completed",
                    "minutes", "debut_season", "first_seen_age"]].rename(columns={"minutes": "minutes_2025_26"})
    table = table.merge(listed, on="fpl_code").merge(projected, on="fpl_code", validate="one_to_one")
    table["persistence_share"] = table["fpl_code"].map(persistence_shares(panel, table["fpl_code"]))
    # How far his own random effects move the projected peak from the average curve:
    # the intercept moves it by u0, the slope by u1 x (mean age_c of the peak seasons).
    mean_target_age_c = table["fpl_code"].map(targets.groupby("fpl_code")["age_c"].mean())
    table["intercept_contribution"] = table["re_intercept"]
    table["slope_contribution"] = (table["re_slope"] * mean_target_age_c) if model.random_slope else 0.0

    references = {name: peak_reference(panel, first_seen_max_age=age) for name, age in SENSITIVITY_REFERENCES.items()}
    table = finish_pa(table, references, calibration.factor, model.age_range[0])
    table["interval_level"] = config.INTERVAL_LEVEL
    table["draws"] = config.SIMULATION_DRAWS
    table["random_slope"] = model.random_slope
    table["model_fitted_on"] = model.training_population
    return table.sort_values("peak_median", ascending=False, ignore_index=True)


def finish_pa(table: pd.DataFrame, references: dict[str, pd.DataFrame], factor: float,
              min_age: int = config.MODEL_AGE_RANGE[0], card_interval: str = config.CARD_INTERVAL,
              low_evidence_minutes: int = config.PA_LOW_EVIDENCE_MINUTES) -> pd.DataFrame:
    """From raw projections to the card's fields.

    Args:
        table: one row per player with position, age_completed,
            history_seasons, pl_minutes_to_date and the raw projection fields
            (peak_median, peak_lower, peak_upper and any others in PROJECTION_FIELDS).
        references: name -> peak_reference output, for the sensitivity
            columns pa_<name>_median, pa_<name>_lower and pa_<name>_upper.
        factor: the backtest's calibration factor, used for the calibrated
            version of the interval.
        min_age: the model's first fitting age. A player whose latest
            completed age is below it has no season the model can use, so he
            gets no PA: status PA_STATUS_TOO_YOUNG and every projection field
            missing.
        card_interval: which version of the interval the card uses, "raw" or "calibrated".
        low_evidence_minutes: fewer career PL minutes than this flags a PA as low evidence.

    Returns:
        A copy of the table with, for the card: pa (1 to 20), pa_pct, pa_label,
        pa_status, pa_low_evidence, pa_display ("normal", "greyed", or missing
        with no PA), pa_range_text and card_interval. Also both intervals
        (peak_*_calibrated, *_display copies clipped to 0 to 1,
        pa_range_text_raw, pa_range_text_calibrated), calibration_factor,
        clipped_for_display, the sensitivity scores and reference_players
        (per reference, players at his position).
    """
    lower_card, upper_card = card_bounds(card_interval)  # fails early on an unknown version
    table = table.copy()
    table["calibration_factor"] = factor
    lower, upper = widen(table["peak_median"], table["peak_lower"], table["peak_upper"], factor)
    table["peak_lower_calibrated"], table["peak_upper_calibrated"] = lower, upper

    too_young = table["age_completed"] < min_age
    if not (too_young == (table["history_seasons"] == 0)).all():
        raise ValueError(f"a player with no seasons in the model's ages should be exactly one who is under {min_age}")
    table["pa_status"] = np.where(too_young, PA_STATUS_TOO_YOUNG, PA_STATUS_EXPERIMENTAL)
    fields = [column for column in PROJECTION_FIELDS if column in table.columns]
    table.loc[too_young, fields] = np.nan

    bounds = ["peak_median", "peak_lower", "peak_upper", "peak_lower_calibrated", "peak_upper_calibrated"]
    for column in bounds:
        table[f"{column}_display"] = table[column].clip(0, 1)
    table["clipped_for_display"] = ((table[bounds] < 0) | (table[bounds] > 1)).any(axis=1)

    # The card: rank among this season's projected U21s, flags and words.
    table["pa_pct"], table["pa"] = prospect_rank(table["peak_median_display"])
    table["pa_label"] = PA_LABEL
    table["pa_low_evidence"] = low_evidence(table["pl_minutes_to_date"], low_evidence_minutes)
    table["pa_display"] = pd.Series(np.where(table["pa_low_evidence"], PA_DISPLAY_GREYED, PA_DISPLAY_NORMAL),
                                    index=table.index, dtype="str").where(~too_young)
    for version, (lower_column, upper_column) in INTERVALS.items():
        table[f"pa_range_text_{version}"] = range_text(table[lower_column], table[upper_column])
    table["card_interval"] = card_interval
    table["pa_range_text"] = table[f"pa_range_text_{card_interval}"]

    # Sensitivity only: the median and the card interval's bounds against what players actually did.
    scored = {"median": "peak_median_display", "lower": lower_card, "upper": upper_card}
    for name, reference in references.items():
        table = pd.concat([table, reference_scores(table, reference, scored, f"pa_{name}")], axis=1)
        table[f"reference_players_{name}"] = table["position"].map(reference.groupby("position").size())
    return table


# ---------------------------------------------------------------------------
# Running from the command line
# ---------------------------------------------------------------------------

def main() -> int:
    """Build CA and PA, write them and print the cohort."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    config.ensure_directories()
    pd.set_option("display.width", 250)

    ca = build_ca()
    ca.to_parquet(config.CA_PATH, index=False)
    print(f"CA: {len(ca)} players with {config.MIN_MINUTES_FOR_PER90}+ minutes, {int(ca['u21'].sum())} of them U21")
    print(ca.groupby("position")["ca"].describe().round(2).to_string())

    panel = read_panel()
    model = load_model()
    calibration = load_calibration()
    pa = build_pa(model, panel, calibration)
    pa.to_parquet(config.PA_PROJECTIONS_PATH, index=False)

    projected = pa[pa["pa_status"] == PA_STATUS_EXPERIMENTAL]
    print(f"\nPA: {len(pa)} cohort players; {len(projected)} with a PA ({PA_STATUS_EXPERIMENTAL}); "
          f"{int((pa['pa_status'] == PA_STATUS_TOO_YOUNG).sum())} '{PA_STATUS_TOO_YOUNG}'")
    print(f"Card PA: {PA_LABEL}, ranked among the {len(projected)} projected medians (all positions together)")
    scores = projected["pa"].astype(float)
    print(f"Card PA distribution: mean {scores.mean():.2f}, median {scores.median():.0f}, "
          f"quartiles {scores.quantile(0.25):.0f} and {scores.quantile(0.75):.0f}, range {scores.min():.0f} to {scores.max():.0f}")
    print(projected["pa"].value_counts().sort_index().rename("players").to_frame().T.to_string())
    print("Mean card PA by position:", projected.groupby("position")["pa"].mean().round(2).to_dict())
    low = projected["pa_low_evidence"]
    print(f"Low evidence (under {config.PA_LOW_EVIDENCE_MINUTES} career PL minutes, shown greyed): {int(low.sum())} of "
          f"{len(projected)} projected players; their card PA {projected.loc[low, 'pa'].min()} to {projected.loc[low, 'pa'].max()}")
    print(f"Also low evidence among the players with no PA: {int(pa.loc[pa['pa_status'] == PA_STATUS_TOO_YOUNG, 'pa_low_evidence'].sum())}")
    for name in SENSITIVITY_REFERENCES:
        column = f"pa_{name}_median"
        print(f"Sensitivity, median scored against {name.replace('_', ' ')}: mean {projected[column].astype(float).mean():.2f}, "
              f"range {projected[column].min()} to {projected[column].max()}; reference sizes "
              f"{projected.groupby('position')[f'reference_players_{name}'].first().to_dict()}")
    print(f"Card interval: {config.CARD_INTERVAL}. Backtest calibration factor (calibrated version, sensitivity "
          f"check): {calibration.factor:.3f} ({calibration.reason})")
    for version in INTERVALS:
        from_zero = projected[f"pa_range_text_{version}"].str.startswith("Range:")
        print(f"{version}: {int(from_zero.sum())} of {len(projected)} ranges start at 0%")
    for label, low_col, high_col in (("raw", "peak_lower", "peak_upper"),
                                     ("calibrated", "peak_lower_calibrated", "peak_upper_calibrated")):
        print(f"Median interval width, {label}: {(projected[high_col] - projected[low_col]).median():.3f} unclipped, "
              f"{(projected[high_col + '_display'] - projected[low_col + '_display']).median():.3f} clipped for display")
    print("How far each player's own random effects move his projected peak, by seasons of history (absolute values):")
    print(projected.assign(slope=projected["slope_contribution"].abs(), intercept=projected["intercept_contribution"].abs())
          .groupby("history_seasons").agg(players=("fpl_code", "size"), median_slope=("slope", "median"),
                                          max_slope=("slope", "max"), median_intercept=("intercept", "median"))
          .round(3).to_string())
    merged = pa.merge(ca[["fpl_code", "ca"]], on="fpl_code", how="left")
    shown = ["web_name", "position", "club", "age", "pl_minutes_to_date", "persistence_share", "ca",
             "peak_median_display", "pa", "pa_low_evidence", "pa_all_players_median", "pa_young_debutants_median",
             "pa_range_text"]
    print(merged[shown].round(3).to_string(index=False))
    print(f"\nCA written to {config.CA_PATH}")
    print(f"PA written to {config.PA_PROJECTIONS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
