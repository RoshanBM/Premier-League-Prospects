"""Backtest of the trajectory model, with time-based splits only.

For each forecast origin T (config.BACKTEST_ORIGINS, 2017-18 to 2020-21):
1. The cohort is every player U21 on 1 Sep of T with PL minutes in T, built
   with the same code as the 2025/26 cohort (cohort.py).
2. The training data is the panel seasons up to and including T, for players
   who had played by T. check_no_leakage stops the run if a later season, or
   a player who debuted later, gets in.
3. Two models are fitted on it, with the same formula, fitting ages and
   fallback rules:
   - the new model, on players first seen at 20 or under only
     (config.FIRST_SEEN_YOUNG_MAX_AGE; check_first_seen_young enforces it).
     This is the survivorship fix: the panel starts in 2016-17, so a player
     first seen older is only in it because he was still in the league;
   - the Day 2 model, on every player, for comparison.
4. Each predicts every cohort player's mean minutes share over his seasons at
   completed ages 23 to 25 (config.BACKTEST_TARGET_AGES), using his position in
   season T. The prediction is compared with what happened, zeros included.

Only target seasons up to 2025-26 exist, so a younger cohort member may have
only one or two of his three target seasons; a player with none is left out.
Prediction and outcome always use the same seasons. "Full window" rows report
only players with all three.

Peak identification. The new model's training data holds few or no seasons
at ages 23 to 25 at the early origins: a player first seen at 20 in 2016-17 is
only 23 in 2019-20. An origin with fewer than
config.PEAK_IDENTIFIED_MIN_SEASONS such seasons is marked "peak not
identified": the curve there is an extrapolation. Pooled results are given for
the identified origins (the headline), for the others separately, and for all
four together (only to compare like for like with Day 2).

Methods compared:
- new model (first seen young): fixed effects plus the player's own random
  effects (BLUP);
- Day 2 model (all players): the same, fitted on every player;
- persistence ("no change"): his mean minutes share over all his panel
  seasons up to T;
- population curve: the new model's fixed effects only.
Sensitivity check (not a primary result): the new model clipped to 0 to 1.

Metrics: MAE, RMSE, Spearman rank correlation, the bias (mean prediction minus
mean actual) and, for both mixed models, coverage of the 80% interval.

Interval calibration (calibration.py). If the new model's pooled coverage over
all four origins is below config.CALIBRATION_TRIGGER_COVERAGE, its intervals
are widened by a factor estimated from earlier origins' errors only, and the
factor from all four origins is saved for the 2025/26 projections.

Writes data/processed/backtest_results.csv, backtest_players.csv,
backtest_summary.md and interval_calibration.json.

Run from the repository root, after python -m src.panel:
    python -m src.validate
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config
from src.calibration import (
    Calibration,
    RollingFactor,
    calibration_factor,
    needs_calibration,
    rolling_factors,
    save_calibration,
    scale_needed,
    widen,
)
from src.cohort import in_cohort, season_players
from src.panel import first_seen_young, read_panel
from src.trajectory import TrajectoryModel, fit_with_fallback, load_model, model_rows, project, training_rows

NEW_MODEL = "new model (first seen young)"
DAY2_MODEL = "Day 2 model (all players)"
PERSISTENCE = "persistence (mean share to date)"
POPULATION = "population curve (new model's fixed effects)"
NEW_MODEL_CLIPPED = "new model, clipped to 0 to 1 (sensitivity)"
METHOD_COLUMNS: dict[str, str] = {
    NEW_MODEL: "predicted",
    DAY2_MODEL: "predicted_all",
    PERSISTENCE: "persistence",
    POPULATION: "population",
    NEW_MODEL_CLIPPED: "predicted_clipped",
}
PRIMARY_METHODS: tuple[str, ...] = (NEW_MODEL, DAY2_MODEL, PERSISTENCE, POPULATION)
POOLED_IDENTIFIED = "pooled, peak identified"
POOLED_NOT_IDENTIFIED = "pooled, peak not identified"
POOLED_ALL = "pooled, all origins"
YOUNG = config.FIRST_SEEN_YOUNG_MAX_AGE


# ---------------------------------------------------------------------------
# Splitting by time
# ---------------------------------------------------------------------------

def check_no_leakage(training: pd.DataFrame, origin: str) -> None:
    """Raise AssertionError if the training data has any season after the origin.

    An explicit check rather than an assert statement, so it still runs when
    Python is started with -O.
    """
    origin_year = config.season_start_year(origin)
    latest = int(training["season_year"].max()) if len(training) else origin_year
    if latest > origin_year:
        raise AssertionError(f"training data for origin {origin} includes season starting {latest}")
    debut_years = training["debut_season"].map(config.season_start_year)
    if (debut_years > origin_year).any():
        raise AssertionError(f"training data for origin {origin} includes players who debuted after it")


def check_first_seen_young(training: pd.DataFrame, max_age: int = YOUNG) -> None:
    """Raise AssertionError if any training row belongs to a player not first seen at max_age or under."""
    if not first_seen_young(training, max_age).all():
        raise AssertionError(f"training data includes players first seen older than {max_age}")


def training_panel(panel: pd.DataFrame, origin: str, first_seen_max_age: int | None = None) -> pd.DataFrame:
    """Panel rows known at the end of the origin season: seasons up to T, players who had played by T.

    Rows up to T for those players do not depend on anything later: a filled
    zero in season s needs only a debut before s and an absence in s. With
    first_seen_max_age, only players first seen at that age or under are kept.
    first_seen_age is the age in the debut season, which is at or before T for
    every player kept, so it was known at T too.
    """
    origin_year = config.season_start_year(origin)
    debut_years = panel["debut_season"].map(config.season_start_year)
    training = panel[(panel["season_year"] <= origin_year) & (debut_years <= origin_year)]
    if first_seen_max_age is not None:
        training = training[first_seen_young(training, first_seen_max_age)]
        check_first_seen_young(training, first_seen_max_age)
    check_no_leakage(training, origin)
    return training


def target_rows(panel: pd.DataFrame, codes: pd.Series, origin: str,
                ages: tuple[int, int] = config.BACKTEST_TARGET_AGES) -> pd.DataFrame:
    """The seasons after the origin at which the players are within the target ages (completed years)."""
    low, high = ages
    later = panel["season_year"] > config.season_start_year(origin)
    in_window = panel["age_completed"].between(low, high).fillna(False).astype(bool)
    return panel[later & in_window & panel["fpl_code"].isin(codes)]


def peak_identification(rows: pd.DataFrame, ages: tuple[int, int] = config.BACKTEST_TARGET_AGES,
                        minimum: int = config.PEAK_IDENTIFIED_MIN_SEASONS) -> dict:
    """How many fitting seasons the model has at the target ages, and whether that is enough.

    Args:
        rows: the model's fitting rows.
        ages: the target ages (completed years, inclusive).
        minimum: fewest seasons at those ages for the peak to count as identified.
    """
    low, high = ages
    window = rows[rows["age_completed"].between(low, high).fillna(False).astype(bool)]
    by_age = window.groupby("age_completed").size()
    counts = {f"seasons at {age}": int(by_age.get(age, 0)) for age in range(low, high + 1)}
    return {
        f"seasons at {low} to {high}": len(window),
        f"players at {low} to {high}": int(window["fpl_code"].nunique()),
        **counts,
        "peak identified": len(window) >= minimum,
    }


# ---------------------------------------------------------------------------
# One origin
# ---------------------------------------------------------------------------

@dataclass
class OriginResult:
    """Everything the backtest produced for one forecast origin."""

    origin: str
    players: pd.DataFrame  # one row per evaluated player
    model: TrajectoryModel  # the new model (first seen young)
    model_all: TrajectoryModel  # the Day 2 model (all players)
    attempts: list[str]
    cohort_size: int
    no_target_seasons: int
    identification: dict

    @property
    def peak_identified(self) -> bool:
        """True if the new model's training data has enough seasons at the target ages."""
        return bool(self.identification["peak identified"])


def fit_and_project(training: pd.DataFrame, cohort_codes: pd.Series, targets: pd.DataFrame,
                    first_seen_max_age: int | None):
    """Fit on the training rows (restricted or not) and project the targets.

    Returns:
        (projections, fitting rows, TrajectoryModel, list of FitAttempt)
    """
    rows = training_rows(training, config.MODEL_AGE_RANGE, first_seen_max_age)
    _, model, attempts = fit_with_fallback(rows, first_seen_max_age)
    history = rows[rows["fpl_code"].isin(cohort_codes)]
    return project(model, history, targets), rows, model, attempts


def run_origin(panel: pd.DataFrame, origin: str) -> OriginResult:
    """Fit both models on data up to the origin, predict the cohort's ages 23 to 25 and line up the outcomes."""
    origin_year = config.season_start_year(origin)
    training_all = training_panel(panel, origin)
    training_young = training_panel(panel, origin, YOUNG)
    players = season_players(origin)
    cohort = players[in_cohort(players)]
    codes = cohort["fpl_code"]
    missing = set(codes) - set(training_young["fpl_code"])
    if missing:  # a cohort player is U21 with minutes at T, so he was first seen at 20 or under
        raise ValueError(f"{origin}: cohort players missing from the restricted training panel: {sorted(missing)}")

    outcomes = target_rows(panel, codes, origin)
    # Position for the forecast is the one in the origin season, known at the time.
    position_at_origin = training_all[training_all["season_year"] == origin_year].set_index("fpl_code")["position"]
    targets = outcomes[["fpl_code", "age_c"]].assign(position=outcomes["fpl_code"].map(position_at_origin))

    projected, young_rows, model, attempts = fit_and_project(training_young, codes, targets, YOUNG)
    projected_all, _, model_all, attempts_all = fit_and_project(training_all, codes, targets, None)

    actual = outcomes.groupby("fpl_code").agg(actual=("minutes_share", "mean"), target_seasons_actual=("season", "size"),
                                              filled_target_seasons=("filled", "sum"))
    persistence = training_all[training_all["fpl_code"].isin(codes)].groupby("fpl_code")["minutes_share"].mean()
    day2 = projected_all.set_index("fpl_code")[["predicted", "sim_median", "sim_lower", "sim_upper"]].add_suffix("_all")
    table = projected.merge(actual, on="fpl_code", validate="one_to_one").join(day2, on="fpl_code")
    if not (table["target_seasons"] == table["target_seasons_actual"]).all():
        raise ValueError(f"{origin}: prediction and outcome use different seasons")
    table["persistence"] = table["fpl_code"].map(persistence)
    table["predicted_clipped"] = table["predicted"].clip(0, 1)
    table["covered"] = (table["sim_lower"] <= table["actual"]) & (table["actual"] <= table["sim_upper"])
    table["covered_all"] = (table["sim_lower_all"] <= table["actual"]) & (table["actual"] <= table["sim_upper_all"])
    table["full_window"] = table["target_seasons"] == config.BACKTEST_TARGET_AGES[1] - config.BACKTEST_TARGET_AGES[0] + 1
    identification = peak_identification(young_rows)
    table["peak_identified"] = identification["peak identified"]
    identity = cohort.set_index("fpl_code")[["name", "position", "exact_age_on_cutoff", "minutes"]]
    table = table.join(identity, on="fpl_code").rename(columns={"exact_age_on_cutoff": "age_at_origin",
                                                                "minutes": "minutes_at_origin"})
    table.insert(0, "origin", origin)
    low, high = config.MODEL_AGE_RANGE
    descriptions = [f"{origin}, new model (first seen young), ages {low} to {high}: " + a.describe() for a in attempts]
    descriptions += [f"{origin}, Day 2 model (all players), ages {low} to {high}: " + a.describe() for a in attempts_all]
    return OriginResult(origin, table, model, model_all, descriptions, len(cohort), len(cohort) - len(table),
                        identification)


# ---------------------------------------------------------------------------
# Calibration of the intervals
# ---------------------------------------------------------------------------

def calibrate(players: pd.DataFrame, origins: list[str], identified: pd.Series) -> tuple[pd.DataFrame, Calibration]:
    """Widen the new model's intervals with rolling factors, if its pooled coverage is too low.

    Args:
        players: every backtest forecast (all origins), with actual and the
            new model's sim_median, sim_lower and sim_upper.
        origins: the backtest origins, in time order.
        identified: per forecast, whether its origin's peak is identified
            (for the sensitivity factor only).

    Returns:
        The forecasts with scale_needed, calibration_factor, origin_calibrated,
        sim_lower_calibrated, sim_upper_calibrated and covered_calibrated added,
        and the Calibration to save.
    """
    players = players.copy()
    raw_coverage = float(players["covered"].mean())
    triggered = needs_calibration(raw_coverage)
    players["scale_needed"] = scale_needed(players["actual"], players["sim_median"], players["sim_lower"],
                                           players["sim_upper"])
    if triggered:
        rolling = rolling_factors(players, origins)
        overall = calibration_factor(players["scale_needed"])
        reason = (f"pooled coverage of the raw {config.INTERVAL_LEVEL:.0%} interval is {raw_coverage:.1%}, below "
                  f"{config.CALIBRATION_TRIGGER_COVERAGE:.0%}")
    else:
        rolling = [RollingFactor(origin, 1.0, tuple(origins[:i]), 0) for i, origin in enumerate(origins)]
        overall = 1.0
        reason = (f"pooled coverage of the raw {config.INTERVAL_LEVEL:.0%} interval is {raw_coverage:.1%}, at least "
                  f"{config.CALIBRATION_TRIGGER_COVERAGE:.0%}, so the intervals are left as they are")
    by_origin = {r.origin: r for r in rolling}
    players["calibration_factor"] = players["origin"].map({o: r.factor for o, r in by_origin.items()})
    players["origin_calibrated"] = players["origin"].map({o: r.calibrated for o, r in by_origin.items()})
    lower, upper = widen(players["sim_median"], players["sim_lower"], players["sim_upper"], players["calibration_factor"])
    players["sim_lower_calibrated"], players["sim_upper_calibrated"] = lower, upper
    players["covered_calibrated"] = (lower <= players["actual"]) & (players["actual"] <= upper)
    sensitivity = {
        "factor from peak-identified origins only": calibration_factor(players.loc[identified, "scale_needed"]),
        "forecasts at peak-identified origins": int(identified.sum()),
    }
    calibration = Calibration(
        triggered=triggered, reason=reason, raw_coverage=raw_coverage,
        trigger_coverage=config.CALIBRATION_TRIGGER_COVERAGE, level=config.INTERVAL_LEVEL,
        factor=overall, origins_used=list(origins), forecasts_used=len(players), first_seen_max_age=YOUNG,
        # Lists rather than tuples, so the object equals what JSON gives back.
        rolling=[{**asdict(r), "origins_used": list(r.origins_used)} for r in rolling], sensitivity=sensitivity,
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return players, calibration


def repeated_players(players: pd.DataFrame) -> pd.Series:
    """Per forecast, True if the same player was already forecast at an earlier origin.

    A player who is U21 in consecutive seasons is forecast at more than one
    origin, against overlapping outcome seasons. Errors at an earlier origin
    then partly predict errors at a later one, which flatters the rolling
    calibration.
    """
    order = players["origin"].map(config.season_start_year)
    ranked = players.assign(_order=order).sort_values("_order", kind="stable")
    return ranked["fpl_code"].duplicated().reindex(players.index)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation: the Pearson correlation of the ranks (ties get their average rank)."""
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a.rank(method="average"), b.rank(method="average"))[0, 1])


def metrics(players: pd.DataFrame, prediction: str) -> dict[str, float]:
    """MAE, RMSE, Spearman, mean prediction, mean actual and bias for one method's column."""
    errors = players[prediction] - players["actual"]
    return {
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "spearman": spearman(players[prediction], players["actual"]),
        "mean_predicted": float(players[prediction].mean()),
        "mean_actual": float(players["actual"].mean()),
        "bias": float(errors.mean()),
    }


def evaluation_groups(players: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Each origin, then the three pools: peak identified, peak not identified, all origins."""
    groups = [(origin, group) for origin, group in players.groupby("origin", sort=True)]
    groups += [
        (POOLED_IDENTIFIED, players[players["peak_identified"]]),
        (POOLED_NOT_IDENTIFIED, players[~players["peak_identified"]]),
        (POOLED_ALL, players),
    ]
    return groups


def results_table(players: pd.DataFrame) -> pd.DataFrame:
    """One row per group (origin or pool), subset and method."""
    coverage = {NEW_MODEL: ("covered", "covered_calibrated"), DAY2_MODEL: ("covered_all", None)}
    rows = []
    for group_name, group in evaluation_groups(players):
        for subset, chosen in (("all", group), ("full window", group[group["full_window"]])):
            for method, column in METHOD_COLUMNS.items():
                row = {"origin": group_name, "subset": subset, "method": method, "players": len(chosen)}
                if group_name in set(players["origin"]):
                    row["peak_identified"] = bool(group["peak_identified"].iloc[0])
                row.update(metrics(chosen, column) if len(chosen) else {})
                raw, calibrated = coverage.get(method, (None, None))
                row["coverage_80"] = float(chosen[raw].mean()) if raw and len(chosen) else np.nan
                row["coverage_80_calibrated"] = float(chosen[calibrated].mean()) if calibrated and len(chosen) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def calibration_table(players: pd.DataFrame, calibration: Calibration) -> pd.DataFrame:
    """Per origin and pooled: factor used, where it came from, coverage before and after."""
    repeated = repeated_players(players)
    rows = []
    for rolling in calibration.rolling:
        chosen = players[players["origin"] == rolling["origin"]]
        rows.append({
            "origin": rolling["origin"],
            "players": len(chosen),
            "peak identified": bool(chosen["peak_identified"].iloc[0]),
            "factor s used": rolling["factor"],
            "estimated from origins": ", ".join(rolling["origins_used"]) or "none (left uncalibrated)",
            "forecasts used": rolling["forecasts_used"],
            "coverage, raw": float(chosen["covered"].mean()),
            "coverage, calibrated": float(chosen["covered_calibrated"].mean()),
            "players already forecast at an earlier origin": int(repeated.loc[chosen.index].sum()),
        })
    later = players[players["origin_calibrated"]]
    for label, chosen in ((POOLED_ALL, players), ("pooled, origins with a factor", later),
                          (POOLED_IDENTIFIED, players[players["peak_identified"]])):
        rows.append({"origin": label, "players": len(chosen),
                     "coverage, raw": float(chosen["covered"].mean()) if len(chosen) else np.nan,
                     "coverage, calibrated": float(chosen["covered_calibrated"].mean()) if len(chosen) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Who the Day 2 model learnt from at the target ages
# ---------------------------------------------------------------------------

def selection_check(rows: pd.DataFrame, ages: tuple[int, int], young: int = YOUNG) -> dict[str, float]:
    """How the rows at the given ages split between players first seen young and players first seen older.

    The panel starts in 2016-17, so a player who was already 23 then is only
    in it because he was still listed in FPL at 23: he is a survivor. A player
    first seen at `young` or under (20, the oldest U21 age) has his later
    seasons filled in if he drops out. The split shows how much of what a
    model fitted on every player learns at the target ages comes from
    survivors. "First seen" is first_seen_age (the age in the debut season).
    """
    low, high = ages
    window = rows[rows["age_completed"].between(low, high).fillna(False).astype(bool)]
    entered_young = first_seen_young(window, young)
    return {
        "rows": len(window),
        "mean share": float(window["minutes_share"].mean()) if len(window) else np.nan,
        "filled zeros": float(window["filled"].mean()) if len(window) else np.nan,
        f"first seen at {young} or under": float(entered_young.mean()) if len(window) else np.nan,
        f"mean share, first seen at {young} or under": float(window.loc[entered_young, "minutes_share"].mean())
        if entered_young.any() else np.nan,
        "mean share, first seen older": float(window.loc[~entered_young, "minutes_share"].mean())
        if (~entered_young).any() else np.nan,
    }


def selection_table(panel: pd.DataFrame, origin_results: list[OriginResult]) -> pd.DataFrame:
    """selection_check for each origin's full training data (target ages) and for the full panel (peak ages)."""
    rows = []
    for result in origin_results:
        training = model_rows(training_panel(panel, result.origin))
        rows.append({"data": f"training up to {result.origin}, ages {config.BACKTEST_TARGET_AGES[0]} to "
                             f"{config.BACKTEST_TARGET_AGES[1]}",
                     **selection_check(training, config.BACKTEST_TARGET_AGES),
                     "cohort's actual mean share": float(result.players["actual"].mean())})
    full = model_rows(panel)
    rows.append({"data": f"full panel (used for PA), ages {config.PEAK_AGE_RANGE[0]} to {config.PEAK_AGE_RANGE[1]}",
                 **selection_check(full, config.PEAK_AGE_RANGE), "cohort's actual mean share": np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------

def markdown_table(frame: pd.DataFrame) -> str:
    """A pandas table as markdown, with floats to 3 decimals."""
    return frame.to_markdown(index=False, floatfmt=".3f")


def pick(results: pd.DataFrame, group: str, subset: str = "all") -> pd.DataFrame:
    """The rows of one group and subset, indexed by method."""
    return results[(results["origin"] == group) & (results["subset"] == subset)].set_index("method")


def beats(results: pd.DataFrame, group: str, metric: str, better: str, worse: str, subset: str = "all") -> bool:
    """True if `better` scores better than `worse` on a metric (lower MAE or RMSE, higher Spearman)."""
    chosen = pick(results, group, subset)[metric]
    return chosen[better] > chosen[worse] if metric == "spearman" else chosen[better] < chosen[worse]


def verdict_lines(results: pd.DataFrame, origin_results: list[OriginResult], calibration: Calibration) -> list[str]:
    """Plain statements of how the new model compares, pooled and by origin."""
    identified = [r.origin for r in origin_results if r.peak_identified]
    not_identified = [r.origin for r in origin_results if not r.peak_identified]
    lines = [
        f"- Peak identified at {len(identified)} of {len(origin_results)} origins "
        f"({', '.join(identified) or 'none'}); not identified at {', '.join(not_identified) or 'none'}.",
    ]
    pools = [(POOLED_IDENTIFIED, "on the peak-identified origins (the headline)")] if identified else []
    pools += [(POOLED_NOT_IDENTIFIED, "on the not-identified origins")] if not_identified else []
    pools += [(POOLED_ALL, "on all four origins")]
    for group, label in pools:
        chosen = pick(results, group)
        lines.append(f"- {label.capitalize()} ({int(chosen.loc[NEW_MODEL, 'players'])} forecasts):")
        for baseline in (PERSISTENCE, POPULATION, DAY2_MODEL):
            parts = []
            for metric, name in (("mae", "MAE"), ("rmse", "RMSE"), ("spearman", "Spearman")):
                verb = "beats" if beats(results, group, metric, NEW_MODEL, baseline) else "does not beat"
                parts.append(f"{name} {chosen.loc[NEW_MODEL, metric]:.3f} against {chosen.loc[baseline, metric]:.3f} "
                             f"({verb})")
            lines.append(f"  - against {baseline}: " + "; ".join(parts) + ".")
        lines.append(f"  - bias (mean prediction minus mean actual): new model {chosen.loc[NEW_MODEL, 'bias']:+.3f}, "
                     f"Day 2 model {chosen.loc[DAY2_MODEL, 'bias']:+.3f}, persistence "
                     f"{chosen.loc[PERSISTENCE, 'bias']:+.3f}.")
        lines.append(f"  - {config.INTERVAL_LEVEL:.0%} interval coverage: new model {chosen.loc[NEW_MODEL, 'coverage_80']:.1%} "
                     f"raw, {chosen.loc[NEW_MODEL, 'coverage_80_calibrated']:.1%} calibrated; Day 2 model "
                     f"{chosen.loc[DAY2_MODEL, 'coverage_80']:.1%}.")
    for metric, name in (("mae", "MAE"), ("spearman", "Spearman")):
        wins = sum(beats(results, r.origin, metric, NEW_MODEL, PERSISTENCE) for r in origin_results)
        lines.append(f"- By origin, the new model beats persistence on {name} at {wins} of {len(origin_results)} origins.")
    lines.append(f"- Calibration: {'applied' if calibration.triggered else 'not applied'}, because the "
                 f"{calibration.reason}.")
    return lines


def plain_verdict(results: pd.DataFrame, origin_results: list[OriginResult]) -> list[str]:
    """A short verdict in plain words, built from the numbers in this run."""
    everything = pick(results, POOLED_ALL)
    identified = [r.origin for r in origin_results if r.peak_identified]
    headline = pick(results, POOLED_IDENTIFIED) if identified else everything
    where = (f"where the peak is identified ({', '.join(identified)}, {int(headline.loc[NEW_MODEL, 'players'])} forecasts)"
             if identified else "over all origins")
    caveat = ", but that rests on a single origin" if len(identified) == 1 else ""
    new, day2, persistence = everything.loc[NEW_MODEL], everything.loc[DAY2_MODEL], everything.loc[PERSISTENCE]
    lower_mae = sum(beats(results, r.origin, "mae", NEW_MODEL, DAY2_MODEL) for r in origin_results)
    beats_persistence = beats(results, POOLED_IDENTIFIED if identified else POOLED_ALL, "mae", NEW_MODEL, PERSISTENCE)
    lines = [
        "In plain words:",
        f"- Fitting only on players first seen young {'reduced' if abs(new['bias']) < abs(day2['bias']) else 'did not reduce'} "
        f"the over-prediction: over all four origins, mean prediction minus mean actual went from {day2['bias']:+.3f} "
        f"(Day 2 model) to {new['bias']:+.3f}. The average error (MAE) went from {day2['mae']:.3f} to {new['mae']:.3f}, "
        f"and is lower at {lower_mae} of {len(origin_results)} origins.",
        f"- The new model {'beats' if beats_persistence else 'still does not beat'} the simple 'minutes so far' "
        f"(persistence) baseline on average error: MAE {headline.loc[NEW_MODEL, 'mae']:.3f} against "
        f"{headline.loc[PERSISTENCE, 'mae']:.3f} {where}, and {new['mae']:.3f} against {persistence['mae']:.3f} over all "
        "four origins.",
        f"- On ranking players (Spearman) it scores {headline.loc[NEW_MODEL, 'spearman']:.3f} against persistence's "
        f"{headline.loc[PERSISTENCE, 'spearman']:.3f} {where}{caveat}.",
        f"- Its {config.INTERVAL_LEVEL:.0%} ranges held the real outcome {new['coverage_80']:.0%} of the time over all "
        f"four origins ({new['coverage_80_calibrated']:.0%} after the rolling widening).",
    ]
    if not beats_persistence:
        lines.append("- So Potential Ability should be shown as experimental, next to the player's minutes share so far.")
    return lines


def setup_table(origin_results: list[OriginResult]) -> pd.DataFrame:
    """One row per origin: cohort, training data, random-effects structure and peak identification."""
    def structure(model: TrajectoryModel) -> str:
        return "intercept and slope" if model.random_slope else "intercept only"

    def spread(model: TrajectoryModel) -> dict[str, float]:
        psi = model.cov_re
        slope = model.random_slope
        return {
            "new model: intercept SD": float(np.sqrt(psi[0, 0])),
            "new model: slope SD": float(np.sqrt(psi[1, 1])) if slope else np.nan,
            "new model: intercept-slope correlation": float(psi[0, 1] / np.sqrt(psi[0, 0] * psi[1, 1])) if slope else np.nan,
        }

    rows = []
    for r in origin_results:
        rows.append({
            "origin": r.origin,
            "cohort (U21 with minutes)": r.cohort_size,
            "evaluated": len(r.players),
            "no target season by 2025-26": r.no_target_seasons,
            "full window": int(r.players["full_window"].sum()),
            "Day 2 model: rows": r.model_all.n_rows,
            "Day 2 model: random effects": structure(r.model_all),
            "new model: rows": r.model.n_rows,
            "new model: players": r.model.n_players,
            "new model: random effects": structure(r.model),
            **spread(r.model),
            **{f"new model: {k}": v for k, v in r.identification.items() if k != "peak identified"},
            "peak identified": r.peak_identified,
        })
    return pd.DataFrame(rows)


def build_summary(results: pd.DataFrame, origin_results: list[OriginResult], selection: pd.DataFrame,
                  calibration: Calibration, players: pd.DataFrame) -> str:
    """The text of backtest_summary.md."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    low, high = config.BACKTEST_TARGET_AGES
    level = f"{config.INTERVAL_LEVEL:.0%}"
    shown = ["origin", "method", "players", "mae", "rmse", "spearman", "mean_predicted", "mean_actual", "bias",
             "coverage_80", "coverage_80_calibrated"]
    primary = results[results["method"].isin(PRIMARY_METHODS)]
    pooled_names = [POOLED_IDENTIFIED, POOLED_NOT_IDENTIFIED, POOLED_ALL]
    pooled = primary[primary["origin"].isin(pooled_names)]
    by_origin = primary[~primary["origin"].isin(pooled_names) & (primary["subset"] == "all")]
    sensitivity = results[results["method"].isin([NEW_MODEL, NEW_MODEL_CLIPPED]) & (results["subset"] == "all")]
    try:
        current = load_model()
        current_line = (f"- The 2025/26 fit (`trajectory_model_params.json`, {current.n_rows} rows, "
                        f"{current.training_population}) kept the "
                        f"{'random slope' if current.random_slope else 'random intercept only'}.")
    except FileNotFoundError:
        current_line = "- The 2025/26 fit has not been run yet (python -m src.trajectory)."
    sens = calibration.sensitivity
    lines = [
        "# Backtest of the trajectory model, before and after the survivorship fix",
        "",
        f"Generated by `src/validate.py` at {stamp} (UTC). Every number below comes from that run. Per-player rows "
        "are in `data/processed/backtest_players.csv`, every metric is in `data/processed/backtest_results.csv`, "
        "and the calibration factors are in `data/processed/interval_calibration.json`. The Day 2 version of this "
        "file is kept in `data/processed/day2_before_fixes/`.",
        "",
        "## Set-up",
        "",
        "- For each origin season T, the cohort is every player U21 on 1 Sep of T with PL minutes in T (`cohort.py`).",
        "- Both models are refitted on panel seasons up to and including T only, for players who had played by T. "
        "`check_no_leakage` raises an error if a later season, or a player who debuted later, gets into the "
        "training data.",
        f"- **New model:** fitted only on players first seen at {YOUNG} or under (age in their first PL season with "
        "minutes), the same ages as the cohort it projects. `check_first_seen_young` raises an error otherwise. "
        "**Day 2 model:** the same model fitted on every player, kept for comparison.",
        f"- Target: each player's mean PL minutes share over his seasons at ages {low} to {high} (completed years on "
        "1 September), zeros included: seasons out of the league count as 0, as coded in the panel.",
        "- Seasons after 2025-26 do not exist yet, so younger players have only one or two target seasons; prediction "
        "and outcome always use the same seasons. 'Full window' means all three.",
        f"- **Peak identified** means the new model's training data has at least {config.PEAK_IDENTIFIED_MIN_SEASONS} "
        f"seasons at ages {low} to {high}. Elsewhere its curve at those ages is an extrapolation, so those origins "
        "are pooled separately.",
        f"- The models' {level} intervals come from {config.SIMULATION_DRAWS} simulated futures per player. "
        f"**Calibration** widens the new model's intervals only if its pooled coverage (all origins) is below "
        f"{config.CALIBRATION_TRIGGER_COVERAGE:.0%}; the factor used at origin T comes only from origins before T.",
        current_line,
        "",
        markdown_table(setup_table(origin_results)),
        "",
        "## Verdict",
        "",
        *plain_verdict(results, origin_results),
        "",
        "In detail:",
        "",
        *verdict_lines(results, origin_results, calibration),
        "",
        "## Before and after, pooled",
        "",
        "Day 2 results are the 'Day 2 model' rows ('pooled, all origins' matches the Day 2 pooled figures). The "
        "population curve now comes from the new model. The same player can appear at more than one origin, so "
        "pooled rows are not independent.",
        "",
        markdown_table(pooled[["origin", "subset", *shown[1:]]]),
        "",
        "## Results by origin",
        "",
        markdown_table(by_origin[shown]),
        "",
        f"## Coverage of the {level} interval, before and after calibration",
        "",
        markdown_table(calibration_table(players, calibration)),
        "",
        f"- Factor for the 2025/26 projections: s = {calibration.factor:.3f}, the {config.INTERVAL_LEVEL:.0%} point of "
        f"the s_i from all {calibration.forecasts_used} forecasts at {', '.join(calibration.origins_used)}.",
        "- The first origin has no earlier origin to learn from, so it stays uncalibrated.",
        "- 'Rolling' here means by origin, not in real time: the outcomes behind an earlier origin's errors (ages "
        f"{low} to {high}) happened after T. A forecast never widens its own interval, but the factor is not one "
        "that could have been known at T. Players forecast at more than one origin (last column) make the rolling "
        "factor look better than it would on new players.",
        "",
        "## Day 2 diagnostic: who a model fitted on every player learns from",
        "",
        "Kept from Day 2 (now using `first_seen_age`, the age in the debut season, where Day 2 used the first season "
        "in the model's ages, so the shares differ slightly).",
        "",
        markdown_table(selection),
        "",
        "## Sensitivity checks (not the primary result)",
        "",
        "- New model clipped to 0 to 1:",
        "",
        markdown_table(sensitivity[shown]),
        "",
        f"- Calibration factor from the peak-identified origins only: "
        f"{sens.get('factor from peak-identified origins only', float('nan')):.3f} "
        f"({sens.get('forecasts at peak-identified origins', 0)} forecasts), against {calibration.factor:.3f} from "
        "all four.",
        "",
        "## Fits",
        "",
        "```",
        *[line for r in origin_results for line in r.attempts],
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Run every origin, calibrate, write the results and print the pooled table."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    config.ensure_directories()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    panel = read_panel()
    origin_results = []
    for origin in config.BACKTEST_ORIGINS:
        print(f"Origin {origin} ...")
        result = run_origin(panel, origin)
        print("\n".join(result.attempts))
        print(f"  peak identification (new model): {result.identification}")
        origin_results.append(result)
    players = pd.concat([r.players for r in origin_results], ignore_index=True)
    players, calibration = calibrate(players, list(config.BACKTEST_ORIGINS), players["peak_identified"])
    players["repeated_from_earlier_origin"] = repeated_players(players)
    results = results_table(players)
    selection = selection_table(panel, origin_results)
    players.to_csv(config.BACKTEST_PLAYERS_PATH, index=False, encoding="utf-8-sig")
    results.to_csv(config.BACKTEST_RESULTS_PATH, index=False)
    save_calibration(calibration)
    config.BACKTEST_SUMMARY_PATH.write_text(build_summary(results, origin_results, selection, calibration, players),
                                            encoding="utf-8")
    print()
    shown = ["origin", "subset", "method", "players", "mae", "rmse", "spearman", "bias", "coverage_80",
             "coverage_80_calibrated"]
    print(results[shown].round(3).to_string(index=False))
    print()
    print("\n".join(plain_verdict(results, origin_results)))
    print()
    print("\n".join(verdict_lines(results, origin_results, calibration)))
    print()
    print(calibration_table(players, calibration).round(3).to_string(index=False))
    print(f"\nFactor for 2025/26: {calibration.factor:.3f}; sensitivity: {calibration.sensitivity}")
    print(f"\nResults: {config.BACKTEST_RESULTS_PATH}\nPlayers: {config.BACKTEST_PLAYERS_PATH}\n"
          f"Summary: {config.BACKTEST_SUMMARY_PATH}\nCalibration: {config.CALIBRATION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
