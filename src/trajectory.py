"""Mixed-effects age-curve model of Premier League minutes share, and projections from it.

The model (statsmodels MixedLM, fitted by REML):

    minutes_share = b0 + b1 * age_c + b2 * age_c^2 + position shift      (fixed effects)
                    + u0_player + u1_player * age_c                     (random effects)
                    + residual

age_c is age minus 21 (config.MODEL_AGE_CENTRE). The fixed effects are the
average age curve; each player's random intercept u0 and random slope u1 say
how far his own curve sits above or below it and how much steeper or flatter
it is. (u0, u1) are normal with mean 0 and covariance Psi, and the residual is
normal with variance sigma^2. docs/trajectory_model.md explains it in plain
English.

Fitting uses the panel rows (src/panel.py) at completed ages
config.MODEL_AGE_RANGE (17 to 30), for players first seen young only: those
whose first PL season with minutes came at config.FIRST_SEEN_YOUNG_MAX_AGE (20)
or under (training_rows). The panel starts in 2016-17, so a player first seen
older is only in it because he was still in the league; fitting on him made
the Day 2 curve too high at 23 to 26 (data/processed/backtest_summary.md).
The Day 2 model, fitted on every player, is still fitted and saved for
comparison (trajectory_model_params_all_players.json). If the random-slope
model fails the checks in check_fit, the model falls back to a random
intercept only.

Projections (used by ability.py for Potential Ability and by validate.py for
the backtest) do not use the statsmodels object. They use the fitted numbers
in TrajectoryModel and the standard formulas for a linear mixed model:

    V   = Z Psi Z' + sigma^2 I                (covariance of one player's seasons)
    A   = Psi Z' V^-1
    u   = A (y - X beta)                      (best linear unbiased prediction, BLUP)
    C   = Psi - A Z Psi                       (its conditional covariance)

where y, X and Z are the player's own seasons. main() checks that these give
the same random effects as statsmodels.

Writes data/processed/trajectory_model_summary.txt,
data/processed/trajectory_model_params.json (the main model) and
data/processed/trajectory_model_params_all_players.json (the Day 2 model).

Run from the repository root, after python -m src.panel:
    python -m src.trajectory
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src import config
from src.panel import first_seen_young, read_panel

FORMULA: str = "minutes_share ~ age_c + I(age_c ** 2) + C(position)"
RE_FORMULA_SLOPE: str = "~age_c"
RE_FORMULA_INTERCEPT: str = "~1"
RANDOM_EFFECT_NAMES_SLOPE: tuple[str, ...] = ("intercept", "age_c")
RANDOM_EFFECT_NAMES_INTERCEPT: tuple[str, ...] = ("intercept",)
POSITION_TERM: str = "C(position)[T.{}]"

# A random-effects covariance counts as singular (degenerate) if a variance is
# effectively zero or the intercept and slope are almost perfectly correlated.
MIN_RANDOM_EFFECT_VARIANCE: float = 1e-8
MAX_RANDOM_EFFECT_CORRELATION: float = 0.99
# statsmodels warns "The MLE may be on the boundary" whenever any random-effect
# variance is below 0.01 in absolute terms. Minutes share runs from 0 to 1, so
# a meaningful slope variance (in share per year, squared) is far smaller than
# that. The warning is recorded but not treated as a failure; the checks above
# test the boundary directly.
BOUNDARY_HEURISTIC_TEXT: str = "The MLE may be on the boundary"


# ---------------------------------------------------------------------------
# The data the model is fitted on
# ---------------------------------------------------------------------------

def model_rows(panel: pd.DataFrame, age_range: tuple[int, int] = config.MODEL_AGE_RANGE) -> pd.DataFrame:
    """Panel rows the model uses: completed age within age_range, with a known age and position."""
    low, high = age_range
    in_range = panel["age_completed"].between(low, high).fillna(False).astype(bool)
    rows = panel[in_range & panel["age"].notna() & panel["position"].notna()]
    return rows.reset_index(drop=True)


def training_rows(
    panel: pd.DataFrame,
    age_range: tuple[int, int] = config.MODEL_AGE_RANGE,
    first_seen_max_age: int | None = config.FIRST_SEEN_YOUNG_MAX_AGE,
) -> pd.DataFrame:
    """The rows a model is fitted on: model_rows, restricted to players first seen young.

    Args:
        panel: the player-season panel (or the part of it known at a backtest origin).
        age_range: completed ages to fit on.
        first_seen_max_age: keep only players whose first_seen_age is at most
            this (20 by default). None keeps every player, as the Day 2 model did.
    """
    rows = model_rows(panel, age_range)
    if first_seen_max_age is None:
        return rows
    return rows[first_seen_young(rows, first_seen_max_age)].reset_index(drop=True)


def training_population(first_seen_max_age: int | None) -> str:
    """Who a model was fitted on, in words."""
    if first_seen_max_age is None:
        return "all players (the Day 2 model)"
    return f"players first seen at {first_seen_max_age} or under"


# ---------------------------------------------------------------------------
# The fitted model as plain numbers
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryModel:
    """A fitted model, stored as plain numbers so it can be saved and used without statsmodels."""

    fe_names: list[str]
    fe_params: np.ndarray  # fixed effects, beta
    fe_cov: np.ndarray  # covariance of the fixed-effect estimates
    cov_re: np.ndarray  # Psi, covariance of a player's random effects
    scale: float  # sigma^2, residual variance
    random_slope: bool
    positions: list[str]
    age_centre: float = config.MODEL_AGE_CENTRE
    age_range: tuple[int, int] = config.MODEL_AGE_RANGE
    n_rows: int = 0
    n_players: int = 0
    last_season: str = ""
    first_seen_max_age: int | None = None  # None: fitted on every player

    @property
    def training_population(self) -> str:
        """Who the model was fitted on, in words."""
        return training_population(self.first_seen_max_age)

    @property
    def random_effect_names(self) -> tuple[str, ...]:
        """Names of the random effects: intercept, plus age_c if there is a random slope."""
        return RANDOM_EFFECT_NAMES_SLOPE if self.random_slope else RANDOM_EFFECT_NAMES_INTERCEPT

    @property
    def reference_position(self) -> str:
        """The position whose curve the intercept describes (the one with no position term)."""
        shifted = {name[len("C(position)[T."):-1] for name in self.fe_names if name.startswith("C(position)")}
        (reference,) = set(self.positions) - shifted
        return reference

    @classmethod
    def from_result(cls, result, rows: pd.DataFrame, random_slope: bool,
                    first_seen_max_age: int | None = None) -> TrajectoryModel:
        """Take the numbers out of a fitted statsmodels MixedLM result."""
        fe_names = list(result.fe_params.index)
        return cls(
            fe_names=fe_names,
            fe_params=result.fe_params.to_numpy(dtype=float),
            fe_cov=result.cov_params().loc[fe_names, fe_names].to_numpy(dtype=float),
            cov_re=np.asarray(result.cov_re, dtype=float),
            scale=float(result.scale),
            random_slope=random_slope,
            positions=sorted(rows["position"].unique()),
            n_rows=len(rows),
            n_players=int(rows["fpl_code"].nunique()),
            last_season=str(rows["season"].max()),
            first_seen_max_age=first_seen_max_age,
        )

    # Designs and predictions -------------------------------------------------

    def fixed_design(self, age_c: np.ndarray, positions: Sequence[str]) -> np.ndarray:
        """The fixed-effects design matrix X (one row per season) in the order of fe_names."""
        age_c = np.asarray(age_c, dtype=float)
        positions = np.asarray(positions, dtype=object)
        unknown = set(positions) - set(self.positions)
        if unknown:
            raise ValueError(f"positions not in the model: {sorted(unknown)}")
        columns = []
        for name in self.fe_names:
            if name == "Intercept":
                columns.append(np.ones_like(age_c))
            elif name == "age_c":
                columns.append(age_c)
            elif name == "I(age_c ** 2)":
                columns.append(age_c**2)
            elif name.startswith("C(position)[T."):
                columns.append((positions == name[len("C(position)[T."):-1]).astype(float))
            else:
                raise ValueError(f"unexpected fixed-effect term {name!r}")
        return np.column_stack(columns)

    def random_design(self, age_c: np.ndarray) -> np.ndarray:
        """The random-effects design matrix Z: a column of ones, plus age_c with a random slope."""
        age_c = np.asarray(age_c, dtype=float)
        return np.column_stack([np.ones_like(age_c), age_c]) if self.random_slope else np.ones((len(age_c), 1))

    def population_mean(self, age_c: np.ndarray, positions: Sequence[str]) -> np.ndarray:
        """The average curve (fixed effects only) at the given ages and positions."""
        return self.fixed_design(age_c, positions) @ self.fe_params

    def peak_age(self) -> float:
        """Age at which the average curve is highest: 21 - b1 / (2 b2), if the curve bends down."""
        b1 = self.fe_params[self.fe_names.index("age_c")]
        b2 = self.fe_params[self.fe_names.index("I(age_c ** 2)")]
        return self.age_centre - b1 / (2 * b2) if b2 < 0 else math.nan

    # Saving ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """A JSON-friendly copy."""
        return {
            "fe_names": self.fe_names,
            "fe_params": self.fe_params.tolist(),
            "fe_cov": self.fe_cov.tolist(),
            "cov_re": self.cov_re.tolist(),
            "scale": self.scale,
            "random_slope": self.random_slope,
            "positions": self.positions,
            "age_centre": self.age_centre,
            "age_range": list(self.age_range),
            "n_rows": self.n_rows,
            "n_players": self.n_players,
            "last_season": self.last_season,
            "first_seen_max_age": self.first_seen_max_age,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrajectoryModel:
        """Rebuild a model saved with to_dict."""
        return cls(
            fe_names=list(data["fe_names"]),
            fe_params=np.asarray(data["fe_params"], dtype=float),
            fe_cov=np.asarray(data["fe_cov"], dtype=float),
            cov_re=np.asarray(data["cov_re"], dtype=float),
            scale=float(data["scale"]),
            random_slope=bool(data["random_slope"]),
            positions=list(data["positions"]),
            age_centre=float(data["age_centre"]),
            age_range=tuple(data["age_range"]),
            n_rows=int(data["n_rows"]),
            n_players=int(data["n_players"]),
            last_season=str(data["last_season"]),
            # Files written before the survivorship fix have no such key: fitted on every player.
            first_seen_max_age=data.get("first_seen_max_age"),
        )


def save_model(model: TrajectoryModel, path=config.TRAJECTORY_PARAMS_PATH) -> None:
    """Write the fitted numbers to JSON."""
    path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def load_model(path=config.TRAJECTORY_PARAMS_PATH) -> TrajectoryModel:
    """Read a model written by save_model. Raises FileNotFoundError with the command to run."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.trajectory")
    return TrajectoryModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Fitting, with checks and a fallback
# ---------------------------------------------------------------------------

@dataclass
class FitAttempt:
    """One attempt at fitting: which model, what statsmodels said, and whether it passed the checks."""

    random_slope: bool
    optimiser_converged: bool
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """True if the fit passed every check."""
        return not self.problems

    def describe(self) -> str:
        """One paragraph for the log and the summary file."""
        kind = "random intercept and slope" if self.random_slope else "random intercept only"
        verdict = "accepted" if self.accepted else "rejected: " + "; ".join(self.problems)
        lines = [f"{kind}: optimiser converged = {self.optimiser_converged}; {verdict}"]
        lines += [f"  warning: {text}" for text in self.warnings] or ["  no warnings"]
        return "\n".join(lines)


def random_effect_problems(cov_re: np.ndarray) -> list[str]:
    """Reasons a random-effects covariance is degenerate (empty if it is fine)."""
    problems = []
    variances = np.diag(cov_re)
    if np.any(variances <= MIN_RANDOM_EFFECT_VARIANCE):
        problems.append(f"a random-effect variance is effectively zero ({variances.min():.2e})")
    if len(variances) == 2 and np.all(variances > 0):
        correlation = cov_re[0, 1] / math.sqrt(variances[0] * variances[1])
        if abs(correlation) >= MAX_RANDOM_EFFECT_CORRELATION:
            problems.append(f"intercept and slope are almost perfectly correlated ({correlation:.3f})")
    if np.linalg.eigvalsh(cov_re).min() <= 0:
        problems.append("random-effects covariance is not positive definite")
    return problems


def check_fit(result, caught: list[str]) -> list[str]:
    """Every reason not to trust a fit: no convergence, a bad Hessian, a singular covariance."""
    problems = []
    if not result.converged:
        problems.append("optimiser did not converge")
    for text in caught:
        if "not positive definite" in text:
            problems.append("Hessian not positive definite at the estimate")
        elif "singular" in text.lower():
            problems.append("statsmodels reports a singular random-effects covariance")
        elif "Gradient optimization failed" in text:
            problems.append("gradient optimisation failed")
    problems += random_effect_problems(np.asarray(result.cov_re, dtype=float))
    return problems


def fit_mixed_model(rows: pd.DataFrame, random_slope: bool = True):
    """Fit the MixedLM by REML, recording every warning instead of printing it.

    statsmodels tries the BFGS, L-BFGS and conjugate-gradient optimisers in
    turn until one converges.

    Returns:
        (statsmodels result, FitAttempt)
    """
    re_formula = RE_FORMULA_SLOPE if random_slope else RE_FORMULA_INTERCEPT
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.mixedlm(FORMULA, rows, groups=rows["fpl_code"], re_formula=re_formula)
        result = model.fit(reml=True)
    texts = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    attempt = FitAttempt(random_slope=random_slope, optimiser_converged=bool(result.converged), warnings=texts)
    attempt.problems = check_fit(result, texts)
    return result, attempt


def fit_with_fallback(rows: pd.DataFrame, first_seen_max_age: int | None = None):
    """Fit with a random slope; if that fails the checks, fall back to a random intercept only.

    Args:
        rows: the fitting rows (see training_rows).
        first_seen_max_age: the restriction the rows were built with, recorded
            on the model (None: every player).

    Returns:
        (statsmodels result, TrajectoryModel, list of FitAttempt in the order tried)
    """
    result, attempt = fit_mixed_model(rows, random_slope=True)
    attempts = [attempt]
    if not attempt.accepted:
        result, attempt = fit_mixed_model(rows, random_slope=False)
        attempts.append(attempt)
        if not attempt.accepted:
            raise RuntimeError("Neither model passed the checks:\n" + "\n".join(a.describe() for a in attempts))
    model = TrajectoryModel.from_result(result, rows, attempt.random_slope, first_seen_max_age)
    return result, model, attempts


# ---------------------------------------------------------------------------
# One player's random effects (BLUP) and projections
# ---------------------------------------------------------------------------

@dataclass
class PlayerPosterior:
    """What a player's own seasons say about his random effects.

    For a given beta, his random effects are normal with mean
    weights @ (y - X @ beta) and covariance cov. A player with no seasons in
    the model's age range has empty weights, so his mean is 0 and his
    covariance is Psi: the population, unchanged.
    """

    weights: np.ndarray  # A, shape (q, n)
    X: np.ndarray  # his fixed-effects design, shape (n, p)
    y: np.ndarray  # his minutes shares, shape (n,)
    cov: np.ndarray  # C, shape (q, q)

    @property
    def seasons(self) -> int:
        """How many of his seasons informed the estimate."""
        return len(self.y)

    def mean(self, beta: np.ndarray) -> np.ndarray:
        """BLUP of his random effects for one beta (shape (p,)) or many (shape (draws, p))."""
        beta = np.atleast_2d(beta)
        if self.seasons == 0:
            return np.zeros((beta.shape[0], self.cov.shape[0]))
        residuals = self.y[None, :] - beta @ self.X.T  # (draws, n)
        return residuals @ self.weights.T  # (draws, q)


def player_posterior(model: TrajectoryModel, age_c: np.ndarray, positions: Sequence[str], shares: np.ndarray) -> PlayerPosterior:
    """The conditional distribution of one player's random effects given his seasons."""
    psi = model.cov_re
    if len(shares) == 0:
        return PlayerPosterior(np.zeros((psi.shape[0], 0)), np.zeros((0, len(model.fe_names))), np.zeros(0), psi.copy())
    X = model.fixed_design(age_c, positions)
    Z = model.random_design(age_c)
    V = Z @ psi @ Z.T + model.scale * np.eye(len(shares))
    weights = np.linalg.solve(V, Z @ psi).T  # Psi Z' V^-1 (V is symmetric)
    cov = psi - weights @ Z @ psi
    cov = (cov + cov.T) / 2  # remove rounding asymmetry
    return PlayerPosterior(weights, X, np.asarray(shares, dtype=float), cov)


def draw_fixed_effects(model: TrajectoryModel, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Draws of beta from its estimated sampling distribution, shape (draws, p)."""
    return rng.multivariate_normal(model.fe_params, model.fe_cov, size=draws)


def simulate_window_mean(
    model: TrajectoryModel,
    posterior: PlayerPosterior,
    target_age_c: np.ndarray,
    position: str,
    beta_draws: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulated mean minutes share over the target seasons, one value per draw.

    Each draw combines three sources of uncertainty:
    1. the fixed effects (beta, from beta_draws);
    2. the player's own random effects given beta (his BLUP plus noise from
       its conditional covariance);
    3. the season-to-season residual (normal with variance sigma^2, drawn
       independently for each target season).
    The variance parameters (Psi and sigma^2) are treated as known.
    """
    target_age_c = np.asarray(target_age_c, dtype=float)
    draws = beta_draws.shape[0]
    X_new = model.fixed_design(target_age_c, [position] * len(target_age_c))
    Z_new = model.random_design(target_age_c)
    u = posterior.mean(beta_draws) + rng.multivariate_normal(np.zeros(posterior.cov.shape[0]), posterior.cov, size=draws)
    noise = rng.normal(0.0, math.sqrt(model.scale), size=(draws, len(target_age_c)))
    shares = beta_draws @ X_new.T + u @ Z_new.T + noise
    return shares.mean(axis=1)


def project(
    model: TrajectoryModel,
    history: pd.DataFrame,
    targets: pd.DataFrame,
    draws: int = config.SIMULATION_DRAWS,
    level: float = config.INTERVAL_LEVEL,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Predict each player's mean minutes share over his target seasons.

    Args:
        model: the fitted model.
        history: the seasons that inform each player's random effects, with
            fpl_code, age_c, position and minutes_share. Only rows inside the
            model's age range should be passed (see model_rows).
        targets: the seasons to predict, with fpl_code, age_c and position
            (one position per player). Every player in targets is projected,
            with or without history.
        draws: Monte Carlo draws per player.
        level: coverage of the interval (0.8 gives the 10th to 90th percentile).
        seed: random seed. The same fixed-effect draws are used for every player.

    Returns:
        One row per player: history_seasons, target_seasons, re_intercept and
        re_slope (BLUPs at the estimated beta), population (fixed effects
        only), predicted (fixed effects plus BLUP), and the simulated median,
        lower and upper bounds. Nothing is clipped.
    """
    rng = np.random.default_rng(seed)
    beta_draws = draw_fixed_effects(model, draws, rng)
    low_q, high_q = (1 - level) / 2, 1 - (1 - level) / 2
    by_player = {code: group for code, group in history.groupby("fpl_code")}
    rows = []
    for code, target in targets.groupby("fpl_code", sort=True):
        positions = target["position"].unique()
        if len(positions) != 1:
            raise ValueError(f"player {code} has more than one target position: {positions}")
        position = positions[0]
        past = by_player.get(code, history.iloc[0:0])
        posterior = player_posterior(model, past["age_c"].to_numpy(), past["position"].to_numpy(), past["minutes_share"].to_numpy())
        blup = posterior.mean(model.fe_params)[0]
        age_c = target["age_c"].to_numpy(dtype=float)
        population = model.population_mean(age_c, [position] * len(age_c))
        predicted = population + model.random_design(age_c) @ blup
        simulated = simulate_window_mean(model, posterior, age_c, position, beta_draws, rng)
        rows.append(
            {
                "fpl_code": code,
                "history_seasons": posterior.seasons,
                "target_seasons": len(age_c),
                "re_intercept": blup[0],
                "re_slope": blup[1] if model.random_slope else np.nan,
                "population": population.mean(),
                "predicted": predicted.mean(),
                "sim_median": float(np.median(simulated)),
                "sim_lower": float(np.quantile(simulated, low_q)),
                "sim_upper": float(np.quantile(simulated, high_q)),
            }
        )
    return pd.DataFrame(rows)


def future_targets(players: pd.DataFrame, ages: tuple[int, int]) -> pd.DataFrame:
    """Future seasons at which each player will be within an age window (completed years).

    Args:
        players: one row per player with fpl_code, season_year, age (decimal),
            age_completed and position, all for his latest season.
        ages: inclusive window in completed years, such as (24, 26).

    Returns:
        One row per player per target season: fpl_code, season_year, age,
        age_completed, age_c and position. A season k years ahead has age
        plus k (to within a leap day) and completed age plus k exactly.
    """
    low, high = ages
    rows = []
    for player in players.itertuples(index=False):
        for target_age in range(low, high + 1):
            years_ahead = target_age - int(player.age_completed)
            if years_ahead < 1:
                continue  # only future seasons are projected
            age = float(player.age) + years_ahead
            rows.append(
                {
                    "fpl_code": player.fpl_code,
                    "season_year": int(player.season_year) + years_ahead,
                    "age": age,
                    "age_completed": target_age,
                    "age_c": age - config.MODEL_AGE_CENTRE,
                    "position": player.position,
                }
            )
    return pd.DataFrame(rows, columns=["fpl_code", "season_year", "age", "age_completed", "age_c", "position"])


# ---------------------------------------------------------------------------
# Checks and the summary file
# ---------------------------------------------------------------------------

def blup_check(result, model: TrajectoryModel, rows: pd.DataFrame) -> tuple[float, float]:
    """Largest gap between this module's BLUPs (and their covariances) and statsmodels' own."""
    worst_mean, worst_cov = 0.0, 0.0
    for code, group in rows.groupby("fpl_code"):
        posterior = player_posterior(model, group["age_c"].to_numpy(), group["position"].to_numpy(), group["minutes_share"].to_numpy())
        ours = posterior.mean(model.fe_params)[0]
        theirs = np.asarray(result.random_effects[code], dtype=float)
        worst_mean = max(worst_mean, float(np.abs(ours - theirs).max()))
        worst_cov = max(worst_cov, float(np.abs(posterior.cov - np.asarray(result.random_effects_cov[code])).max()))
    return worst_mean, worst_cov


def chi2_mixture_p_value(statistic: float) -> float:
    """p-value for adding a random slope: a 50:50 mix of chi-squared with 1 and 2 degrees of freedom.

    Testing a variance against zero puts it on the edge of its range, so the
    usual chi-squared(2) reference is too conservative; the 50:50 mixture is
    the standard correction (Self and Liang, 1987). Closed forms avoid a scipy
    dependency: P(chi2_1 > x) = erfc(sqrt(x / 2)), P(chi2_2 > x) = exp(-x / 2).
    """
    if statistic <= 0:
        return 1.0
    return 0.5 * math.erfc(math.sqrt(statistic / 2)) + 0.5 * math.exp(-statistic / 2)


def curve_table(model: TrajectoryModel) -> pd.DataFrame:
    """The average curve (fixed effects only) by position, at each age in the model range."""
    ages = np.arange(model.age_range[0], model.age_range[1] + 1, dtype=float)
    table = {"age": ages.astype(int)}
    for position in model.positions:
        table[position] = model.population_mean(ages - model.age_centre, [position] * len(ages)).round(3)
    return pd.DataFrame(table)


def describe_random_effects(model: TrajectoryModel) -> list[str]:
    """Standard deviations (and correlation) of the random effects and the residual, in words."""
    psi = model.cov_re
    lines = [
        f"Residual SD (season-to-season noise): {math.sqrt(model.scale):.3f} minutes share",
        f"Random intercept SD (player-to-player spread at age 21): {math.sqrt(psi[0, 0]):.3f}",
    ]
    if model.random_slope:
        correlation = psi[0, 1] / math.sqrt(psi[0, 0] * psi[1, 1])
        lines += [
            f"Random slope SD (spread of yearly change around the average): {math.sqrt(psi[1, 1]):.4f} per year",
            f"Correlation between intercept and slope: {correlation:.3f}",
        ]
    share_between = psi[0, 0] / (psi[0, 0] + model.scale)
    lines.append(f"Share of variance at age 21 that is between players (ICC): {share_between:.2f}")
    return lines


def build_summary(result, model: TrajectoryModel, attempts: list[FitAttempt], rows: pd.DataFrame,
                  extra: list[str]) -> str:
    """The text written to trajectory_model_summary.txt."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    low, high = model.age_range
    lines = [
        "Trajectory model: PL minutes share by age (statsmodels MixedLM, REML)",
        f"Generated by src/trajectory.py at {stamp} (UTC) from data/processed/player_season_panel.parquet",
        "",
        f"Formula: {FORMULA}",
        f"Random effects per player: {', '.join(model.random_effect_names)}",
        f"Fitted on: {model.training_population}",
        f"Age centred at {model.age_centre:g}; fitted on completed ages {low} to {high}",
        f"Rows: {model.n_rows} player-seasons, {model.n_players} players, seasons up to {model.last_season}",
        f"Filled zeros among them: {int(rows['filled'].sum())}; observed zeros: "
        f"{int(rows['zero_source'].eq('observed').fillna(False).sum())}",
        "",
        "Fit attempts:",
        *[attempt.describe() for attempt in attempts],
        "",
        str(result.summary()),
        "",
        "Random effects and residual:",
        *describe_random_effects(model),
        "",
        f"Peak of the average curve: age {model.peak_age():.1f} (the same for every position, "
        f"since position only shifts the curve up or down). Reference position: {model.reference_position}.",
        "",
        "Average curve (fixed effects only) by position:",
        curve_table(model).to_string(index=False),
        "",
        *extra,
    ]
    return "\n".join(lines) + "\n"


def rows_by_age(rows: pd.DataFrame) -> pd.DataFrame:
    """Fitting rows per completed age: seasons, players, filled zeros and mean share."""
    return rows.groupby("age_completed").agg(
        seasons=("fpl_code", "size"), players=("fpl_code", "nunique"), filled=("filled", "sum"),
        mean_share=("minutes_share", "mean"),
    ).round(3).reset_index()


def compare_models(models: dict[str, TrajectoryModel]) -> list[str]:
    """Side-by-side fixed effects, spreads and average curves of models fitted on different players."""
    names = list(models)
    params = pd.DataFrame({name: pd.Series(m.fe_params, index=m.fe_names) for name, m in models.items()})
    spreads = pd.DataFrame({
        name: {
            "rows": m.n_rows,
            "players": m.n_players,
            "random slope": m.random_slope,
            "peak age of the average curve": round(m.peak_age(), 2),
            "random intercept SD": round(math.sqrt(m.cov_re[0, 0]), 4),
            "random slope SD": round(math.sqrt(m.cov_re[1, 1]), 4) if m.random_slope else np.nan,
            "residual SD": round(math.sqrt(m.scale), 4),
        }
        for name, m in models.items()
    })
    curves = {"age": curve_table(models[names[0]])["age"]}
    for position in models[names[0]].positions:
        for name in names:
            curves[f"{position}, {name}"] = curve_table(models[name])[position]
    return [
        "Fixed effects:",
        params.round(4).to_string(),
        "",
        spreads.to_string(),
        "",
        "Average curve (fixed effects only) by position:",
        pd.DataFrame(curves).to_string(index=False),
    ]


def main() -> int:
    """Fit the main model (players first seen young) and the Day 2 model, run the checks, write the files."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    config.ensure_directories()
    panel = read_panel()
    young_age = config.FIRST_SEEN_YOUNG_MAX_AGE
    rows = training_rows(panel, first_seen_max_age=young_age)
    all_rows = training_rows(panel, first_seen_max_age=None)
    print(f"Fitting on {len(rows)} rows, {rows['fpl_code'].nunique()} players ({training_population(young_age)}; "
          f"all players: {len(all_rows)} rows, {all_rows['fpl_code'].nunique()} players) ...")
    result, model, attempts = fit_with_fallback(rows, young_age)
    for attempt in attempts:
        print(attempt.describe())

    low, high = config.PEAK_AGE_RANGE
    at_peak = rows["age_completed"].between(low, high).fillna(False).astype(bool)
    extra = [
        f"Rows left out because the player was first seen at {young_age + 1} or older: {len(all_rows) - len(rows)} "
        f"({all_rows['fpl_code'].nunique() - rows['fpl_code'].nunique()} players)",
        f"Fitting rows at the peak ages {low} to {high}: {int(at_peak.sum())} seasons, "
        f"{rows.loc[at_peak, 'fpl_code'].nunique()} players",
        "",
        "Fitting rows by age (completed years):",
        rows_by_age(rows).to_string(index=False),
        "",
    ]
    if model.random_slope:
        # Same fixed effects, so the REML likelihoods can be compared.
        simple, simple_attempt = fit_mixed_model(rows, random_slope=False)
        statistic = 2 * (result.llf - simple.llf)
        extra += [
            "Random slope against random intercept only (REML likelihood ratio test):",
            f"  log-likelihood {result.llf:.1f} against {simple.llf:.1f}; statistic {statistic:.1f}; "
            f"p-value {chi2_mixture_p_value(statistic):.2g} (50:50 mixture of chi-squared 1 and 2)",
            f"  intercept-only fit: {simple_attempt.describe().splitlines()[0]}",
            "",
        ]
    worst_mean, worst_cov = blup_check(result, model, rows)
    extra += [
        "Check of the projection code: random effects worked out with the formulas in src/trajectory.py",
        f"  largest difference from statsmodels' random_effects: {worst_mean:.2e}",
        f"  largest difference from statsmodels' random_effects_cov: {worst_cov:.2e}",
        "",
    ]

    print("Fitting the Day 2 model (all players) for comparison ...")
    _, all_model, all_attempts = fit_with_fallback(all_rows, None)
    for attempt in all_attempts:
        print(attempt.describe())
    extra += [
        "Comparison with the Day 2 model, fitted on every player (saved to trajectory_model_params_all_players.json):",
        *[attempt.describe() for attempt in all_attempts],
        "",
        *compare_models({"first seen young": model, "all players": all_model}),
    ]
    summary = build_summary(result, model, attempts, rows, extra)
    config.TRAJECTORY_SUMMARY_PATH.write_text(summary, encoding="utf-8")
    save_model(model)
    save_model(all_model, config.TRAJECTORY_ALL_PLAYERS_PARAMS_PATH)
    print("\n".join(summary.splitlines()[-(len(extra) + 25):]))
    print(f"\nSummary written to {config.TRAJECTORY_SUMMARY_PATH}")
    print(f"Parameters written to {config.TRAJECTORY_PARAMS_PATH} and {config.TRAJECTORY_ALL_PLAYERS_PARAMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
