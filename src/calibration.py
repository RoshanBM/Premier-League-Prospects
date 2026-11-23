"""Widening the projection intervals using the backtest's own errors.

The Day 2 backtest (validate.py) found that the model's 80% intervals held
the actual outcome far less often than 80% of the time. This module widens
them by one scale factor s, estimated from past forecast errors, without
using any forecast's own outcome to widen its interval.

For a past forecast with median m, interval bounds l and u, and actual
outcome y, the factor that would just stretch the interval to reach y is

    s_i = (y - m) / (u - m)     if y > m
    s_i = (m - y) / (m - l)     otherwise

so s_i <= 1 means the raw interval already held it. s is the 80th percentile
of the s_i (config.INTERVAL_LEVEL), and never below 1, so intervals are never
made narrower. The calibrated interval keeps the median and stretches each
side by s:

    lower = m - s (m - l),    upper = m + s (u - m)

Rolling, to avoid leakage: in the backtest, the s used at origin T is
estimated only from forecasts made at origins before T (rolling_factors), so
the first origin stays uncalibrated. The 2025/26 projections use s estimated
from every origin.

validate.py saves the result to data/processed/interval_calibration.json;
ability.py and figures.py read it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def scale_needed(actual, median, lower, upper) -> np.ndarray:
    """For each forecast, the factor s_i that would just stretch its interval to the actual outcome.

    0 when the outcome equals the median; at most 1 when the raw interval
    already held it. Raises ValueError if an interval has no width on the
    side of the outcome, where s_i would not be defined.
    """
    actual, median, lower, upper = (np.asarray(x, dtype=float) for x in (actual, median, lower, upper))
    above = actual > median
    half_width = np.where(above, upper - median, median - lower)
    gap = np.where(above, actual - median, median - actual)
    if np.any(half_width <= 0):
        raise ValueError("an interval has no width on the side of its outcome")
    return gap / half_width


def calibration_factor(scales, level: float = config.INTERVAL_LEVEL) -> float:
    """The level-th percentile of the s_i, never below 1. With no past forecasts, 1 (no widening)."""
    scales = np.asarray(scales, dtype=float)
    if scales.size == 0:
        return 1.0
    return max(1.0, float(np.quantile(scales, level)))


def widen(median, lower, upper, factor) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated (lower, upper): each side of the interval stretched by `factor` around the median.

    `factor` is one number, or one per interval (as in the backtest, where it
    depends on the origin).
    """
    factor = np.asarray(factor, dtype=float)
    if np.any(factor < 1):
        raise ValueError(f"a calibration factor below 1 would narrow the interval: {factor.min()}")
    median = np.asarray(median, dtype=float)
    return median - factor * (median - np.asarray(lower, dtype=float)), median + factor * (np.asarray(upper, dtype=float) - median)


# ---------------------------------------------------------------------------
# Rolling factors for the backtest
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RollingFactor:
    """The factor used at one backtest origin, and which earlier forecasts it came from."""

    origin: str
    factor: float
    origins_used: tuple[str, ...]
    forecasts_used: int

    @property
    def calibrated(self) -> bool:
        """False for an origin with no earlier forecasts to learn from (the first one)."""
        return self.forecasts_used > 0


def rolling_factors(forecasts: pd.DataFrame, origins: Sequence[str],
                    level: float = config.INTERVAL_LEVEL) -> list[RollingFactor]:
    """For each origin, the factor estimated only from forecasts made at earlier origins.

    Args:
        forecasts: one row per backtest forecast, with `origin` and
            `scale_needed` (from scale_needed).
        origins: the backtest origins, in time order.
        level: the percentile of the s_i to use.

    Raises:
        ValueError: if the origins are not in time order, since "earlier"
            would then be wrong.
    """
    years = [config.season_start_year(origin) for origin in origins]
    if years != sorted(years) or len(set(years)) != len(years):
        raise ValueError(f"origins must be distinct and in time order: {list(origins)}")
    factors = []
    for i, origin in enumerate(origins):
        earlier = tuple(origins[:i])
        used = forecasts.loc[forecasts["origin"].isin(earlier), "scale_needed"]
        factors.append(RollingFactor(origin, calibration_factor(used, level), earlier, int(len(used))))
    return factors


# ---------------------------------------------------------------------------
# What is saved for the projections
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    """The calibration decision and factors, as saved to interval_calibration.json."""

    triggered: bool  # False if coverage was already good enough, and every factor is 1
    reason: str
    raw_coverage: float  # pooled coverage of the raw interval in the backtest
    trigger_coverage: float
    level: float
    factor: float  # for projections made now: estimated from every origin
    origins_used: list[str]
    forecasts_used: int
    first_seen_max_age: int | None  # the model the errors came from
    rolling: list[dict] = field(default_factory=list)  # one RollingFactor (as a dict) per origin
    sensitivity: dict = field(default_factory=dict)  # other factors worth knowing, not used for projections
    generated: str = ""

    def rolling_factor(self, origin: str) -> float:
        """The factor used at one backtest origin."""
        return next(r["factor"] for r in self.rolling if r["origin"] == origin)


def needs_calibration(raw_coverage: float, trigger: float = config.CALIBRATION_TRIGGER_COVERAGE) -> bool:
    """True if the raw interval's pooled coverage is below the trigger."""
    return raw_coverage < trigger


def save_calibration(calibration: Calibration, path: Path = config.CALIBRATION_PATH) -> None:
    """Write the calibration to JSON."""
    path.write_text(json.dumps(asdict(calibration), indent=2), encoding="utf-8")


def load_calibration(path: Path = config.CALIBRATION_PATH) -> Calibration:
    """Read a calibration written by save_calibration. Raises FileNotFoundError with the command to run."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.validate")
    return Calibration(**json.loads(path.read_text(encoding="utf-8")))
