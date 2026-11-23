"""Trajectory charts for a quick visual check of the model's projections.

For each chosen 2025/26 U21 player, one interactive chart (Plotly, saved as
HTML) of PL minutes share against age:
- his seasons so far (a square is a season out of the PL, counted as 0; a
  hollow marker is a season outside the model's age range, which the model
  does not use);
- the model's projection for him (fixed effects plus his own random effects;
  the model fitted on players first seen at 20 or under) from next season to
  age 26, with an 80% band for each season;
- his projected peak (mean of ages 24 to 26) with the card's 80% interval;
- the average curve for his position (fixed effects only), for reference.
The title gives the card's range text and PA (rank among 2025/26 U21
prospects), as in pa_projections_2025_26.parquet.

A PA flagged as low evidence (under 450 career PL minutes, pa_display
"greyed") is drawn in grey: the band, the projection line and the peak marker.
A player with no PA (all his seasons so far before 17) gets his seasons and
the average curve only, with the reason in the title. Projected values are
clipped to 0 to 1 for display only.

The charts show the same version of the interval as the card
(config.CARD_INTERVAL, "raw"). With "calibrated", every band is widened by
the backtest's factor; that factor was estimated for the 3-season peak mean,
so applying it to single seasons would be an approximation.

Charts go to data/processed/figures/ (gitignored), with one shared copy of
plotly.min.js, so they open offline in a browser.

Run from the repository root, after python -m src.validate and python -m src.ability:
    python -m src.figures                       # the default four players
    python -m src.figures --players Hall Diouf  # FPL web names
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import config
from src.ability import PA_DISPLAY_GREYED, PA_STATUS_EXPERIMENTAL, card_bounds
from src.calibration import load_calibration, widen
from src.clean import normalise_name
from src.panel import read_panel
from src.trajectory import (
    TrajectoryModel,
    draw_fixed_effects,
    load_model,
    model_rows,
    player_posterior,
    simulate_window_mean,
)

# One player with plenty of history, one with a little, one with none in the
# model's age range, and one whose PA is low evidence (1 career PL minute).
# See docs/day2_log.md.
DEFAULT_PLAYERS: tuple[str, ...] = ("Hall", "Nwaneri", "Dowman", "Oriola")

# Colours from the dataviz reference palette (light mode): the player is
# series 1; the average curve is secondary ink, so it reads as context.
PLAYER_COLOUR = "#2a78d6"
BAND_COLOUR = "rgba(42, 120, 214, 0.18)"
REFERENCE_COLOUR = "#52514e"
TEXT_COLOUR = "#0b0b0b"
GRID_COLOUR = "#e4e3df"
SURFACE_COLOUR = "#fcfcfb"
# A low-evidence projection: muted ink, lighter than the average curve so the two stay apart.
GREYED_COLOUR = "#a3a29d"
GREYED_BAND_COLOUR = "rgba(163, 162, 157, 0.22)"


@dataclass(frozen=True)
class ProjectionStyle:
    """Colours for the projection elements: the player's colour, or grey for a low-evidence PA."""

    line: str
    band: str

    @classmethod
    def for_display(cls, display: str) -> ProjectionStyle:
        """The style for a pa_display value ("normal" or "greyed")."""
        return cls(GREYED_COLOUR, GREYED_BAND_COLOUR) if display == PA_DISPLAY_GREYED else cls(PLAYER_COLOUR, BAND_COLOUR)


def find_player(panel: pd.DataFrame, web_name: str) -> int:
    """The fpl_code of the 2025/26 player with this FPL web name (accents and case ignored)."""
    current = panel[panel["season"] == config.CURRENT_SEASON]
    hits = current[current["web_name"].map(normalise_name) == normalise_name(web_name)]
    if len(hits) != 1:
        raise ValueError(f"expected one {config.CURRENT_SEASON} player called {web_name!r}, found {len(hits)}")
    return int(hits["fpl_code"].iloc[0])


def season_bands(model: TrajectoryModel, history: pd.DataFrame, latest: pd.Series, factor: float = 1.0,
                 last_age: int = config.PEAK_AGE_RANGE[1], seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Projected share for each future season up to last_age: point projection, median and 80% band.

    The band is widened around the median by `factor` (the calibration factor; 1 leaves it as simulated).
    """
    rng = np.random.default_rng(seed)
    beta_draws = draw_fixed_effects(model, config.SIMULATION_DRAWS, rng)
    posterior = player_posterior(model, history["age_c"].to_numpy(), history["position"].to_numpy(),
                                 history["minutes_share"].to_numpy())
    blup = posterior.mean(model.fe_params)[0]
    low_q, high_q = (1 - config.INTERVAL_LEVEL) / 2, 1 - (1 - config.INTERVAL_LEVEL) / 2
    rows = []
    for years_ahead in range(1, last_age - int(latest["age_completed"]) + 1):
        age = float(latest["age"]) + years_ahead
        age_c = np.array([age - model.age_centre])
        draws = simulate_window_mean(model, posterior, age_c, latest["position"], beta_draws, rng)
        point = model.population_mean(age_c, [latest["position"]])[0] + (model.random_design(age_c) @ blup)[0]
        median = float(np.median(draws))
        lower, upper = widen(median, np.quantile(draws, low_q), np.quantile(draws, high_q), factor)
        rows.append({"age": age, "point": point, "median": median, "lower": float(lower), "upper": float(upper)})
    return pd.DataFrame(rows)


def subtitle_for(projection: pd.Series) -> str:
    """The lines under the title: the card's range text, its PA and status, then the numbers behind them."""
    if projection["pa_status"] != PA_STATUS_EXPERIMENTAL:
        return (f"No projection: {projection['pa_status']} (every season so far is outside the model's ages).<br>"
                f"Mean share so far {projection['persistence_share']:.2f}.")
    evidence = (f"; shown in grey: low evidence (under {config.PA_LOW_EVIDENCE_MINUTES} PL minutes so far)"
                if projection["pa_display"] == PA_DISPLAY_GREYED else "")
    return (f"{projection['pa_range_text']}.<br>PA {projection['pa']} ({projection['pa_label']}), "
            f"{projection['pa_status']}{evidence}.<br>Projected peak {projection['peak_median_display']:.2f}; "
            f"mean share so far {projection['persistence_share']:.2f}; {int(projection['pl_minutes_to_date'])} PL "
            f"minutes; {int(projection['history_seasons'])} season(s) inform the player's own curve. Values clipped "
            f"to 0 to 1.")


def trajectory_figure(model: TrajectoryModel, panel: pd.DataFrame, pa: pd.DataFrame, code: int,
                      factor: float = 1.0) -> go.Figure:
    """The chart for one player; `factor` is the calibration factor for the bands."""
    seasons = panel[panel["fpl_code"] == code].sort_values("season_year")
    latest = seasons.iloc[-1]
    history = model_rows(seasons, model.age_range)
    used = seasons["season"].isin(history["season"]).to_numpy()  # inside the model's age range
    filled = seasons["filled"].to_numpy(dtype=bool)
    kind = np.select([filled, ~used], ["out of the PL (counted as 0)", "outside the model's ages (not used)"],
                     default="used by the model")
    projection = pa[pa["fpl_code"] == code].iloc[0]
    projected = projection["pa_status"] == PA_STATUS_EXPERIMENTAL
    style = ProjectionStyle.for_display(projection["pa_display"])
    ages = np.arange(model.age_range[0], model.age_range[1] + 1, dtype=float)
    average = model.population_mean(ages - model.age_centre, [latest["position"]] * len(ages))
    band_name = f"{config.INTERVAL_LEVEL:.0%} band, each season" + (f" (widened x{factor:.2f})" if factor > 1 else "")

    figure = go.Figure()
    if projected:
        bands = season_bands(model, history, latest, factor)
        figure.add_trace(go.Scatter(
            x=np.r_[bands["age"], bands["age"][::-1]],
            y=np.r_[bands["upper"].clip(0, 1), bands["lower"].clip(0, 1)[::-1]],
            mode="lines", fill="toself", fillcolor=style.band, line={"width": 0}, hoverinfo="skip", name=band_name))
    figure.add_trace(go.Scatter(
        x=ages, y=average, mode="lines", line={"color": REFERENCE_COLOUR, "width": 2, "dash": "dash"},
        name=f"average {latest['position']} (fixed effects)",
        hovertemplate="age %{x:.0f}: average %{y:.2f}<extra></extra>"))
    if projected:
        figure.add_trace(go.Scatter(
            x=np.r_[latest["age"], bands["age"]], y=np.r_[latest["minutes_share"], bands["point"].clip(0, 1)],
            mode="lines", line={"color": style.line, "width": 2, "dash": "dot"}, name="projection",
            hovertemplate="age %{x:.1f}: projected %{y:.2f}<extra></extra>"))
    figure.add_trace(go.Scatter(
        x=seasons["age"], y=seasons["minutes_share"], mode="lines+markers",
        line={"color": PLAYER_COLOUR, "width": 2},
        marker={"size": 10, "color": np.where(used, PLAYER_COLOUR, SURFACE_COLOUR),
                "symbol": np.where(filled, "square", "circle"), "line": {"color": PLAYER_COLOUR, "width": 2}},
        customdata=np.c_[seasons["season"], seasons["minutes"], kind],
        name="seasons so far (square: out of the PL; hollow: outside the model's ages)",
        hovertemplate="%{customdata[0]}, age %{x:.1f}: %{customdata[1]} minutes, share %{y:.2f} "
                      "(%{customdata[2]})<extra></extra>"))
    if projected:
        # Centre of his own peak seasons, in decimal age (completed age 25 falls at his age now + the gap).
        peak_age = float(latest["age"]) + float(np.mean(config.PEAK_AGE_RANGE)) - int(latest["age_completed"])
        median = projection["peak_median_display"]
        lower_column, upper_column = card_bounds(projection["card_interval"])
        lower, upper = projection[lower_column], projection[upper_column]
        figure.add_trace(go.Scatter(
            x=[peak_age], y=[median], mode="markers",
            marker={"size": 12, "symbol": "diamond", "color": style.line,
                    "line": {"color": SURFACE_COLOUR, "width": 2}},
            error_y={"type": "data", "symmetric": False, "array": [upper - median], "arrayminus": [median - lower],
                     "color": style.line, "thickness": 2, "width": 8},
            name=f"projected peak, mean of ages {config.PEAK_AGE_RANGE[0]} to {config.PEAK_AGE_RANGE[1]}",
            hovertemplate=(f"peak: median {median:.2f}, {projection['card_interval']} {config.INTERVAL_LEVEL:.0%} "
                           f"interval {lower:.2f} to {upper:.2f}; PA {projection['pa']} ({projection['pa_label']}), "
                           f"{projection['pa_status']}<extra></extra>")))
    title = (f"{latest['name']} ({latest['position']}, {latest['club']}): PL minutes share by age<br>"
             f"<sup>{subtitle_for(projection)}</sup>")
    figure.update_layout(
        title={"text": title, "x": 0.01}, paper_bgcolor=SURFACE_COLOUR, plot_bgcolor=SURFACE_COLOUR,
        font={"color": TEXT_COLOUR, "family": "Segoe UI, Arial, sans-serif"},
        xaxis={"title": "age on 1 September", "range": [model.age_range[0] - 2, model.age_range[1] + 0.5],
               "gridcolor": GRID_COLOUR, "zeroline": False, "dtick": 1},
        yaxis={"title": "PL minutes share (minutes / 3,420)", "range": [-0.02, 1.02], "gridcolor": GRID_COLOUR,
               "zeroline": False, "tickformat": ".0%"},
        legend={"orientation": "h", "y": -0.2}, hovermode="closest", margin={"t": 130},
    )
    return figure


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Save trajectory charts for 2025/26 U21 players.")
    parser.add_argument("--players", nargs="+", default=list(DEFAULT_PLAYERS), metavar="WEB_NAME",
                        help="FPL web names (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Draw and save one chart per player."""
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panel = read_panel()
    model = load_model()
    # Bands match the card: widened only if the card shows the calibrated interval.
    factor = load_calibration().factor if config.CARD_INTERVAL == "calibrated" else 1.0
    pa = pd.read_parquet(config.PA_PROJECTIONS_PATH)
    for web_name in args.players:
        code = find_player(panel, web_name)
        path = config.FIGURES_DIR / f"trajectory_{normalise_name(web_name).replace(' ', '_')}.html"
        trajectory_figure(model, panel, pa, code, factor).write_html(path, include_plotlyjs="directory")
        row = pa[pa["fpl_code"] == code].iloc[0]
        if row["pa_status"] == PA_STATUS_EXPERIMENTAL:
            outcome = f"PA {row['pa']} ({row['pa_display']}); peak {row['peak_median_display']:.2f}; {row['pa_range_text']}"
        else:
            outcome = f"no projection ({row['pa_status']})"
        print(f"{web_name}: {int(row['history_seasons'])} season(s) in the model range; {outcome}; written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
