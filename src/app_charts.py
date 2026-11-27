"""Charts for the Streamlit player card: an attribute pizza (mplsoccer) and a CA contribution bar (Plotly).

Colours match src/figures.py (the dataviz reference palette, light mode). Each
chart shows one player, so it uses one colour for every slice or bar: length
carries the value, and text stays in the ink colours.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # draw off screen; Streamlit shows the finished figure

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure
from mplsoccer import PyPizza

from src import config
from src.figures import GRID_COLOUR, PLAYER_COLOUR, REFERENCE_COLOUR, SURFACE_COLOUR, TEXT_COLOUR

# A pizza needs at least this many slices to read as a shape; goalkeepers have one attribute.
MIN_PIZZA_ATTRIBUTES: int = 3


def pizza_figure(attributes: dict[str, int]) -> Figure:
    """An mplsoccer pizza of 1 to 20 attributes: one slice per attribute, its length the rating.

    Args:
        attributes: display name -> rating (1 to 20), in the order to draw.

    Raises:
        ValueError: with fewer than MIN_PIZZA_ATTRIBUTES attributes.
    """
    if len(attributes) < MIN_PIZZA_ATTRIBUTES:
        raise ValueError(f"a pizza needs at least {MIN_PIZZA_ATTRIBUTES} attributes, got {len(attributes)}")
    params = [name.replace(" ", "\n") for name in attributes]
    values = [int(value) for value in attributes.values()]
    pizza = PyPizza(
        params=params,
        min_range=[config.ATTRIBUTE_MIN] * len(values),
        max_range=[config.ATTRIBUTE_MAX] * len(values),
        background_color=SURFACE_COLOUR,
        straight_line_color=GRID_COLOUR,
        straight_line_lw=1,
        last_circle_color=REFERENCE_COLOUR,
        last_circle_lw=1,
        other_circle_color=GRID_COLOUR,
        other_circle_lw=1,
        other_circle_ls="-",
        inner_circle_size=8,
    )
    figure, _ = pizza.make_pizza(
        values,
        figsize=(6, 6),
        param_location=119,
        slice_colors=[PLAYER_COLOUR] * len(values),
        value_colors=[TEXT_COLOUR] * len(values),
        value_bck_colors=[SURFACE_COLOUR] * len(values),
        blank_alpha=0.12,
        color_blank_space="same",
        kwargs_slices={"edgecolor": SURFACE_COLOUR, "zorder": 2, "linewidth": 2},
        kwargs_params={"color": TEXT_COLOUR, "fontsize": 11, "va": "center"},
        kwargs_values={"color": TEXT_COLOUR, "fontsize": 11, "zorder": 3,
                       "bbox": {"edgecolor": PLAYER_COLOUR, "facecolor": SURFACE_COLOUR,
                                "boxstyle": "round,pad=0.25", "lw": 1}},
    )
    figure.set_facecolor(SURFACE_COLOUR)
    return figure


def close_figure(figure: Figure) -> None:
    """Free a Matplotlib figure once Streamlit has drawn it."""
    plt.close(figure)


def ca_contribution_figure(contributions: pd.DataFrame, ca: float) -> go.Figure:
    """Horizontal bars of each attribute's contribution (weight x rating) to CA; they add up to CA.

    Args:
        contributions: app_data.ca_contributions output (attribute, weight, rating, contribution, reason).
        ca: the player's CA, for the title.
    """
    ordered = contributions.iloc[::-1]  # Plotly draws the first bar at the bottom; largest goes on top
    labels = [f"{c:.1f}  ({w:.2f} x {r})" for c, w, r in zip(ordered["contribution"], ordered["weight"],
                                                              ordered["rating"])]
    figure = go.Figure(go.Bar(
        x=ordered["contribution"], y=ordered["attribute"], orientation="h",
        marker={"color": PLAYER_COLOUR, "cornerradius": 4},
        text=labels, textposition="outside", textfont={"color": TEXT_COLOUR},
        customdata=ordered[["weight", "rating", "reason"]].to_numpy(),
        hovertemplate=("%{y}: weight %{customdata[0]:.2f} x rating %{customdata[1]} = %{x:.2f}"
                       "<br>%{customdata[2]}<extra></extra>"),
    ))
    figure.update_layout(
        title={"text": f"What makes up Current Ability {ca:.1f}: weight x rating, summed", "x": 0.0,
               "font": {"size": 14}},
        paper_bgcolor=SURFACE_COLOUR, plot_bgcolor=SURFACE_COLOUR,
        font={"color": TEXT_COLOUR, "family": "Segoe UI, Arial, sans-serif"},
        xaxis={"title": "contribution to Current Ability", "gridcolor": GRID_COLOUR, "zeroline": False,
               "range": [0, max(float(contributions["contribution"].max()) * 1.45, 1.0)]},
        yaxis={"title": None, "automargin": True, "ticksuffix": "  "},  # a little space between label and bar
        bargap=0.35, height=70 + 48 * len(contributions), margin={"l": 120, "r": 10, "t": 50, "b": 40},
        showlegend=False,
    )
    return figure
