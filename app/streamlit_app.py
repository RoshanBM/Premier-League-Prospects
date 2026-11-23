"""Streamlit app: shortlist, player card, trajectory, similar players and the method, for 2025/26 U21 prospects.

Run from the repository root, after python run_pipeline.py:
    streamlit run app/streamlit_app.py

The app only reads files in data/processed/ (see src/app_data.py). If any are
missing it shows one message saying to run the pipeline, and nothing else.
Data loading is cached; it reloads by itself when the pipeline rewrites the files.

Pages (tabs):
1. Shortlist: filter and sort the 2025/26 players; click a row to open the card.
2. Player card: attributes (mplsoccer pizza and table), Current Ability with
   each attribute's contribution, Potential Ability, and the scouting report.
3. Trajectory: the Plotly chart from src/figures.py.
4. Similar players: current and historical comparisons.
5. About the method: what CA and PA mean, the backtest verdict, limitations, sources.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# Streamlit runs this file with app/ on the path; the src package lives one level up.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.ability import PA_STATUS_EXPERIMENTAL  # noqa: E402
from src.app_charts import MIN_PIZZA_ATTRIBUTES, ca_contribution_figure, close_figure, pizza_figure  # noqa: E402
from src.app_data import (  # noqa: E402
    METHOD_DAY2,
    METHOD_MODEL,
    METHOD_PERSISTENCE,
    PLAYER_QUERY_KEY,
    POOL_ALL,
    POOL_IDENTIFIED,
    REQUIRED_FILES,
    SHORTLIST_COLUMNS,
    SORT_OPTIONS,
    SORT_PA,
    AppData,
    MissingDataError,
    ShortlistFilters,
    attribute_rows,
    backtest_row,
    backtest_table,
    ca_contributions,
    current_matches,
    data_dir,
    filter_players,
    historical_matches,
    load_app_data,
    missing_files,
    player_from_query,
    player_label,
    shortlist_table,
)
from src.calibration import load_calibration  # noqa: E402
from src.figures import GREYED_COLOUR, REFERENCE_COLOUR, TEXT_COLOUR, trajectory_figure  # noqa: E402
from src.report_text import (  # noqa: E402
    CUTOFF_TEXT,
    POSITION_WORDS,
    SEASON_TEXT,
    count_of,
    facts_for,
    ordinal,
    scouting_report,
)
from src.trajectory import TrajectoryModel, load_model  # noqa: E402

APP_TITLE = "Premier League U21 Prospects"
PAGE_SHORTLIST = "Shortlist"
PAGE_CARD = "Player card"
PAGE_TRAJECTORY = "Trajectory"
PAGE_SIMILAR = "Similar players"
PAGE_ABOUT = "About the method"
PAGES = (PAGE_SHORTLIST, PAGE_CARD, PAGE_TRAJECTORY, PAGE_SIMILAR, PAGE_ABOUT)
MISSING_DATA_MESSAGE = ("The processed data is not there yet. From the repository root, run "
                        "`python run_pipeline.py`, then reload this page.")
ATTRIBUTE_CAPTION = (f"Percentile-based ratings against PL players at the same position "
                     f"({config.MIN_MINUTES_FOR_PER90}+ minutes, {SEASON_TEXT})")
EXPERIMENTAL_NOTE = ("Experimental: in a backtest on past seasons, this projection did not beat the simple "
                     "\"minutes so far\" figure on average error. Read it next to the minutes share so far.")
LOW_EVIDENCE_NOTE = (f"Low evidence: fewer than {config.PA_LOW_EVIDENCE_MINUTES} Premier League minutes so far, so "
                     "the projection mostly reflects the average curve for the position.")
TILE_CSS = f"""
<style>
.tile {{border: 1px solid #e4e3df; border-radius: 8px; padding: 0.6rem 0.9rem; margin-bottom: 0.6rem;}}
.tile-label {{color: {REFERENCE_COLOUR}; font-size: 0.85rem;}}
.tile-value {{font-size: 2.1rem; font-weight: 600; line-height: 1.2; font-variant-numeric: tabular-nums;}}
.tile-note {{color: {REFERENCE_COLOUR}; font-size: 0.85rem;}}
</style>
"""


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------

def files_stamp(directory: Path) -> tuple[float, ...]:
    """Modification times of the required files, so the cache reloads after a pipeline run."""
    return tuple((directory / name).stat().st_mtime for name in REQUIRED_FILES.values())


@st.cache_resource(show_spinner="Loading the processed data ...")
def cached_data(directory: str, stamp: tuple[float, ...]) -> AppData:
    """Every table, loaded once per set of files. Read-only: pages never modify these frames."""
    return load_app_data(Path(directory))


@st.cache_resource(show_spinner=False)
def cached_model(directory: str, stamp: tuple[float, ...]) -> TrajectoryModel:
    """The fitted trajectory model."""
    return load_model(Path(directory) / REQUIRED_FILES["model"])


@st.cache_data(show_spinner="Simulating the projection ...")
def cached_trajectory(code: int, directory: str, stamp: tuple[float, ...]) -> go.Figure:
    """The trajectory chart for one player (src/figures.py), with the same interval as the card."""
    data = cached_data(directory, stamp)
    factor = 1.0
    if config.CARD_INTERVAL == "calibrated":
        factor = load_calibration(Path(directory) / config.CALIBRATION_PATH.name).factor
    figure = trajectory_figure(cached_model(directory, stamp), data.panel, data.pa, code, factor)
    figure.update_layout(height=620)
    return figure


# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------

def stat_tile(label: str, value: str, note: str = "", greyed: bool = False) -> None:
    """A headline number with a label and a note; greyed for a low-evidence value."""
    colour = GREYED_COLOUR if greyed else TEXT_COLOUR
    st.markdown(
        f'<div class="tile"><div class="tile-label">{html.escape(label)}</div>'
        f'<div class="tile-value" style="color:{colour}">{html.escape(value)}</div>'
        f'<div class="tile-note">{html.escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def position_word(code: str, plural: bool = False) -> str:
    """"DEF" -> "defender" (or "defenders")."""
    return POSITION_WORDS.get(code, (code, code))[1 if plural else 0]


def player_header(player: pd.Series) -> None:
    """Name, then club, position, age and minutes."""
    st.subheader(str(player["name"]))
    club = player["club"] if isinstance(player["club"], str) and player["club"] else "no club listed"
    age = f"aged {int(player['age_on_cutoff'])} on {CUTOFF_TEXT}" if pd.notna(player["age_on_cutoff"]) else "age unknown"
    st.markdown(f"{club} · {position_word(player['position']).capitalize()} · {age} · "
                f"{count_of(int(player['minutes']), 'Premier League minute')} in {SEASON_TEXT}")


def percent(share: float) -> str:
    """0.316 -> "32%"."""
    return f"{share:.0%}"


# ---------------------------------------------------------------------------
# Sidebar: the player picker
# ---------------------------------------------------------------------------

def default_player(players: pd.DataFrame) -> int:
    """The player shown first: the top of the default shortlist (U21s by Current Ability)."""
    return int(filter_players(players, ShortlistFilters())["fpl_code"].iloc[0])


def sidebar(players: pd.DataFrame) -> pd.Series:
    """The player picker and a short glossary. Returns the selected player's row."""
    # U21 cohort first, then everyone else, each by name. Typing in the box searches it.
    ordered = players.sort_values(["in_cohort", "name"], ascending=[False, True])
    labels = {int(row.fpl_code): player_label(pd.Series(row._asdict())) for row in ordered.itertuples(index=False)}
    # On the first run, the player comes from the URL (?player=<FPL code>) if it names a known player,
    # otherwise the top of the default shortlist. After that, the picker or a shortlist click decides.
    if PLAYER_QUERY_KEY not in st.session_state:
        from_url = player_from_query(st.query_params.get(PLAYER_QUERY_KEY), labels)
        st.session_state[PLAYER_QUERY_KEY] = from_url if from_url is not None else default_player(players)
    with st.sidebar:
        st.title(APP_TITLE)
        st.caption(f"Premier League season {SEASON_TEXT}. Under 21 means under 21 on {CUTOFF_TEXT}.")
        st.selectbox("Player (type to search)", options=list(labels), format_func=labels.get, key=PLAYER_QUERY_KEY,
                     help="U21 cohort first, then every other player with Premier League minutes this season. "
                          "Clicking a row in the shortlist also selects a player. The page address keeps the "
                          "page and the player's FPL code, so it can be bookmarked or shared.")
        st.markdown("**Current Ability (CA)**: how productive a player was per 90 minutes this season, against "
                    "Premier League players at the same position, on a 1 to 20 scale.")
        st.markdown("**Potential Ability (PA)**: a projection of how much Premier League football a U21 is likely "
                    "to play at ages 24 to 26, shown as a rank (1 to 20) among this season's U21 prospects. "
                    "Experimental.")
        st.caption("Data: Fantasy Premier League (via the vaastav archive) and the Transfermarkt dataset by "
                   "dcaribou. Non-commercial portfolio project.")
    selected = st.session_state.get(PLAYER_QUERY_KEY)
    if selected not in labels:  # an empty or out-of-date value: fall back to the default player
        selected = default_player(players)
    st.query_params[PLAYER_QUERY_KEY] = str(selected)  # the URL names the player by FPL code
    return players[players["fpl_code"] == selected].iloc[0]


# ---------------------------------------------------------------------------
# Page 1: shortlist
# ---------------------------------------------------------------------------

def select_from_shortlist() -> None:
    """Callback: a clicked shortlist row becomes the selected player, and the card opens."""
    rows = st.session_state["shortlist_table"]["selection"]["rows"]
    codes = st.session_state.get("shortlist_codes", [])
    if rows and rows[0] < len(codes):
        st.session_state[PLAYER_QUERY_KEY] = codes[rows[0]]
        st.session_state["page"] = PAGE_CARD


def shortlist_filters(players: pd.DataFrame) -> ShortlistFilters:
    """The filter widgets, laid out in three rows."""
    ages = players["age_on_cutoff"].dropna()
    low_age, high_age = int(ages.min()), int(ages.max())
    first = st.columns([1.4, 1, 1.6], gap="large")
    age_range = first[0].slider(f"Age on {CUTOFF_TEXT}", low_age, high_age, (low_age, high_age))
    min_minutes = first[1].number_input(f"Minimum minutes in {SEASON_TEXT}", min_value=0,
                                        max_value=int(players["minutes"].max()), value=0, step=90)
    ca_range = first[2].slider("Current Ability range", float(config.ATTRIBUTE_MIN), float(config.ATTRIBUTE_MAX),
                               (float(config.ATTRIBUTE_MIN), float(config.ATTRIBUTE_MAX)), step=0.5, format="%.1f",
                               help=f"The full range also keeps players with no CA (under "
                                    f"{config.MIN_MINUTES_FOR_PER90} minutes); a narrower range drops them.")
    second = st.columns([2.2, 1.8], gap="large")
    positions = second[0].pills("Position", list(config.FPL_POSITIONS.values()), selection_mode="multi",
                                default=list(config.FPL_POSITIONS.values()),
                                format_func=lambda p: position_word(p, plural=True).capitalize())
    sort_by = second[1].segmented_control("Sort by", SORT_OPTIONS, default=SORT_OPTIONS[0]) or SORT_OPTIONS[0]
    third = st.columns([1, 1.3, 3])
    u21_only = third[0].checkbox("U21 only", value=True,
                                 help=f"The 2025/26 U21 cohort: under {config.U21_AGE_LIMIT} on {CUTOFF_TEXT}, "
                                      f"with Premier League minutes this season.")
    hide_low = third[1].checkbox("Hide low-evidence PA",
                                 help=f"Hide players with fewer than {config.PA_LOW_EVIDENCE_MINUTES} career "
                                      f"Premier League minutes, whose PA mostly reflects the average curve.")
    return ShortlistFilters(ages=age_range, positions=tuple(positions), min_minutes=int(min_minutes),
                            ca_range=ca_range, u21_only=u21_only, hide_low_evidence=hide_low, sort_by=sort_by)


def page_shortlist(data: AppData) -> None:
    """Filters, the sortable table and a note on PA."""
    st.markdown("Current Ability (CA) rates this season's output per 90 minutes; Potential Ability (PA) ranks the "
                "projected minutes at ages 24 to 26 among this season's U21 prospects. Both run from 1 to 20.")
    filters = shortlist_filters(data.players)
    if filters.sort_by == SORT_PA:
        st.info("Potential Ability is experimental: in the backtest it did not beat the simple \"minutes so far\" "
                "figure on average error. Compare it with the minutes share so far.")
    chosen = filter_players(data.players, filters)
    st.session_state["shortlist_codes"] = [int(code) for code in chosen["fpl_code"]]
    st.caption(f"{len(chosen)} players. Click a row to open that player's card.")
    st.dataframe(
        shortlist_table(chosen), hide_index=True, width="stretch", key="shortlist_table",
        height=min(36 + 35 * max(len(chosen), 1), 720),
        on_select=select_from_shortlist, selection_mode="single-row",
        column_config={
            SHORTLIST_COLUMNS["age_on_cutoff"]: st.column_config.NumberColumn(format="%d", help=f"Age on {CUTOFF_TEXT}"),
            SHORTLIST_COLUMNS["minutes"]: st.column_config.NumberColumn(format="%d"),
            SHORTLIST_COLUMNS["ca"]: st.column_config.NumberColumn(
                format="%.1f", help=f"Current Ability, 1 to 20; blank under {config.MIN_MINUTES_FOR_PER90} minutes"),
            SHORTLIST_COLUMNS["pa"]: st.column_config.NumberColumn(
                format="%d", help="Potential Ability: rank among 2025/26 U21 prospects, 1 to 20 (experimental)"),
            SHORTLIST_COLUMNS["persistence_pct"]: st.column_config.NumberColumn(
                format="%.0f%%", help="Average share of league minutes per season so far (2016-17 to 2025-26)"),
            SHORTLIST_COLUMNS["pa_status_label"]: st.column_config.TextColumn(
                help="experimental; low evidence = fewer than 450 career PL minutes"),
        },
    )


# ---------------------------------------------------------------------------
# Page 2: player card
# ---------------------------------------------------------------------------

def card_attributes(player: pd.Series, data: AppData) -> None:
    """The pizza chart and the table of ratings with the per-90 numbers behind them."""
    st.markdown("#### Attributes")
    rows = attribute_rows(player, data.player_season)
    if rows.empty:
        st.info(f"No attributes: fewer than {config.MIN_MINUTES_FOR_PER90} Premier League minutes in {SEASON_TEXT}, "
                "so per-90 numbers would be too noisy.")
        return
    st.caption(ATTRIBUTE_CAPTION)
    if len(rows) >= MIN_PIZZA_ATTRIBUTES:
        figure = pizza_figure(dict(zip(rows["Attribute"], rows["Rating (1 to 20)"])))
        st.pyplot(figure, width="stretch")
        close_figure(figure)
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Passing, progression, dribbling, aerial ability and pressing are not in any open data source for "
               "recent Premier League seasons, so there are no attributes for them.")


def card_current_ability(player: pd.Series) -> None:
    """CA as a number, its rank in position, and what each attribute contributes."""
    st.markdown("#### Current Ability")
    if pd.isna(player["ca"]):
        st.info(f"No Current Ability: it needs attributes, which need {config.MIN_MINUTES_FOR_PER90}+ minutes.")
        return
    rank = (f"{ordinal(int(player['ca_rank_in_position']))} of {int(player['ca_pool'])} "
            f"{position_word(player['position'], plural=True)} with {config.MIN_MINUTES_FOR_PER90}+ minutes")
    stat_tile("Current Ability (1 to 20)", f"{player['ca']:.1f}", rank)
    st.plotly_chart(ca_contribution_figure(ca_contributions(player), float(player["ca"])), theme=None,
                    config={"displayModeBar": False})
    st.caption("Each bar is the attribute's weight times its rating; the bars add up to CA. The weights are chosen "
               "by hand for each position (hover for the reason).")


def card_potential_ability(player: pd.Series) -> None:
    """PA rank (greyed when low evidence), status, range and minutes share so far."""
    st.markdown("#### Potential Ability")
    if not player["in_cohort"]:
        st.info(f"Potential Ability is only projected for the {SEASON_TEXT} U21 cohort (under "
                f"{config.U21_AGE_LIMIT} on {CUTOFF_TEXT}).")
        return
    so_far = (f"Minutes share so far: {percent(player['persistence_share'])} of league minutes per season "
              f"({count_of(int(player['pl_minutes_to_date']), 'career PL minute')})")
    if player["pa_status"] != PA_STATUS_EXPERIMENTAL:
        stat_tile("Potential Ability", "Not enough data to project", so_far)
        st.caption(f"Every Premier League season so far came before age {config.MODEL_AGE_RANGE[0]}, the youngest "
                   "age the model is fitted on, so it could only give the average curve.")
        return
    low = bool(player["pa_low_evidence"])
    stat_tile(f"Potential Ability: {player['pa_label']} (1 to 20)", str(int(player["pa"])), so_far, greyed=low)
    badges = st.columns([1, 1, 2])
    with badges[0]:
        st.badge("Experimental", icon=":material/science:", color="orange")
    if low:
        with badges[1]:
            st.badge("Low evidence", icon=":material/visibility_off:", color="gray")
    st.markdown(f"**{player['pa_range_text']}** (80% range).")
    st.caption(EXPERIMENTAL_NOTE)
    if low:
        st.caption(LOW_EVIDENCE_NOTE)


def page_card(player: pd.Series, data: AppData) -> None:
    """Header, attributes, CA, PA and the scouting report."""
    player_header(player)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        card_attributes(player, data)
    with right:
        card_current_ability(player)
        card_potential_ability(player)
    st.markdown("#### Scouting report")
    facts = facts_for(int(player["fpl_code"]), data.player_season, data.ca, data.pa, data.similar)
    with st.container(border=True):
        st.write(scouting_report(facts))
    st.caption("Written by fixed rules from the numbers above, not by a language model.")


# ---------------------------------------------------------------------------
# Page 3: trajectory
# ---------------------------------------------------------------------------

def page_trajectory(player: pd.Series, directory: Path, stamp: tuple[float, ...]) -> None:
    """The trajectory chart and how to read it."""
    player_header(player)
    if not player["in_cohort"]:
        st.info(f"Trajectories are projected only for the {SEASON_TEXT} U21 cohort.")
        return
    if player["pa_status"] == PA_STATUS_EXPERIMENTAL:
        st.markdown(f"**{player['pa_range_text']}** (80% range). Projection status: experimental"
                    + ("; low evidence, so the projection is drawn in grey." if player["pa_low_evidence"] else "."))
    else:
        st.markdown(f"**Not enough data to project**: every season so far is before age "
                    f"{config.MODEL_AGE_RANGE[0]}. The chart shows the seasons so far and the average curve only.")
    st.plotly_chart(cached_trajectory(int(player["fpl_code"]), str(directory), stamp), theme=None)
    projection = ("The shaded band is the 80% range for each future season, and the diamond is the projected peak "
                  "(the mean of ages 24 to 26) with its 80% range. "
                  if player["pa_status"] == PA_STATUS_EXPERIMENTAL else "")
    st.caption("Minutes share = Premier League minutes / 3,420 (38 matches of 90 minutes). Squares are seasons out "
               "of the PL, counted as 0; hollow markers are seasons outside the model's ages (17 to 30). "
               f"{projection}The dashed line is the average curve for the position.")


# ---------------------------------------------------------------------------
# Page 4: similar players
# ---------------------------------------------------------------------------

def page_similar(player: pd.Series, data: AppData) -> None:
    """Five current and five historical comparisons."""
    player_header(player)
    code = int(player["fpl_code"])
    current = current_matches(data.similar, code)
    if current.empty:
        st.info(f"Similar players are worked out only for U21s with at least {config.MIN_MINUTES_FOR_PER90} "
                f"Premier League minutes in {SEASON_TEXT}, because they compare per-90 numbers.")
        return
    similarity = st.column_config.NumberColumn(format="%.2f", help="Cosine similarity: 1 = same profile, "
                                                                     "0 = unrelated, -1 = opposite")
    st.markdown(f"#### Closest current players ({SEASON_TEXT})")
    st.dataframe(current, hide_index=True, width="stretch", column_config={"Similarity": similarity})
    st.caption("Same position, compared on goals, expected goals (xG), assists, expected assists (xA), tackles, "
               "clearances, blocks and interceptions, and recoveries per 90 (goalkeepers: saves, goals conceded and "
               "expected goals conceded), each standardised within the position.")
    st.markdown("#### Closest earlier players at the same age, and what they went on to play")
    st.dataframe(historical_matches(data.similar, code), hide_index=True, width="stretch",
                 column_config={"Similarity": similarity})
    st.caption("Historical matches use a smaller set of stats, the only ones every season since 2016-17 has: "
               "minutes share and goals, assists, influence, creativity and threat per 90 (the last three are FPL "
               "scores with an unpublished formula), standardised within position and season. Seasons 2016-17 to "
               "2024-25, same position, within a year of age, each player's best-matching season. \"At 24 to 26\" is "
               "the mean share of league minutes in seasons at those ages, with seasons out of the league counted "
               "as 0. Low similarity scores are loose matches.")


# ---------------------------------------------------------------------------
# Page 5: about the method
# ---------------------------------------------------------------------------

def backtest_verdict(backtest: pd.DataFrame) -> str:
    """The backtest verdict in one paragraph, with numbers read from backtest_results.csv."""
    model_all = backtest_row(backtest, POOL_ALL, METHOD_MODEL)
    persistence_all = backtest_row(backtest, POOL_ALL, METHOD_PERSISTENCE)
    day2_all = backtest_row(backtest, POOL_ALL, METHOD_DAY2)
    model_peak = backtest_row(backtest, POOL_IDENTIFIED, METHOD_MODEL)
    persistence_peak = backtest_row(backtest, POOL_IDENTIFIED, METHOD_PERSISTENCE)
    cut = 1 - model_all["bias"] / day2_all["bias"]
    return (
        f"We went back to four earlier seasons (2017-18 to 2020-21), used only what was known then, and predicted "
        f"how much that season's under-21s would play at ages 23 to 25 ({int(model_all['players'])} forecasts). "
        f"**The model does not beat the simple \"minutes so far\" figure on average error**: "
        f"{model_all['mae']:.3f} against {persistence_all['mae']:.3f} over all four seasons, and "
        f"{model_peak['mae']:.3f} against {persistence_peak['mae']:.3f} in 2020-21, the only test season with "
        f"enough data at ages 23 to 25 to check the peak. Errors are in shares of league minutes (0.10 is 10 "
        f"percentage points). Fitting the model only on players first seen young cut its average over-prediction "
        f"from {day2_all['bias']:+.3f} to {model_all['bias']:+.3f}, by {cut:.0%}. In 2020-21 it ranked players "
        f"slightly better than minutes so far (Spearman {model_peak['spearman']:.3f} against "
        f"{persistence_peak['spearman']:.3f}), but that is one season and {int(model_peak['players'])} players. "
        f"That is why Potential Ability is labelled experimental and shown next to the minutes share so far."
    )


def page_about(data: AppData) -> None:
    """A short plain-English summary of the method, the backtest, the limits and the sources."""
    st.markdown("### What Current Ability and Potential Ability mean")
    st.markdown(
        f"- **Attributes (1 to 20).** Each of this season's per-90 numbers (goals, expected goals, assists, "
        f"tackles and so on) becomes a percentile against every Premier League player at the same position with "
        f"at least {config.MIN_MINUTES_FOR_PER90} minutes, not only under-21s. The percentile maps to 1 to 20, so "
        f"a 15 means better than about 70% of the league at that position.\n"
        f"- **Current Ability (CA)** is a weighted average of the attributes, with weights chosen by hand for each "
        f"position (for example, defending counts most for defenders). It measures output per 90 minutes in "
        f"{SEASON_TEXT}, not how much a player played.\n"
        f"- **Potential Ability (PA)** comes from a statistical age curve of Premier League minutes. It projects "
        f"how much of the league's minutes a U21 is likely to play at ages 24 to 26, then ranks that projection "
        f"against every other {SEASON_TEXT} U21 prospect on a 1 to 20 scale. It says nothing about per-90 output, "
        f"so a player can have a high CA and a low PA, or the reverse.\n"
        f"- **Ranges, not single numbers.** Each projection comes with an 80% range in words, for example "
        f"\"Range: from not playing regularly in the PL to about 45% of league minutes at 24 to 26\". For most "
        f"prospects the range starts at zero: partly because dropping out of the league is a real outcome, and "
        f"partly because the model treats minutes as a smooth quantity although half the seasons it learns from "
        f"are zeros, so its lower end often falls below zero and is cut to 0%.\n"
        f"- **Low evidence.** A PA based on fewer than {config.PA_LOW_EVIDENCE_MINUTES} career Premier League "
        f"minutes is greyed out: with so little to go on, it mostly reflects the average curve. Players whose "
        f"seasons so far all came before age {config.MODEL_AGE_RANGE[0]} get no PA."
    )
    st.markdown("### Does it work? The backtest")
    st.markdown(backtest_verdict(data.backtest))
    st.dataframe(backtest_table(data.backtest), hide_index=True, width="stretch",
                 column_config={name: st.column_config.NumberColumn(format="%.3f")
                                for name in ("Average error (MAE)", "Ranking (Spearman)", "Bias")})
    st.caption("MAE: mean absolute error (lower is better). Spearman: how well the order of players matches what "
               "happened (1 is perfect). Bias: mean prediction minus mean outcome. Source: "
               "data/processed/backtest_results.csv.")
    st.markdown("### Main limitations")
    st.markdown(
        "- Only players who already have Premier League minutes are covered; there is no academy or youth data.\n"
        "- There is no Championship or other EFL data, so loans in England look like time out of the league.\n"
        "- The data starts in 2016-17. The model is therefore fitted only on players first seen at 20 or under, "
        "and some players who debuted young before 2016-17 are wrongly left out.\n"
        "- The peak age was tested at only one backtest season (2020-21).\n"
        "- The model treats minutes share as a smooth normal quantity, although half of the seasons it learns "
        "from are zeros; impossible values are cut to 0% to 100% for display.\n"
        "- PA is relative to this season's prospects, and minutes depend on club, manager and role as well as "
        "ability.\n"
        "- The attribute set is limited: no passing, progression, dribbling, aerial or pressing data.\n"
        "- Defensive stats exist only from 2025/26, so trajectories use minutes, not output."
    )
    st.markdown("### Data sources and credits")
    st.markdown(
        "- **Fantasy Premier League**, via the vaastav/Fantasy-Premier-League archive on GitHub: minutes, goals, "
        "expected goals and assists, defensive actions and positions, 2016-17 to 2025-26. The data belongs to the "
        "Premier League, whose terms allow personal, non-commercial use.\n"
        "- **Transfermarkt**, via the dcaribou/transfermarkt-datasets dataset: birth dates for older seasons and "
        "checks on minutes. The repository is CC0, but the data was collected from Transfermarkt, so it is used "
        "for non-commercial purposes only. Credit to Transfermarkt and the dataset's author.\n"
        "- No raw data is redistributed with this project. StatsBomb event data is not used yet.\n"
        "- This is a non-commercial portfolio project."
    )


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

def main() -> None:
    """Check the data, then draw the sidebar and the open page."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    directory = data_dir()
    if missing_files(directory):
        st.warning(MISSING_DATA_MESSAGE)
        return
    stamp = files_stamp(directory)
    try:
        data = cached_data(str(directory), stamp)
    except (MissingDataError, ValueError):
        st.warning(MISSING_DATA_MESSAGE)
        return
    st.markdown(TILE_CSS, unsafe_allow_html=True)
    player = sidebar(data.players)
    tabs = st.tabs(PAGES, key="page", on_change="rerun", bind="query-params")
    pages = dict(zip(PAGES, tabs))
    # Only the open page is drawn, so the trajectory simulation runs only when it is looked at.
    with pages[PAGE_SHORTLIST]:
        if pages[PAGE_SHORTLIST].open:
            page_shortlist(data)
    with pages[PAGE_CARD]:
        if pages[PAGE_CARD].open:
            page_card(player, data)
    with pages[PAGE_TRAJECTORY]:
        if pages[PAGE_TRAJECTORY].open:
            page_trajectory(player, directory, stamp)
    with pages[PAGE_SIMILAR]:
        if pages[PAGE_SIMILAR].open:
            page_similar(player, data)
    with pages[PAGE_ABOUT]:
        if pages[PAGE_ABOUT].open:
            page_about(data)


main()
