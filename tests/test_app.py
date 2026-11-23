"""Tests for the Streamlit app (app/streamlit_app.py) and its data helpers (src/app_data.py, src/app_charts.py).

- Smoke tests run the real app script with streamlit.testing.v1.AppTest and
  check that no page raises an exception, for a regular starter, a
  low-evidence player, an under-17 debutant and a player outside the U21 cohort.
- Missing files: the app shows only the "run the pipeline" message.
- Data loading: the processed files have the columns the app uses; missing
  files and missing columns are reported clearly.
- The shortlist filters and the page helpers, on toy tables.

Tests that need the processed data skip themselves if it has not been built.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src import config
from src.ability import PA_STATUS_EXPERIMENTAL, PA_STATUS_TOO_YOUNG
from src.app_data import (
    DATA_DIR_VARIABLE,
    EXPECTED_COLUMNS,
    METHOD_NAMES,
    REQUIRED_FILES,
    SORT_MINUTES,
    SORT_PA,
    STATUS_EXPERIMENTAL,
    STATUS_LOW_EVIDENCE,
    STATUS_NOT_U21,
    STATUS_TOO_YOUNG,
    MissingDataError,
    ShortlistFilters,
    attribute_rows,
    backtest_row,
    backtest_table,
    build_player_table,
    ca_contributions,
    data_dir,
    filter_players,
    historical_matches,
    load_app_data,
    missing_columns,
    missing_files,
    pa_status_label,
    peak_text,
    player_from_query,
    read_tables,
    shortlist_table,
    two_decimals,
)

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
PAGES = ("Shortlist", "Player card", "Trajectory", "Similar players", "About the method")
HAVE_DATA = not missing_files(config.PROCESSED_DIR)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="processed data not built (python run_pipeline.py)")
APP_TIMEOUT = 180


# ---------------------------------------------------------------------------
# Smoke tests of the real app
# ---------------------------------------------------------------------------

def run_app(player: int | None = None, page: str | None = None) -> AppTest:
    """Run the app script once, optionally with a player and a page chosen."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=APP_TIMEOUT)
    if player is not None:
        app.session_state["player"] = player
    if page is not None:
        app.session_state["page"] = page
    return app.run()


@pytest.fixture(scope="module")
def test_players() -> dict[str, int]:
    """One player of each kind the pages must handle, chosen from the data rather than by name."""
    if not HAVE_DATA:
        pytest.skip("processed data not built (python run_pipeline.py)")
    data = load_app_data(config.PROCESSED_DIR)
    players = data.players
    cohort = players[players["in_cohort"]]
    starter = cohort[cohort["pa_status"].eq(PA_STATUS_EXPERIMENTAL) & cohort["ca"].notna()
                     & ~cohort["pa_low_evidence"].astype(bool)].sort_values("minutes", ascending=False)
    low = cohort[cohort["pa_status"].eq(PA_STATUS_EXPERIMENTAL) & cohort["pa_low_evidence"].astype(bool)]
    too_young = cohort[cohort["pa_status"].eq(PA_STATUS_TOO_YOUNG)]
    senior = players[~players["in_cohort"] & players["ca"].notna()]
    return {"starter": int(starter["fpl_code"].iloc[0]), "low evidence": int(low["fpl_code"].iloc[0]),
            "under 17": int(too_young["fpl_code"].iloc[0]), "not U21": int(senior["fpl_code"].iloc[0])}


@needs_data
def test_app_runs_without_an_exception():
    app = run_app()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == list(PAGES)
    assert len(app.dataframe) == 1  # the shortlist


@needs_data
def test_default_shortlist_is_the_u21_cohort():
    app = run_app()
    table = app.dataframe[0].value
    assert len(table) == int(pd.read_parquet(config.PA_PROJECTIONS_PATH).shape[0])
    assert list(table.columns) == ["Name", "Club", "Position", "Age", "Minutes 2025/26", "Current Ability",
                                   "PA rank", "Minutes share so far", "PA status"]


@needs_data
@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("kind", ["starter", "low evidence", "under 17", "not U21"])
def test_every_page_loads_for_each_kind_of_player(test_players, kind, page):
    app = run_app(test_players[kind], page)
    assert not app.exception, [error.value for error in app.exception]


@needs_data
def test_under_17_card_says_not_enough_data(test_players):
    app = run_app(test_players["under 17"], "Player card")
    assert any("Not enough data to project" in block.value for block in app.markdown)


@needs_data
def test_low_evidence_card_is_flagged(test_players):
    app = run_app(test_players["low evidence"], "Player card")
    texts = " ".join(block.value for block in app.markdown) + " ".join(c.value for c in app.caption)
    assert "Low evidence" in texts


@needs_data
def test_about_page_quotes_the_backtest_file():
    app = run_app(page="About the method")
    backtest = pd.read_csv(config.BACKTEST_RESULTS_PATH)
    model = backtest_row(backtest, "pooled, all origins", "new model (first seen young)")
    persistence = backtest_row(backtest, "pooled, all origins", "persistence (mean share to date)")
    text = " ".join(block.value for block in app.markdown)
    assert f"{model['mae']:.3f} against {persistence['mae']:.3f}" in text
    assert "does not beat" in text


def test_missing_files_show_only_the_pipeline_message(tmp_path, monkeypatch):
    monkeypatch.setenv(DATA_DIR_VARIABLE, str(tmp_path))
    app = run_app()
    assert not app.exception
    assert len(app.warning) == 1
    assert "python run_pipeline.py" in app.warning[0].value
    assert not app.tabs and not app.dataframe and not app.sidebar.children


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def test_data_dir_follows_the_environment_variable(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_DIR_VARIABLE, raising=False)
    assert data_dir() == config.PROCESSED_DIR
    monkeypatch.setenv(DATA_DIR_VARIABLE, str(tmp_path))
    assert data_dir() == tmp_path


def test_every_required_file_is_reported_missing_in_an_empty_folder(tmp_path):
    assert [path.name for path in missing_files(tmp_path)] == list(REQUIRED_FILES.values())


def test_reading_an_empty_folder_raises_a_clear_error(tmp_path):
    with pytest.raises(MissingDataError) as error:
        read_tables(tmp_path)
    assert len(error.value.missing) == len(REQUIRED_FILES)


def test_missing_columns_are_listed_by_table():
    tables = {"ca": pd.DataFrame({"fpl_code": [1]}), "pa": pd.DataFrame(columns=list(EXPECTED_COLUMNS["pa"]))}
    gaps = missing_columns(tables)
    assert set(gaps) == {"ca"} and "ca" in gaps["ca"]


@needs_data
@pytest.mark.parametrize("name", sorted(EXPECTED_COLUMNS))
def test_processed_files_have_the_columns_the_app_uses(name):
    path = config.PROCESSED_DIR / REQUIRED_FILES[name]
    table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    assert not [column for column in EXPECTED_COLUMNS[name] if column not in table.columns]


@needs_data
def test_player_table_from_the_real_files():
    data = load_app_data(config.PROCESSED_DIR)
    players = data.players
    assert players["fpl_code"].is_unique
    assert (players["minutes"] > 0).all()
    assert int(players["in_cohort"].sum()) == len(data.pa)
    assert players.loc[players["ca"].notna(), "ca"].between(config.ATTRIBUTE_MIN, config.ATTRIBUTE_MAX).all()
    # Contributions add up to CA for every player with a CA.
    for _, player in players[players["ca"].notna()].head(50).iterrows():
        assert ca_contributions(player)["contribution"].sum() == pytest.approx(player["ca"])


@needs_data
def test_backtest_table_has_every_method_for_both_pools():
    table = backtest_table(pd.read_csv(config.BACKTEST_RESULTS_PATH))
    assert len(table) == 2 * len(METHOD_NAMES)
    assert set(table["Method"]) == set(METHOD_NAMES.values())


def test_a_missing_backtest_row_is_an_error():
    with pytest.raises(KeyError):
        backtest_row(pd.DataFrame(columns=["origin", "method", "subset"]), "pooled, all origins", "x")


# ---------------------------------------------------------------------------
# The shortlist and page helpers, on toy tables
# ---------------------------------------------------------------------------

def toy_players() -> pd.DataFrame:
    """Five players: two U21 starters, a low-evidence U21, an under-17 and a senior player with no CA."""
    attributes = {a: [10, 12, 8, np.nan, np.nan] for a in
                  ("finishing", "shot_threat", "chance_creation", "defending", "ball_recovery")}
    player_season = pd.DataFrame({
        "fpl_code": [1, 2, 3, 4, 5], "name": list("ABCDE"), "web_name": list("ABCDE"),
        "position": ["DEF", "MID", "MID", "MID", "FWD"], "club": ["X"] * 5, "age_on_cutoff": [20, 19, 18, 15, 30],
        "exact_age_on_cutoff": [20.5, 19.5, 18.5, 15.5, 30.5], "u21": [True, True, True, True, False],
        "minutes": [2000, 1500, 600, 150, 300], "enough_minutes": [True, True, True, False, False],
        "goalkeeping": [np.nan] * 5, **attributes,
    })
    ca = pd.DataFrame({"fpl_code": [1, 2, 3], "position": ["DEF", "MID", "MID"], "ca": [12.0, 14.0, 9.0],
                       "ca_rank_in_position": [1, 1, 2],
                       **{f"contrib_{a}": [1.0, 1.0, 1.0] for a in
                          ("finishing", "shot_threat", "chance_creation", "defending", "ball_recovery", "goalkeeping")}})
    pa = pd.DataFrame({
        "fpl_code": [1, 2, 3, 4], "pa": [15, 18, 9, pd.NA], "pa_pct": [0.7, 0.9, 0.4, np.nan],
        "pa_label": ["rank"] * 4,
        "pa_status": [PA_STATUS_EXPERIMENTAL] * 3 + [PA_STATUS_TOO_YOUNG],
        "pa_low_evidence": [False, False, True, True], "pa_display": ["normal", "normal", "greyed", None],
        "pa_range_text": ["r"] * 3 + [None], "persistence_share": [0.4, 0.3, 0.05, 0.04],
        "peak_median_display": [0.4, 0.5, 0.1, np.nan], "pl_minutes_to_date": [5000, 3000, 300, 150],
        "history_seasons": [3, 2, 1, 0], "card_interval": ["raw"] * 4,
    })
    pa["pa"] = pa["pa"].astype("Int64")
    return build_player_table(player_season, ca, pa)


def test_player_table_marks_the_cohort_and_status():
    players = toy_players()
    assert players["in_cohort"].tolist() == [True, True, True, True, False]
    assert players["pa_status_label"].tolist() == [STATUS_EXPERIMENTAL, STATUS_EXPERIMENTAL, STATUS_LOW_EVIDENCE,
                                                   STATUS_TOO_YOUNG, STATUS_NOT_U21]
    assert players["ca_pool"].tolist()[:3] == [1, 2, 2]


def test_status_label_for_a_missing_status_is_not_u21():
    frame = pd.DataFrame({"pa_status": [None], "pa_low_evidence": [None]})
    assert pa_status_label(frame).tolist() == [STATUS_NOT_U21]


def test_default_filters_keep_the_u21_cohort_sorted_by_ca_with_missing_last():
    chosen = filter_players(toy_players(), ShortlistFilters())
    assert chosen["fpl_code"].tolist() == [2, 1, 3, 4]


def test_u21_only_off_adds_senior_players():
    chosen = filter_players(toy_players(), ShortlistFilters(u21_only=False))
    assert 5 in chosen["fpl_code"].tolist()


def test_hide_low_evidence_drops_flagged_players_including_those_without_pa():
    chosen = filter_players(toy_players(), ShortlistFilters(hide_low_evidence=True))
    assert chosen["fpl_code"].tolist() == [2, 1]


def test_a_narrower_ca_range_drops_players_without_ca():
    chosen = filter_players(toy_players(), ShortlistFilters(ca_range=(10.0, 20.0)))
    assert chosen["fpl_code"].tolist() == [2, 1]


def test_age_position_and_minutes_filters():
    players = toy_players()
    assert filter_players(players, ShortlistFilters(ages=(18, 19)))["fpl_code"].tolist() == [2, 3]
    assert filter_players(players, ShortlistFilters(positions=("DEF",)))["fpl_code"].tolist() == [1]
    assert filter_players(players, ShortlistFilters(min_minutes=1000))["fpl_code"].tolist() == [2, 1]


def test_sorting_by_pa_and_by_minutes():
    players = toy_players()
    assert filter_players(players, ShortlistFilters(sort_by=SORT_PA))["fpl_code"].tolist() == [2, 1, 3, 4]
    assert filter_players(players, ShortlistFilters(sort_by=SORT_MINUTES))["fpl_code"].tolist() == [1, 2, 3, 4]


def test_shortlist_table_shows_persistence_as_a_percentage():
    table = shortlist_table(filter_players(toy_players(), ShortlistFilters()))
    assert table["Minutes share so far"].tolist() == pytest.approx([30.0, 40.0, 5.0, 4.0])


def test_attribute_rows_skip_players_without_attributes():
    players = toy_players()
    per90 = {c: 0.1 for c in ("goals_per90", "goals_minus_xg_per90", "xg_per90", "xa_per90", "assists_per90",
                              "defensive_actions_per90", "recoveries_per90", "saves_per90", "goals_prevented_per90")}
    player_season = players.assign(**per90)
    assert len(attribute_rows(players.iloc[0], player_season)) == 5
    assert attribute_rows(players.iloc[4], player_season).empty


@pytest.mark.parametrize("value, expected", [
    ("487838", 487838), (" 487838 ", 487838), (["1", "487838"], 487838),
    (None, None), ([], None), ("Lewis Hall (Newcastle, DEF, 20)", None), ("12.5", None), ("999", None),
])
def test_player_from_query_reads_an_fpl_code(value, expected):
    assert player_from_query(value, {487838, 1}) == expected


def test_two_decimals_has_no_negative_zero():
    assert two_decimals(-0.001) == "0.00"
    assert two_decimals(1.234) == "1.23"


def test_peak_text():
    assert peak_text(0.44, 3) == "44% of league minutes (3 seasons)"
    assert peak_text(0.94, 1) == "94% of league minutes (1 season)"
    assert peak_text(np.nan, 0) == "not reached by 2025/26"


def test_historical_matches_say_when_the_peak_is_not_reached():
    similar = pd.DataFrame({
        "fpl_code": [1, 1], "comparison": ["historical"] * 2, "rank": [2, 1], "match_name": ["B", "A"],
        "match_season": ["2023-24", "2020-21"], "match_club": [None, "Club"], "match_age": [20.3, 21.9],
        "match_minutes": [1746, 3029], "similarity": [0.64, 0.90], "match_next_season_share": [np.nan, 0.83],
        "match_peak_share": [np.nan, 0.44], "match_peak_seasons": [0, 3],
    })
    table = historical_matches(similar, 1)
    assert table["Player"].tolist() == ["A", "B"]
    assert table["At 24 to 26"].tolist() == ["44% of league minutes (3 seasons)", "not reached by 2025/26"]
    assert table["Next season"].tolist() == ["83% of league minutes", "not in the data"]


def test_pizza_needs_at_least_three_attributes():
    from src.app_charts import close_figure, pizza_figure

    with pytest.raises(ValueError):
        pizza_figure({"Goalkeeping": 7})
    figure = pizza_figure({"Finishing": 14, "Shot Threat": 9, "Chance Creation": 15})
    close_figure(figure)
