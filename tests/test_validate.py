"""Tests for src/validate.py: the time split never lets data after the origin into training,
the restricted training data only has players first seen young, and the calibration is rolling.

The checks on the real panel skip themselves if it has not been built yet.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.validate import (
    calibrate,
    check_first_seen_young,
    check_no_leakage,
    peak_identification,
    repeated_players,
    spearman,
    target_rows,
    training_panel,
)

YOUNG = config.FIRST_SEEN_YOUNG_MAX_AGE


def toy_panel() -> pd.DataFrame:
    """Two players over four seasons; player 2 debuts in 2019-20."""
    seasons = ["2016-17", "2017-18", "2018-19", "2019-20"]
    rows = []
    for code, debut in ((1, "2016-17"), (2, "2019-20")):
        for age, season in enumerate(seasons, start=19):
            rows.append({"fpl_code": code, "season": season, "season_year": config.season_start_year(season),
                         "debut_season": debut, "age_completed": age, "minutes_share": 0.5, "filled": False})
    return pd.DataFrame(rows)


class TestLeakage:
    def test_training_stops_at_the_origin(self):
        training = training_panel(toy_panel(), "2017-18")
        assert training["season_year"].max() == 2017
        assert set(training["season"]) == {"2016-17", "2017-18"}

    def test_players_who_debut_later_are_left_out(self):
        training = training_panel(toy_panel(), "2018-19")
        assert set(training["fpl_code"]) == {1}

    def test_a_later_season_raises(self):
        with pytest.raises(AssertionError):
            check_no_leakage(toy_panel(), "2017-18")

    def test_a_later_debutant_raises_even_without_later_seasons(self):
        panel = toy_panel()
        early = panel[panel["season_year"] <= 2017]
        with pytest.raises(AssertionError):
            check_no_leakage(early, "2017-18")

    def test_clean_training_data_passes(self):
        check_no_leakage(training_panel(toy_panel(), "2019-20"), "2019-20")


def test_targets_are_after_the_origin_and_inside_the_age_window():
    targets = target_rows(toy_panel(), pd.Series([1]), "2016-17", ages=(20, 21))
    assert targets["season"].tolist() == ["2017-18", "2018-19"]


def test_spearman_is_one_for_any_increasing_relationship():
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a**3) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)


def toy_panel_with_first_seen() -> pd.DataFrame:
    """toy_panel plus first_seen_age, and a third player first seen at 24 in 2016-17.

    Player 1 is first seen at 19 (2016-17), player 2 at 22 (2019-20) and
    player 3 at 24 (2016-17).
    """
    panel = toy_panel()
    panel["first_seen_age"] = panel["fpl_code"].map({1: 19, 2: 22})
    older = panel[panel["fpl_code"] == 1].assign(fpl_code=3, first_seen_age=24, age_completed=lambda d: d["age_completed"] + 5)
    return pd.concat([panel, older], ignore_index=True)


class TestRestrictedTraining:
    def test_only_players_first_seen_young_are_kept(self):
        training = training_panel(toy_panel_with_first_seen(), "2019-20", first_seen_max_age=YOUNG)
        assert set(training["fpl_code"]) == {1}
        assert (training["first_seen_age"] <= YOUNG).all()

    def test_without_the_restriction_every_player_who_had_played_stays(self):
        assert set(training_panel(toy_panel_with_first_seen(), "2019-20")["fpl_code"]) == {1, 2, 3}

    @pytest.mark.parametrize("origin", ["2016-17", "2017-18", "2018-19"])
    def test_restricted_training_never_goes_past_the_origin(self, origin):
        training = training_panel(toy_panel_with_first_seen(), origin, first_seen_max_age=YOUNG)
        assert training["season_year"].max() <= config.season_start_year(origin)
        assert (training["debut_season"].map(config.season_start_year) <= config.season_start_year(origin)).all()

    def test_a_player_first_seen_older_raises(self):
        with pytest.raises(AssertionError):
            check_first_seen_young(toy_panel_with_first_seen(), YOUNG)

    def test_a_missing_first_seen_age_counts_as_not_young(self):
        panel = toy_panel_with_first_seen()
        panel.loc[panel["fpl_code"] == 1, "first_seen_age"] = np.nan
        assert training_panel(panel, "2019-20", first_seen_max_age=YOUNG).empty


def test_peak_identification_counts_seasons_at_the_target_ages():
    rows = pd.DataFrame({"fpl_code": [1, 1, 2, 3, 4], "age_completed": [22, 23, 24, 25, 26]})
    result = peak_identification(rows, ages=(23, 25), minimum=3)
    assert result["seasons at 23 to 25"] == 3 and result["players at 23 to 25"] == 3
    assert [result[f"seasons at {a}"] for a in (23, 24, 25)] == [1, 1, 1]
    assert result["peak identified"]
    assert not peak_identification(rows, ages=(23, 25), minimum=4)["peak identified"]


def toy_forecasts(covered_share: float) -> pd.DataFrame:
    """Forecasts at two origins: interval 0.3 to 0.5 around 0.4, a given share of outcomes inside it."""
    rows = []
    for origin in ("2017-18", "2018-19"):
        for i in range(10):
            inside = i < round(covered_share * 10)
            rows.append({"origin": origin, "fpl_code": i, "sim_median": 0.4, "sim_lower": 0.3, "sim_upper": 0.5,
                         "actual": 0.45 if inside else 0.4 + 0.1 * (1 + i), "peak_identified": origin == "2018-19"})
    frame = pd.DataFrame(rows)
    frame["covered"] = (frame["sim_lower"] <= frame["actual"]) & (frame["actual"] <= frame["sim_upper"])
    return frame


class TestCalibrate:
    def test_good_coverage_leaves_the_intervals_alone(self):
        players, calibration = calibrate(toy_forecasts(0.8), ["2017-18", "2018-19"], toy_forecasts(0.8)["peak_identified"])
        assert not calibration.triggered and calibration.factor == 1.0
        assert (players["sim_lower_calibrated"] == players["sim_lower"]).all()
        assert (players["covered_calibrated"] == players["covered"]).all()

    def test_poor_coverage_widens_later_origins_only_from_earlier_errors(self):
        forecasts = toy_forecasts(0.3)
        players, calibration = calibrate(forecasts, ["2017-18", "2018-19"], forecasts["peak_identified"])
        assert calibration.triggered
        first = players[players["origin"] == "2017-18"]
        second = players[players["origin"] == "2018-19"]
        assert (first["calibration_factor"] == 1.0).all() and not first["origin_calibrated"].any()
        expected = np.quantile(first["scale_needed"], config.INTERVAL_LEVEL)
        assert second["calibration_factor"].iloc[0] == pytest.approx(max(1.0, expected))
        assert calibration.rolling[1]["origins_used"] == ["2017-18"]

    def test_calibrated_intervals_stay_ordered_and_at_least_as_wide(self):
        forecasts = toy_forecasts(0.3)
        players, calibration = calibrate(forecasts, ["2017-18", "2018-19"], forecasts["peak_identified"])
        assert calibration.factor >= 1.0
        assert (players["sim_lower_calibrated"] <= players["sim_median"]).all()
        assert (players["sim_median"] <= players["sim_upper_calibrated"]).all()
        assert (players["sim_lower_calibrated"] <= players["sim_lower"]).all()
        assert (players["sim_upper_calibrated"] >= players["sim_upper"]).all()


def test_repeated_players_are_flagged_at_later_origins_only():
    players = pd.DataFrame({"origin": ["2018-19", "2017-18", "2018-19"], "fpl_code": [1, 1, 2]})
    assert repeated_players(players).tolist() == [True, False, False]


@pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built (python -m src.panel)")
@pytest.mark.parametrize("origin", config.BACKTEST_ORIGINS)
def test_real_training_data_never_goes_past_the_origin(origin):
    panel = pd.read_parquet(config.PANEL_PATH)
    training = training_panel(panel, origin)
    assert training["season_year"].max() <= config.season_start_year(origin)
    assert (training["debut_season"].map(config.season_start_year) <= config.season_start_year(origin)).all()


@pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built (python -m src.panel)")
@pytest.mark.parametrize("origin", config.BACKTEST_ORIGINS)
def test_real_restricted_training_data(origin):
    panel = pd.read_parquet(config.PANEL_PATH)
    training = training_panel(panel, origin, first_seen_max_age=YOUNG)
    assert len(training) > 0
    assert training["first_seen_age"].notna().all() and (training["first_seen_age"] <= YOUNG).all()
    assert training["season_year"].max() <= config.season_start_year(origin)
    assert (training["debut_season"].map(config.season_start_year) <= config.season_start_year(origin)).all()


@pytest.mark.skipif(not config.BACKTEST_PLAYERS_PATH.exists(), reason="backtest not run (python -m src.validate)")
def test_real_backtest_calibrated_intervals_are_ordered():
    players = pd.read_csv(config.BACKTEST_PLAYERS_PATH)
    assert (players["calibration_factor"] >= 1.0).all()
    assert (players["sim_lower_calibrated"] <= players["sim_median"]).all()
    assert (players["sim_median"] <= players["sim_upper_calibrated"]).all()
    first = players[players["origin"] == config.BACKTEST_ORIGINS[0]]
    assert (first["calibration_factor"] == 1.0).all()
