"""Tests for src/panel.py: ages, age centring and the zero-fill logic with its observed/filled flag.

The check on the real panel skips itself if the panel has not been built yet
(data/ is not in git).
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import config
from src.clean import age_on
from src.panel import (
    build_panel,
    centre_age,
    completed_age,
    decimal_age,
    first_seen_young,
    pre_panel_minutes,
    truncation_counts,
)

SEASONS = ("2016-17", "2017-18", "2018-19", "2019-20", "2020-21")


def observed(rows: list[tuple[int, str, str, int]]) -> pd.DataFrame:
    """A minimal observed table from (fpl_code, season, position, minutes) tuples."""
    frame = pd.DataFrame(rows, columns=["fpl_code", "season", "position", "minutes"])
    frame["name"] = "Player " + frame["fpl_code"].astype(str)
    frame["web_name"] = "P" + frame["fpl_code"].astype(str)
    frame["team_code"] = 1
    return frame


def born(**dates: str) -> pd.Series:
    """Birth dates keyed by fpl_code, e.g. born(**{"1": "2000-01-01"})."""
    return pd.Series({int(code): pd.Timestamp(value) for code, value in dates.items()})


def row(panel: pd.DataFrame, code: int, season: str) -> pd.Series:
    """The single panel row of one player in one season."""
    found = panel[(panel["fpl_code"] == code) & (panel["season"] == season)]
    assert len(found) == 1, f"expected one row for {code} in {season}, found {len(found)}"
    return found.iloc[0]


class TestAges:
    def test_centring_at_21(self):
        assert centre_age(21.0) == 0.0
        assert centre_age(19.5) == -1.5
        assert centre_age(np.array([17.0, 30.0])).tolist() == [-4.0, 9.0]

    def test_the_centre_is_the_configured_one(self):
        assert config.MODEL_AGE_CENTRE == 21.0
        assert centre_age(25.0, centre=20.0) == 5.0

    def test_decimal_age_on_1_september(self):
        ages = decimal_age(pd.Series(pd.to_datetime(["2005-03-01", "2004-09-01"])), pd.Series([2025, 2025]))
        assert ages.tolist() == pytest.approx([20.5, 21.0], abs=0.01)

    def test_completed_age_agrees_with_the_u21_rule(self):
        births = pd.Series(pd.date_range("2003-01-01", "2006-12-31", freq="7D"))
        for year in (2018, 2025):
            expected = age_on(births, date(year, 9, 1))
            result = completed_age(births, pd.Series(year, index=births.index))
            assert (result == expected).all()

    def test_birthday_on_the_cut_off_date_counts(self):
        # floor(decimal age) would give 20 here; the U21 rule says 21.
        result = completed_age(pd.Series(pd.to_datetime(["2004-09-01"])), pd.Series([2025]))
        assert result.iloc[0] == 21

    def test_missing_birth_date_gives_missing_ages(self):
        births = pd.Series(pd.to_datetime([None, "2004-09-02"]))
        assert completed_age(births, pd.Series([2025, 2025])).isna().tolist() == [True, False]
        assert decimal_age(births, pd.Series([2025, 2025])).isna().tolist() == [True, False]


class TestZeroFill:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        rows = observed([
            # Player 1: plays, listed with 0 minutes, missing for two seasons, back.
            (1, "2016-17", "MID", 900), (1, "2017-18", "MID", 0), (1, "2020-21", "FWD", 500),
            # Player 2: listed but never plays.
            (2, "2016-17", "DEF", 0), (2, "2017-18", "DEF", 0),
            # Player 3: listed with 0 minutes before his debut, then plays, then leaves.
            (3, "2017-18", "DEF", 0), (3, "2018-19", "DEF", 1200),
            # Player 4: plays at 26, then leaves.
            (4, "2016-17", "GK", 3420),
        ])
        births = born(**{"1": "1998-01-01", "2": "1998-01-01", "3": "1999-01-01", "4": "1990-01-01"})
        return build_panel(rows, births, seasons=SEASONS)

    def test_players_who_never_played_are_left_out(self, panel):
        assert 2 not in set(panel["fpl_code"])

    def test_one_row_per_player_per_season(self, panel):
        assert not panel.duplicated(["fpl_code", "season"]).any()

    def test_absent_seasons_after_a_debut_are_filled_with_zero(self, panel):
        for season in ("2018-19", "2019-20"):
            filled = row(panel, 1, season)
            assert bool(filled["filled"]) and filled["minutes"] == 0 and filled["minutes_share"] == 0
            assert filled["zero_source"] == "filled"

    def test_listed_with_no_minutes_is_an_observed_zero(self, panel):
        listed = row(panel, 1, "2017-18")
        assert not bool(listed["filled"]) and listed["zero_source"] == "observed"
        pre_debut = row(panel, 3, "2017-18")
        assert not bool(pre_debut["filled"]) and pre_debut["zero_source"] == "observed"

    def test_seasons_with_minutes_have_no_zero_flag(self, panel):
        assert pd.isna(row(panel, 1, "2016-17")["zero_source"])
        assert row(panel, 1, "2016-17")["minutes_share"] == pytest.approx(900 / config.MAX_SEASON_MINUTES)

    def test_seasons_before_a_debut_are_never_filled(self, panel):
        assert row(panel, 3, "2017-18")["zero_source"] == "observed"
        assert panel[(panel["fpl_code"] == 3) & (panel["season"] == "2016-17")].empty

    def test_filling_runs_to_the_last_season(self, panel):
        # Player 3 leaves after 2018-19; 2019-20 and 2020-21 are filled.
        assert [bool(row(panel, 3, s)["filled"]) for s in ("2019-20", "2020-21")] == [True, True]

    def test_filling_stops_after_the_age_limit(self, panel):
        # Player 4, born 1 Jan 1990: 26 on 1 Sep 2016, 27 in 2017-18, 28 in 2018-19.
        seasons = panel[panel["fpl_code"] == 4].set_index("season")
        assert seasons.loc["2017-18", "age_completed"] == config.ZERO_FILL_MAX_AGE
        assert bool(seasons.loc["2017-18", "filled"])
        assert "2018-19" not in seasons.index

    def test_a_filled_season_carries_the_last_known_position(self, panel):
        assert row(panel, 1, "2018-19")["position"] == "MID"
        assert row(panel, 1, "2020-21")["position"] == "FWD"

    def test_flag_and_zero_source_agree(self, panel):
        assert (panel["filled"] == panel["zero_source"].eq("filled").fillna(False)).all()
        zeros = panel["minutes"] == 0
        assert panel.loc[zeros, "zero_source"].notna().all()
        assert panel.loc[~zeros, "zero_source"].isna().all()

    def test_age_centring_in_the_panel(self, panel):
        assert (panel["age_c"] == panel["age"] - config.MODEL_AGE_CENTRE).all()

    def test_no_birth_date_means_no_filling(self):
        rows = observed([(5, "2016-17", "MID", 900)])
        panel = build_panel(rows, pd.Series(dtype="datetime64[s]"), seasons=SEASONS)
        assert len(panel) == 1 and not panel["filled"].any()
        assert panel["age"].isna().all()

    def test_a_player_listed_twice_in_a_season_is_an_error(self):
        rows = observed([(6, "2016-17", "MID", 900), (6, "2016-17", "MID", 100)])
        with pytest.raises(ValueError):
            build_panel(rows, born(**{"6": "2000-01-01"}), seasons=SEASONS)


class TestFirstSeenAge:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        rows = observed([
            # Player 1: listed with 0 minutes at 18, first minutes at 19, then leaves (filled seasons follow).
            (1, "2016-17", "MID", 0), (1, "2017-18", "MID", 500),
            # Player 2: first minutes at 24.
            (2, "2016-17", "DEF", 900),
            # Player 3: no birth date.
            (3, "2016-17", "FWD", 300),
        ])
        panel = build_panel(rows, born(**{"1": "1998-06-01", "2": "1992-06-01"}), seasons=SEASONS)
        panel["tm_player_id"] = panel["fpl_code"].map({1: 11, 2: 22, 3: 33})
        return panel

    def test_it_is_the_age_in_the_first_season_with_minutes(self, panel):
        # Listed at 18 with no minutes, so first seen at 19, not 18.
        assert (panel.loc[panel["fpl_code"] == 1, "first_seen_age"] == 19).all()
        assert (panel.loc[panel["fpl_code"] == 2, "first_seen_age"] == 24).all()

    def test_filled_seasons_carry_it_too(self, panel):
        filled = panel[(panel["fpl_code"] == 1) & panel["filled"]]
        assert len(filled) > 0 and (filled["first_seen_age"] == 19).all()

    def test_no_birth_date_gives_no_first_seen_age(self, panel):
        assert panel.loc[panel["fpl_code"] == 3, "first_seen_age"].isna().all()

    def test_first_seen_young_uses_the_threshold(self, panel):
        young = panel.loc[first_seen_young(panel, 20), "fpl_code"].unique()
        assert set(young) == {1}
        assert set(panel.loc[first_seen_young(panel, 24), "fpl_code"].unique()) == {1, 2}

    def test_truncation_check_finds_pre_panel_minutes(self, panel):
        tm = pd.DataFrame({
            "tm_player_id": [11, 22, 22, 33],
            "season_year": [2015, 2012, 2016, 2014],  # 2016 is inside the panel, so it is ignored
            "minutes": [300, 450, 900, 100],
        })
        hits = pre_panel_minutes(panel, tm).set_index("fpl_code")
        assert set(hits.index) == {1, 2}  # player 3 has no first_seen_age, so he is in neither group
        assert bool(hits.loc[1, "first_seen_young"]) and hits.loc[1, "tm_first_seen_age"] == 17
        # Player 2, born 1 Jun 1992, was 20 on 1 Sep 2012: really first seen young.
        assert not bool(hits.loc[2, "first_seen_young"]) and bool(hits.loc[2, "young_by_tm"])
        assert hits.loc[2, "tm_minutes_before_panel"] == 450
        counts = dict(truncation_counts(panel, pre_panel_minutes(panel, tm)))
        assert counts["  of those, 20 or under in their first such season"] == 1


@pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built (python -m src.panel)")
def test_real_first_seen_age_is_the_age_in_the_debut_season():
    panel = pd.read_parquet(config.PANEL_PATH)
    debut_rows = panel[panel["season"] == panel["debut_season"]].set_index("fpl_code")
    players = panel.drop_duplicates("fpl_code").set_index("fpl_code")
    assert (players["first_seen_age"].fillna(-1) == debut_rows["age_completed"].reindex(players.index).fillna(-1)).all()
    assert (panel.groupby("fpl_code")["first_seen_age"].nunique(dropna=False) == 1).all()


@pytest.mark.skipif(not config.PANEL_PATH.exists(), reason="panel not built (python -m src.panel)")
def test_real_panel_keeps_its_promises():
    panel = pd.read_parquet(config.PANEL_PATH)
    assert not panel.duplicated(["fpl_code", "season"]).any()
    filled = panel[panel["filled"]]
    assert (filled["minutes"] == 0).all()
    assert (filled["age_completed"] <= config.ZERO_FILL_MAX_AGE).all()
    debut_years = filled["debut_season"].map(config.season_start_year)
    assert (filled["season_year"] > debut_years).all()
    assert panel["minutes_share"].between(0, 1).all()
