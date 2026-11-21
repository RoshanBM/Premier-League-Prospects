"""Tests for the U21 age logic and the cohort of any season (src/clean.py, src/config.py and src/cohort.py).

The check on real files skips itself if the data has not been downloaded and
matched yet (data/ is not in git).
"""

from datetime import date

import pandas as pd
import pytest

from src import config
from src.clean import age_on, exact_age
from src.cohort import cutoff_birthday, in_cohort, is_u21, output_path, season_players


def dates(*values: str | None) -> pd.Series:
    """A Series of birth dates; None means missing."""
    return pd.Series(pd.to_datetime(list(values)))


def test_the_agreed_rule_is_under_21_on_1_september_2025():
    assert config.U21_CUTOFF_DATE == date(2025, 9, 1)
    assert config.U21_AGE_LIMIT == 21


def test_each_season_cuts_off_on_1_september_of_its_first_year():
    assert config.u21_cutoff("2018-19") == date(2018, 9, 1)
    assert config.u21_cutoff("2016-17") == date(2016, 9, 1)
    assert config.u21_cutoff(config.CURRENT_SEASON) == config.U21_CUTOFF_DATE


class TestAgeOn:
    @pytest.mark.parametrize(
        "born, age",
        [
            ("2004-09-01", 21),  # 21st birthday on the cut-off date itself
            ("2004-09-02", 20),  # one day younger
            ("2004-08-31", 21),
            ("2005-09-01", 20),
            ("2009-12-31", 15),
        ],
    )
    def test_age_on_the_cut_off_date(self, born, age):
        assert age_on(dates(born), config.U21_CUTOFF_DATE).iloc[0] == age

    def test_leap_day_birthday(self):
        born = dates("2004-02-29")
        assert age_on(born, date(2025, 2, 28)).iloc[0] == 20
        assert age_on(born, date(2025, 3, 1)).iloc[0] == 21
        assert age_on(born, date(2028, 2, 29)).iloc[0] == 24

    def test_missing_birth_date_gives_missing_age(self):
        assert age_on(dates(None, "2004-09-02"), config.U21_CUTOFF_DATE).isna().tolist() == [True, False]


class TestIsU21:
    @pytest.mark.parametrize(
        "born, expected",
        [("2004-09-01", False), ("2004-09-02", True), ("2004-08-31", False), ("2009-12-31", True)],
    )
    def test_cut_off_boundary(self, born, expected):
        assert bool(is_u21(dates(born)).iloc[0]) is expected

    def test_missing_birth_date_is_neither_in_nor_out(self):
        assert is_u21(dates(None)).isna().iloc[0]

    def test_another_cut_off_date(self):
        assert not bool(is_u21(dates("2005-09-01"), cutoff=date(2026, 9, 1)).iloc[0])
        assert bool(is_u21(dates("2005-09-02"), cutoff=date(2026, 9, 1)).iloc[0])

    def test_agrees_with_a_plain_date_comparison(self):
        births = pd.Series(pd.date_range("2003-01-01", "2006-12-31", freq="D"))
        assert (is_u21(births).astype(bool) == (births > pd.Timestamp("2004-09-01"))).all()


class TestCutoffBirthday:
    @pytest.mark.parametrize(
        "cutoff, birthday",
        [(date(2025, 9, 1), "2004-09-01"), (date(2018, 9, 1), "1997-09-01"), (date(2028, 2, 29), "2007-02-28")],
    )
    def test_the_latest_birth_date_that_is_too_old(self, cutoff, birthday):
        assert cutoff_birthday(cutoff) == pd.Timestamp(birthday)

    @pytest.mark.parametrize("cutoff", [date(2025, 9, 1), date(2018, 9, 1), date(2028, 2, 29)])
    def test_agrees_with_the_u21_rule(self, cutoff):
        # Born on the cut-off birthday: too old. Born a day later: U21.
        birthday = cutoff_birthday(cutoff)
        result = is_u21(pd.Series([birthday, birthday + pd.Timedelta(days=1)]), cutoff)
        assert result.tolist() == [False, True]


def test_the_cohort_is_u21_players_with_minutes():
    players = pd.DataFrame(
        {"u21": pd.array([True, True, False, pd.NA], dtype="boolean"), "minutes": [10, 0, 900, 900]}
    )
    # U21 with minutes; U21 without; too old; birth date unknown.
    assert in_cohort(players).tolist() == [True, False, False, False]


def test_a_non_default_cut_off_is_written_to_its_own_file():
    assert output_path("2018-19", date(2018, 9, 1)).name == "u21_cohort_2018_19.csv"
    assert output_path("2018-19", date(2019, 1, 1)).name == "u21_cohort_2018_19_cutoff_2019_01_01.csv"


def test_an_unknown_season_is_rejected():
    with pytest.raises(ValueError):
        season_players("2015-16")


def test_exact_age_in_years_with_decimals():
    assert exact_age(dates("2004-09-01"), config.U21_CUTOFF_DATE).iloc[0] == pytest.approx(21.0, abs=0.01)
    assert exact_age(dates("2005-03-01"), config.U21_CUTOFF_DATE).iloc[0] == pytest.approx(20.5, abs=0.01)


# ---------------------------------------------------------------------------
# The real files
# ---------------------------------------------------------------------------

NEEDED_FILES = [
    *(config.fpl_raw_path(season, name) for season in config.FPL_SEASONS for name in config.FPL_SEASON_FILES),
    config.MATCHES_PATH,
]
needs_data = pytest.mark.skipif(
    not all(path.exists() for path in NEEDED_FILES),
    reason="raw FPL files or the match table missing (python -m src.load_data, then python -m src.matching)",
)


@needs_data
def test_an_earlier_season_uses_its_own_minutes_and_cut_off():
    from src.load_data import read_fpl_players

    players = season_players("2018-19")
    raw = read_fpl_players("2018-19")
    assert (players["season"] == "2018-19").all()
    assert dict(zip(players["fpl_code"], players["minutes"])) == dict(zip(raw["code"], raw["minutes"]))
    cohort = players[in_cohort(players)]
    assert len(cohort) > 0
    assert (cohort["age_on_cutoff"] < config.U21_AGE_LIMIT).all()
    assert (cohort["birth_date"] > pd.Timestamp("1997-09-01")).all()
    assert players["starts"].isna().all()  # FPL has no starts before 2022-23
