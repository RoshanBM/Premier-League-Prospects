"""Tests for the U21 age logic around the cut-off date (src/clean.py and src/cohort.py)."""

from datetime import date

import pandas as pd
import pytest

from src import config
from src.clean import age_on, exact_age
from src.cohort import is_u21


def dates(*values: str | None) -> pd.Series:
    """A Series of birth dates; None means missing."""
    return pd.Series(pd.to_datetime(list(values)))


def test_the_agreed_rule_is_under_21_on_1_september_2025():
    assert config.U21_CUTOFF_DATE == date(2025, 9, 1)
    assert config.U21_AGE_LIMIT == 21


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


def test_exact_age_in_years_with_decimals():
    assert exact_age(dates("2004-09-01"), config.U21_CUTOFF_DATE).iloc[0] == pytest.approx(21.0, abs=0.01)
    assert exact_age(dates("2005-03-01"), config.U21_CUTOFF_DATE).iloc[0] == pytest.approx(20.5, abs=0.01)
