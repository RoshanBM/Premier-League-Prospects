"""Tests for src/calibration.py: the scale factors, rolling estimation without leakage, and widened intervals."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.calibration import (
    Calibration,
    calibration_factor,
    load_calibration,
    needs_calibration,
    rolling_factors,
    save_calibration,
    scale_needed,
    widen,
)

ORIGINS = ["2017-18", "2018-19", "2019-20", "2020-21"]


class TestScaleNeeded:
    def test_an_outcome_inside_the_interval_needs_at_most_one(self):
        s = scale_needed([0.45, 0.35, 0.4], [0.4, 0.4, 0.4], [0.3, 0.3, 0.3], [0.6, 0.6, 0.6])
        assert s.tolist() == pytest.approx([0.25, 0.5, 0.0])

    def test_an_outcome_above_uses_the_upper_side(self):
        assert scale_needed([0.8], [0.4], [0.3], [0.6])[0] == pytest.approx(2.0)

    def test_an_outcome_below_uses_the_lower_side(self):
        assert scale_needed([0.0], [0.4], [0.3], [0.6])[0] == pytest.approx(4.0)

    def test_an_interval_with_no_width_is_an_error(self):
        with pytest.raises(ValueError):
            scale_needed([0.5], [0.4], [0.3], [0.4])


class TestFactor:
    def test_it_is_the_level_percentile(self):
        assert calibration_factor(np.arange(1.0, 11.0), level=0.8) == pytest.approx(np.quantile(np.arange(1.0, 11.0), 0.8))

    def test_it_is_never_below_one(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        for _ in range(50):
            assert calibration_factor(rng.uniform(0, 1.2, size=rng.integers(1, 40))) >= 1.0
        assert calibration_factor([0.1, 0.2, 0.3]) == 1.0

    def test_no_past_forecasts_means_no_widening(self):
        assert calibration_factor([]) == 1.0

    def test_the_trigger(self):
        assert needs_calibration(0.65, trigger=0.70)
        assert not needs_calibration(0.70, trigger=0.70)


def forecasts(scales_by_origin: dict[str, list[float]]) -> pd.DataFrame:
    """A forecast table with the given s_i at each origin."""
    return pd.DataFrame([(origin, s) for origin, scales in scales_by_origin.items() for s in scales],
                        columns=["origin", "scale_needed"])


class TestRolling:
    @pytest.fixture
    def table(self) -> pd.DataFrame:
        return forecasts({"2017-18": [1.0, 2.0, 3.0], "2018-19": [4.0, 5.0], "2019-20": [6.0], "2020-21": [50.0]})

    def test_each_origin_uses_only_earlier_origins(self, table):
        for i, factor in enumerate(rolling_factors(table, ORIGINS)):
            assert factor.origins_used == tuple(ORIGINS[:i])
            assert factor.forecasts_used == int(table["origin"].isin(ORIGINS[:i]).sum())
            expected = calibration_factor(table.loc[table["origin"].isin(ORIGINS[:i]), "scale_needed"])
            assert factor.factor == pytest.approx(expected)

    def test_the_first_origin_stays_uncalibrated(self, table):
        first = rolling_factors(table, ORIGINS)[0]
        assert first.factor == 1.0 and not first.calibrated and first.origins_used == ()

    def test_changing_a_later_origin_never_changes_an_earlier_factor(self, table):
        before = rolling_factors(table, ORIGINS)
        changed = table.copy()
        changed.loc[changed["origin"] == "2018-19", "scale_needed"] = 1000.0
        after = rolling_factors(changed, ORIGINS)
        # 2018-19's own factor uses only 2017-18, so it must not move either.
        assert [f.factor for f in after[:2]] == [f.factor for f in before[:2]]
        # The origins after it do use its errors.
        assert after[2].factor > before[2].factor and after[3].factor > before[3].factor

    def test_an_origin_never_uses_its_own_errors(self, table):
        changed = table.copy()
        changed.loc[changed["origin"] == "2020-21", "scale_needed"] = 0.0
        assert rolling_factors(changed, ORIGINS)[3].factor == rolling_factors(table, ORIGINS)[3].factor

    def test_every_factor_is_at_least_one(self, table):
        small = table.assign(scale_needed=0.1)
        assert all(f.factor >= 1.0 for f in rolling_factors(small, ORIGINS))
        assert all(f.factor >= 1.0 for f in rolling_factors(table, ORIGINS))

    def test_origins_out_of_time_order_are_rejected(self, table):
        with pytest.raises(ValueError):
            rolling_factors(table, ["2018-19", "2017-18"])


class TestWiden:
    def test_order_is_kept_after_calibration(self):
        rng = np.random.default_rng(config.RANDOM_SEED)
        median = rng.uniform(-0.2, 1.0, 200)
        lower = median - rng.uniform(0.0, 0.5, 200)
        upper = median + rng.uniform(0.0, 0.5, 200)
        for factor in (1.0, 1.43, 3.0, rng.uniform(1, 2, 200)):
            new_lower, new_upper = widen(median, lower, upper, factor)
            assert (new_lower <= median).all() and (median <= new_upper).all()
            assert (new_lower <= lower + 1e-12).all() and (new_upper >= upper - 1e-12).all()

    def test_each_side_is_stretched_by_the_factor(self):
        lower, upper = widen([0.4], [0.3], [0.6], 2.0)
        assert lower[0] == pytest.approx(0.2) and upper[0] == pytest.approx(0.8)

    def test_a_factor_of_one_changes_nothing(self):
        lower, upper = widen([0.4], [0.3], [0.6], 1.0)
        assert lower[0] == pytest.approx(0.3) and upper[0] == pytest.approx(0.6)

    def test_a_factor_below_one_is_an_error(self):
        with pytest.raises(ValueError):
            widen([0.4], [0.3], [0.6], 0.9)


def test_saving_and_loading_keeps_the_calibration(tmp_path):
    calibration = Calibration(triggered=True, reason="test", raw_coverage=0.6, trigger_coverage=0.7, level=0.8,
                              factor=1.4, origins_used=ORIGINS, forecasts_used=10, first_seen_max_age=20,
                              rolling=[{"origin": "2017-18", "factor": 1.0, "origins_used": [], "forecasts_used": 0}])
    path = tmp_path / "calibration.json"
    save_calibration(calibration, path)
    copy = load_calibration(path)
    assert copy == calibration
    assert copy.rolling_factor("2017-18") == 1.0


@pytest.mark.skipif(not config.CALIBRATION_PATH.exists(), reason="calibration not built (python -m src.validate)")
def test_real_calibration_is_consistent():
    calibration = load_calibration()
    assert calibration.factor >= 1.0
    assert all(r["factor"] >= 1.0 for r in calibration.rolling)
    assert [r["origin"] for r in calibration.rolling] == list(config.BACKTEST_ORIGINS)
    for i, r in enumerate(calibration.rolling):
        assert r["origins_used"] == list(config.BACKTEST_ORIGINS[:i])
    if not calibration.triggered:
        assert calibration.factor == 1.0
