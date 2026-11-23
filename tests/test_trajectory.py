"""Tests for src/trajectory.py: the projection formulas agree with statsmodels, and the fit checks work.

The model is fitted on simulated data with known parameters, so these tests
need no downloaded data.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src import config
from src.trajectory import (
    TrajectoryModel,
    chi2_mixture_p_value,
    fit_mixed_model,
    player_posterior,
    random_effect_problems,
    training_rows,
)


@pytest.fixture(scope="module")
def simulated() -> pd.DataFrame:
    """120 players, 2 to 6 seasons each, from a known random-intercept-and-slope model."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []
    shift = {"DEF": 0.0, "FWD": -0.08, "GK": 0.02, "MID": -0.04}
    for code in range(120):
        position = ["DEF", "FWD", "GK", "MID"][code % 4]
        u0, u1 = rng.multivariate_normal([0, 0], [[0.03, 0.001], [0.001, 0.001]])
        start = rng.uniform(17, 24)
        for k in range(rng.integers(2, 7)):
            age_c = start + k - 21
            share = 0.3 + shift[position] + 0.04 * age_c - 0.005 * age_c**2 + u0 + u1 * age_c + rng.normal(0, 0.2)
            rows.append({"fpl_code": code, "position": position, "age_c": age_c, "minutes_share": share,
                         "season": f"{2016 + k}-{17 + k}"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted(simulated):
    result, attempt = fit_mixed_model(simulated, random_slope=True)
    return result, attempt, TrajectoryModel.from_result(result, simulated, random_slope=True)


def test_blups_match_statsmodels(simulated, fitted):
    result, _, model = fitted
    for code, group in simulated.groupby("fpl_code"):
        posterior = player_posterior(model, group["age_c"].to_numpy(), group["position"].to_numpy(),
                                     group["minutes_share"].to_numpy())
        assert np.allclose(posterior.mean(model.fe_params)[0], np.asarray(result.random_effects[code]), atol=1e-10)
        assert np.allclose(posterior.cov, np.asarray(result.random_effects_cov[code]), atol=1e-10)


def test_fixed_design_matches_statsmodels_predictions(simulated, fitted):
    result, _, model = fitted
    ours = model.population_mean(simulated["age_c"].to_numpy(), simulated["position"].to_numpy())
    assert np.allclose(ours, np.asarray(result.predict(simulated)), atol=1e-10)


def test_a_player_with_no_seasons_gets_the_population_prior(fitted):
    _, _, model = fitted
    posterior = player_posterior(model, np.array([]), [], np.array([]))
    assert np.allclose(posterior.mean(model.fe_params), 0.0)
    assert np.allclose(posterior.cov, model.cov_re)


def test_more_seasons_mean_less_uncertainty(simulated, fitted):
    _, _, model = fitted
    group = simulated.groupby("fpl_code").filter(lambda g: len(g) >= 4).query("fpl_code == fpl_code.min()")
    one = player_posterior(model, group["age_c"].to_numpy()[:1], group["position"].to_numpy()[:1],
                           group["minutes_share"].to_numpy()[:1])
    four = player_posterior(model, group["age_c"].to_numpy()[:4], group["position"].to_numpy()[:4],
                            group["minutes_share"].to_numpy()[:4])
    assert np.trace(four.cov) < np.trace(one.cov) < np.trace(model.cov_re)


def test_saving_and_loading_keeps_the_numbers(fitted):
    _, _, model = fitted
    copy = TrajectoryModel.from_dict(model.to_dict())
    assert np.allclose(copy.fe_params, model.fe_params) and np.allclose(copy.cov_re, model.cov_re)
    assert copy.random_slope == model.random_slope and copy.fe_names == model.fe_names


def test_the_training_restriction_is_saved_and_old_files_mean_all_players(fitted):
    _, _, model = fitted
    model.first_seen_max_age = 20
    assert TrajectoryModel.from_dict(model.to_dict()).first_seen_max_age == 20
    old = model.to_dict()
    del old["first_seen_max_age"]  # a file written before the survivorship fix
    assert TrajectoryModel.from_dict(old).first_seen_max_age is None
    model.first_seen_max_age = None


class TestTrainingRows:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        return pd.DataFrame({
            "fpl_code": [1, 1, 2, 2, 3],
            "age_completed": pd.array([19, 20, 24, 25, 18], dtype="Int64"),
            "age": [19.5, 20.5, 24.5, 25.5, 18.5],
            "position": ["MID"] * 5,
            "first_seen_age": pd.array([19, 19, 24, 24, pd.NA], dtype="Int64"),
        })

    def test_only_players_first_seen_young_are_kept(self, panel):
        rows = training_rows(panel, (17, 30), first_seen_max_age=20)
        assert set(rows["fpl_code"]) == {1}
        assert (rows["first_seen_age"] <= 20).all()

    def test_none_keeps_every_player(self, panel):
        assert set(training_rows(panel, (17, 30), first_seen_max_age=None)["fpl_code"]) == {1, 2, 3}

    def test_the_age_range_still_applies(self, panel):
        assert set(training_rows(panel, (17, 19), first_seen_max_age=None)["age_completed"]) == {18, 19}


def test_unknown_positions_are_rejected(fitted):
    _, _, model = fitted
    with pytest.raises(ValueError):
        model.fixed_design(np.array([0.0]), ["CB"])


class TestFitChecks:
    def test_a_healthy_covariance_passes(self):
        assert random_effect_problems(np.array([[0.03, 0.001], [0.001, 0.001]])) == []

    def test_perfect_correlation_is_flagged(self):
        assert random_effect_problems(np.array([[0.04, 0.002], [0.002, 0.0001]]))

    def test_a_zero_variance_is_flagged(self):
        assert random_effect_problems(np.array([[0.03, 0.0], [0.0, 0.0]]))


def test_chi2_mixture_p_value():
    assert chi2_mixture_p_value(0.0) == 1.0
    # 0.5 x P(chi2_1 > 2.71) + 0.5 x P(chi2_2 > 2.71) = 0.5 x 0.0997 + 0.5 x 0.2579
    assert chi2_mixture_p_value(2.71) == pytest.approx(0.1788, abs=1e-3)
    assert math.isclose(chi2_mixture_p_value(1e6), 0.0, abs_tol=1e-12)
