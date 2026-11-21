"""Tests for src/features.py: per-90 arithmetic, percentiles and the 1 to 20 mapping."""

import math

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import add_features, attribute_score, mid_rank_percentile, per_90


class TestPer90:
    def test_total_divided_by_minutes_times_90(self):
        result = per_90(pd.Series([5.0, 1.0, 0.0, 2.7]), pd.Series([900, 450, 1800, 1350]))
        assert result.tolist() == pytest.approx([0.5, 0.2, 0.0, 0.18])

    def test_missing_below_the_threshold_and_kept_at_it(self):
        result = per_90(pd.Series([3.0, 3.0]), pd.Series([config.MIN_MINUTES_FOR_PER90 - 1, config.MIN_MINUTES_FOR_PER90]))
        assert math.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx(0.6)

    def test_threshold_can_be_changed(self):
        assert per_90(pd.Series([1.0]), pd.Series([90]), min_minutes=90).iloc[0] == pytest.approx(1.0)

    def test_negative_totals_such_as_goals_minus_xg_work(self):
        assert per_90(pd.Series([-1.5]), pd.Series([900])).iloc[0] == pytest.approx(-0.15)


class TestMidRankPercentile:
    def test_distinct_values(self):
        assert mid_rank_percentile(pd.Series([10, 20, 30, 40])).tolist() == pytest.approx([0.125, 0.375, 0.625, 0.875])

    def test_ties_share_one_percentile(self):
        # Three players on 0: none below, three level, so (0 + 3/2) / 4.
        assert mid_rank_percentile(pd.Series([0, 0, 0, 5])).tolist() == pytest.approx([0.375, 0.375, 0.375, 0.875])

    def test_missing_values_stay_missing_and_are_not_counted(self):
        result = mid_rank_percentile(pd.Series([1.0, np.nan, 3.0]))
        assert result.iloc[[0, 2]].tolist() == pytest.approx([0.25, 0.75])
        assert math.isnan(result.iloc[1])

    def test_always_strictly_between_0_and_1(self):
        values = pd.Series(np.random.default_rng(config.RANDOM_SEED).normal(size=500))
        result = mid_rank_percentile(values)
        assert (result > 0).all() and (result < 1).all()


class TestAttributeScore:
    @pytest.mark.parametrize(
        "percentile, expected",
        [(0.0, 1), (0.0499, 1), (0.05, 2), (0.5, 11), (0.7, 15), (0.9499, 19), (0.95, 20), (1.0, 20)],
    )
    def test_examples(self, percentile, expected):
        assert attribute_score(pd.Series([percentile])).iloc[0] == expected

    def test_exact_boundary_survives_floating_point_error(self):
        # 0.5 computed as an average can come out as 0.49999999999999994; it must still give 11.
        assert attribute_score(pd.Series([0.49999999999999994])).iloc[0] == 11

    def test_whole_range_stays_within_1_and_20_and_reaches_both(self):
        scores = attribute_score(pd.Series(np.linspace(0, 1, 10_001)))
        assert scores.min() == config.ATTRIBUTE_MIN and scores.max() == config.ATTRIBUTE_MAX

    def test_never_decreases_as_the_percentile_rises(self):
        scores = attribute_score(pd.Series(np.linspace(0, 1, 10_001))).astype(int)
        assert (scores.diff().dropna() >= 0).all()

    def test_each_value_covers_a_twentieth_of_the_range(self):
        # 20,000 evenly spaced percentiles below 1: each attribute value should get 1,000 of them.
        scores = attribute_score(pd.Series((np.arange(20_000) + 0.5) / 20_000))
        assert scores.value_counts().sort_index().tolist() == [1_000] * 20

    def test_missing_stays_missing(self):
        assert attribute_score(pd.Series([np.nan, 0.5])).isna().tolist() == [True, False]

    @pytest.mark.parametrize("percentile", [-0.01, 1.01])
    def test_out_of_range_is_rejected(self, percentile):
        with pytest.raises(ValueError):
            attribute_score(pd.Series([percentile]))


def made_up_players() -> pd.DataFrame:
    """Two qualifying defenders, one defender below 450 minutes, two goalkeepers and one midfielder."""
    return pd.DataFrame(
        {
            "position": ["DEF", "DEF", "DEF", "GK", "GK", "MID"],
            "minutes": [900, 1800, 300, 900, 900, 900],
            "goals_scored": [1, 0, 2, 0, 0, 3],
            "expected_goals": [0.5, 0.2, 1.0, 0.0, 0.0, 2.0],
            "expected_assists": [0.3, 0.6, 0.1, 0.0, 0.0, 1.0],
            "assists": [0, 2, 1, 0, 0, 2],
            "tackles": [20, 10, 5, 0, 0, 15],
            "clearances_blocks_interceptions": [40, 30, 10, 5, 5, 10],
            "recoveries": [50, 60, 20, 20, 20, 40],
            "saves": [0, 0, 0, 30, 10, 0],
            "expected_goals_conceded": [10.0, 20.0, 5.0, 10.0, 10.0, 8.0],
            "goals_conceded": [9, 22, 6, 8, 12, 7],
        }
    )


class TestAddFeatures:
    def test_no_attributes_below_the_minutes_threshold(self):
        result = add_features(made_up_players())
        attributes = ["finishing", "shot_threat", "chance_creation", "defending", "ball_recovery", "goalkeeping"]
        assert result.loc[2, attributes].isna().all()

    def test_goalkeepers_get_only_goalkeeping(self):
        result = add_features(made_up_players())
        keepers, outfield = result[result["position"] == "GK"], result[result["position"] != "GK"]
        assert keepers["goalkeeping"].notna().all() and keepers["shot_threat"].isna().all()
        assert outfield["goalkeeping"].isna().all()

    def test_percentiles_are_within_position_and_ignore_low_minute_players(self):
        result = add_features(made_up_players())
        # xG per 90: defenders 0.05 and 0.01 (the 300-minute defender is not in the pool of two).
        assert result.loc[[0, 1], "xg_pct"].tolist() == pytest.approx([0.75, 0.25])
        assert result.loc[[0, 1], "shot_threat"].tolist() == [16, 6]
        # A pool of one sits on the median.
        assert result.loc[5, "xg_pct"] == pytest.approx(0.5)
        assert result.loc[5, "shot_threat"] == 11

    def test_two_input_attributes_average_the_percentiles(self):
        result = add_features(made_up_players())
        expected = attribute_score(result[["goals_pct", "goals_minus_xg_pct"]].mean(axis=1))
        assert result["finishing"].tolist() == expected.tolist()

    def test_goals_prevented_rewards_conceding_fewer_than_expected(self):
        result = add_features(made_up_players())
        # Goalkeeper 3 conceded 8 from 10 expected, goalkeeper 4 conceded 12; 3 also made more saves.
        assert result.loc[3, "goalkeeping"] > result.loc[4, "goalkeeping"]
