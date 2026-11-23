"""Tests for src/similarity.py: same position only, and a player's similarity with himself is 1.

The check on the real output skips itself if it has not been built yet.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.similarity import cosine_similarity, similar_players, standardise

COLUMNS = ["a", "b", "c"]


def toy_players() -> pd.DataFrame:
    """Eight players at two positions with random stats."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    frame = pd.DataFrame(rng.normal(size=(8, 3)), columns=COLUMNS)
    frame["fpl_code"] = range(1, 9)
    frame["position"] = ["MID", "MID", "MID", "MID", "DEF", "DEF", "DEF", "DEF"]
    frame["age"] = [19.0, 20.0, 24.0, 30.0, 19.5, 20.5, 25.0, 31.0]
    frame["season_year"] = [2025, 2020, 2018, 2025, 2025, 2019, 2017, 2025]
    frame[COLUMNS] = standardise(frame, COLUMNS, ["position"]).to_numpy()
    return frame


class TestCosineSimilarity:
    def test_a_player_with_himself_is_one(self):
        vectors = toy_players()[COLUMNS].to_numpy()
        assert np.allclose(np.diag(cosine_similarity(vectors, vectors)), 1.0)

    def test_opposite_profiles_are_minus_one(self):
        assert cosine_similarity([1.0, -2.0], [-1.0, 2.0])[0, 0] == pytest.approx(-1.0)

    def test_scale_does_not_matter(self):
        assert cosine_similarity([1.0, 2.0], [3.0, 6.0])[0, 0] == pytest.approx(1.0)

    def test_an_all_zero_profile_has_no_similarity(self):
        assert np.isnan(cosine_similarity([0.0, 0.0], [1.0, 2.0])[0, 0])


class TestSimilarPlayers:
    def test_matches_are_at_the_same_position_only(self):
        players = toy_players()
        matches = similar_players(players, players, COLUMNS, n=5)
        position = players.set_index("fpl_code")["position"]
        assert (matches["fpl_code"].map(position) == matches["match_position"]).all()

    def test_a_player_is_never_his_own_match(self):
        players = toy_players()
        matches = similar_players(players, players, COLUMNS, n=5)
        assert (matches["fpl_code"] != matches["match_fpl_code"]).all()

    def test_ranks_follow_similarity(self):
        players = toy_players()
        matches = similar_players(players, players, COLUMNS, n=3)
        for _, group in matches.groupby("fpl_code"):
            assert group["similarity"].is_monotonic_decreasing
            assert group["rank"].tolist() == list(range(1, len(group) + 1))

    def test_age_window_and_earlier_seasons(self):
        players = toy_players()
        query = players[players["fpl_code"] == 1]  # MID, 19.0
        matches = similar_players(query, players, COLUMNS, n=5, age_window=1.0, before_season=2025)
        assert matches["match_fpl_code"].tolist() == [2]  # the only earlier MID within a year of age

    def test_standardising_gives_mean_0_and_sd_1_within_position(self):
        players = toy_players()
        for _, group in players.groupby("position"):
            assert np.allclose(group[COLUMNS].mean(), 0.0)
            assert np.allclose(group[COLUMNS].std(ddof=0), 1.0)

    def test_a_stat_with_no_spread_is_ignored(self):
        frame = pd.DataFrame({"x": [1.0, 1.0, 1.0], "position": ["GK"] * 3})
        assert (standardise(frame, ["x"], ["position"])["x"] == 0).all()


@pytest.mark.skipif(not config.SIMILAR_PLAYERS_PATH.exists() or not config.PLAYER_SEASON_PATH.exists(),
                    reason="similar players not built (python -m src.similarity)")
def test_real_comparisons_stay_within_position():
    table = pd.read_parquet(config.SIMILAR_PLAYERS_PATH)
    panel = pd.read_parquet(config.PANEL_PATH)
    position_by_season = panel.set_index(["fpl_code", "season"])["position"]
    keys = pd.MultiIndex.from_arrays([table["match_fpl_code"], table["match_season"]])
    assert (position_by_season.reindex(keys).to_numpy() == table["position"].to_numpy()).all()
    assert (table["fpl_code"] != table["match_fpl_code"]).all()
    assert table["similarity"].between(-1, 1).all()
