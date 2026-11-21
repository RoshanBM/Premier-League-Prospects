"""No duplicate player-seasons: unit tests on made-up tables, then checks on the real files.

The checks on real files skip themselves if the data has not been downloaded
and processed yet (data/ is not in git).
"""

import pandas as pd
import pytest

from src import config
from src.clean import fpl_birth_dates, fpl_player_seasons


def players_raw(codes: list[int], element_types: list[int], birth_dates: list[str] | None = None) -> pd.DataFrame:
    """A minimal players_raw.csv table."""
    frame = pd.DataFrame(
        {
            "code": codes,
            "id": range(1, len(codes) + 1),
            "first_name": [f"First{c}" for c in codes],
            "second_name": [f"Second{c}" for c in codes],
            "web_name": [f"Web{c}" for c in codes],
            "element_type": element_types,
            "team_code": 1,
            "minutes": 900,
        }
    )
    if birth_dates is not None:
        frame["birth_date"] = birth_dates
    return frame


class TestFplPlayerSeasons:
    def test_one_row_per_player_per_season(self):
        seasons = fpl_player_seasons(
            {"2023-24": players_raw([10, 20], [2, 3]), "2024-25": players_raw([10, 30], [2, 4])}
        )
        assert len(seasons) == 4
        assert not seasons.duplicated(["season", "code"]).any()

    def test_managers_are_left_out(self):
        seasons = fpl_player_seasons({"2024-25": players_raw([10, 99], [3, 5])})
        assert seasons["code"].tolist() == [10]

    def test_birth_dates_are_missing_before_fpl_had_them(self):
        seasons = fpl_player_seasons(
            {"2023-24": players_raw([10], [2]), "2024-25": players_raw([10], [2], ["2004-09-02"])}
        )
        assert seasons.set_index("season")["birth_date"].isna().to_dict() == {"2023-24": True, "2024-25": False}


class TestFplBirthDates:
    def test_one_date_per_player_from_any_season(self):
        seasons = fpl_player_seasons(
            {"2023-24": players_raw([10], [2]), "2024-25": players_raw([10], [2], ["2004-09-02"])}
        )
        assert fpl_birth_dates(seasons).to_dict() == {10: pd.Timestamp("2004-09-02")}

    def test_conflicting_dates_are_an_error(self):
        seasons = fpl_player_seasons(
            {"2024-25": players_raw([10], [2], ["2004-09-02"]), "2025-26": players_raw([10], [2], ["2004-09-03"])}
        )
        with pytest.raises(ValueError):
            fpl_birth_dates(seasons)


# ---------------------------------------------------------------------------
# The real files
# ---------------------------------------------------------------------------

RAW_FILES = [config.fpl_raw_path(season, "players_raw.csv") for season in config.FPL_SEASONS]
needs_raw = pytest.mark.skipif(
    not all(path.exists() for path in RAW_FILES), reason="raw FPL files not downloaded (python -m src.load_data)"
)
needs_processed = pytest.mark.skipif(
    not all(p.exists() for p in (config.PLAYER_SEASON_PATH, config.U21_CARDS_PATH, config.MATCHES_PATH)),
    reason="processed files not built (python -m src.matching, then python -m src.features)",
)


@needs_raw
def test_real_fpl_seasons_have_one_row_per_player_per_season():
    from src.load_data import read_fpl_players

    seasons = fpl_player_seasons({season: read_fpl_players(season) for season in config.FPL_SEASONS})
    assert not seasons.duplicated(["season", "code"]).any()
    assert not seasons.duplicated(["season", "id"]).any()


@needs_processed
def test_player_season_table_has_one_row_per_player():
    table = pd.read_parquet(config.PLAYER_SEASON_PATH)
    assert table["fpl_code"].is_unique


@needs_processed
def test_u21_cards_have_one_row_per_player_and_only_u21s_who_played():
    cards = pd.read_parquet(config.U21_CARDS_PATH)
    assert cards["fpl_code"].is_unique
    assert cards["u21"].all()
    assert (cards["minutes"] > 0).all()
    assert (cards["age_on_cutoff"] < config.U21_AGE_LIMIT).all()


@needs_processed
def test_match_table_links_each_player_once():
    matches = pd.read_csv(config.MATCHES_PATH)
    assert matches["fpl_code"].is_unique
    assert matches["tm_player_id"].is_unique
