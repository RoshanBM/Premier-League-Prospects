"""Tests for src/ability.py: CA weights and contributions; PA intervals, the card's rank among
this season's prospects, the low-evidence flag, the sensitivity references and the range text.

The checks on the real projections skip themselves if they have not been built yet.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src import config
from src.ability import (
    CA_WEIGHTS,
    PA_DISPLAY_GREYED,
    PA_DISPLAY_NORMAL,
    PA_LABEL,
    PA_STATUS_EXPERIMENTAL,
    PA_STATUS_TOO_YOUNG,
    SENSITIVITY_REFERENCES,
    Weight,
    card_bounds,
    check_weights,
    current_ability,
    finish_pa,
    low_evidence,
    nearest_five_percent,
    peak_reference,
    percentile_against,
    persistence_shares,
    prospect_rank,
    range_text,
)
from src.features import ATTRIBUTES
from src.trajectory import TrajectoryModel, future_targets, project

ATTRIBUTE_COLUMNS = [attribute.column for attribute in ATTRIBUTES]
AGES = f"at {config.PEAK_AGE_RANGE[0]} to {config.PEAK_AGE_RANGE[1]}"


# ---------------------------------------------------------------------------
# Current Ability
# ---------------------------------------------------------------------------

class TestCaWeights:
    @pytest.mark.parametrize("position", sorted(CA_WEIGHTS))
    def test_each_position_sums_to_one(self, position):
        assert sum(w.value for w in CA_WEIGHTS[position].values()) == pytest.approx(1.0)

    def test_every_position_has_weights(self):
        assert set(CA_WEIGHTS) == set(config.FPL_POSITIONS.values())

    def test_every_weight_has_a_reason(self):
        assert all(w.reason.strip() for by in CA_WEIGHTS.values() for w in by.values())

    def test_the_configured_weights_pass_the_check(self):
        check_weights()

    def test_weights_that_do_not_sum_to_one_are_rejected(self):
        bad = {"GK": {"goalkeeping": Weight(0.9, "too little")}}
        with pytest.raises(ValueError):
            check_weights(bad)

    def test_an_outfield_attribute_for_goalkeepers_is_rejected(self):
        bad = {"GK": {"goalkeeping": Weight(0.5, "x"), "finishing": Weight(0.5, "y")}}
        with pytest.raises(ValueError):
            check_weights(bad)


class TestCurrentAbility:
    @pytest.fixture
    def players(self) -> pd.DataFrame:
        frame = pd.DataFrame({"position": ["FWD", "DEF", "GK", "MID"]})
        for column in ATTRIBUTE_COLUMNS:
            frame[column] = pd.array([10, 12, pd.NA, 8], dtype="Int64")
        frame.loc[2, "goalkeeping"] = 15
        frame.loc[[0, 1, 3], "goalkeeping"] = pd.NA
        return frame

    def test_contributions_add_up_to_ca(self, players):
        result = current_ability(players)
        contributions = result[[f"contrib_{c}" for c in ATTRIBUTE_COLUMNS]].sum(axis=1)
        assert np.allclose(contributions, result["ca"])

    def test_equal_attributes_give_that_value(self, players):
        result = current_ability(players)
        assert result.loc[0, "ca"] == pytest.approx(10.0)  # forward with every attribute at 10
        assert result.loc[2, "ca"] == pytest.approx(15.0)  # goalkeeper: goalkeeping only

    def test_ca_stays_on_the_1_to_20_scale(self, players):
        result = current_ability(players)
        assert result["ca"].between(config.ATTRIBUTE_MIN, config.ATTRIBUTE_MAX).all()

    def test_a_missing_attribute_means_no_ca(self, players):
        players.loc[0, "finishing"] = pd.NA
        result = current_ability(players)
        assert math.isnan(result.loc[0, "ca"])
        assert result.loc[0, [f"contrib_{c}" for c in ATTRIBUTE_COLUMNS]].isna().all()


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------

def test_future_targets_are_the_peak_age_seasons():
    player = pd.DataFrame({"fpl_code": [1], "season_year": [2025], "age": [19.4], "age_completed": [19],
                           "position": ["MID"]})
    targets = future_targets(player, (24, 26))
    assert targets["season_year"].tolist() == [2030, 2031, 2032]
    assert targets["age"].tolist() == pytest.approx([24.4, 25.4, 26.4])
    assert targets["age_c"].tolist() == pytest.approx([3.4, 4.4, 5.4])


def toy_model() -> TrajectoryModel:
    """A small hand-made model with a random intercept and slope."""
    names = ["Intercept", "C(position)[T.MID]", "age_c", "I(age_c ** 2)"]
    return TrajectoryModel(
        fe_names=names,
        fe_params=np.array([0.3, -0.05, 0.04, -0.005]),
        fe_cov=np.diag([1e-4, 1e-4, 1e-5, 1e-7]),
        cov_re=np.array([[0.03, 0.001], [0.001, 0.001]]),
        scale=0.04,
        random_slope=True,
        positions=["DEF", "MID"],
    )


@pytest.mark.parametrize("history_seasons", [0, 1, 3])
def test_simulated_intervals_are_ordered(history_seasons):
    history = pd.DataFrame({
        "fpl_code": [7] * history_seasons,
        "age_c": [-2.0, -1.0, 0.0][:history_seasons],
        "position": ["MID"] * history_seasons,
        "minutes_share": [0.4, 0.5, 0.6][:history_seasons],
    })
    targets = pd.DataFrame({"fpl_code": [7, 7, 7], "age_c": [3.0, 4.0, 5.0], "position": ["MID"] * 3})
    result = project(toy_model(), history, targets, draws=500)
    assert (result["sim_lower"] <= result["sim_median"]).all()
    assert (result["sim_median"] <= result["sim_upper"]).all()
    assert result["history_seasons"].iloc[0] == history_seasons


# ---------------------------------------------------------------------------
# PA on the card: rank among this season's prospects
# ---------------------------------------------------------------------------

class TestProspectRank:
    def test_the_day_1_mid_rank_rule_and_mapping(self):
        # Percentiles (rank - 0.5) / 4 = 0.125, 0.375, 0.625, 0.875 -> 1 + floor(20 x pct) = 3, 8, 13, 18.
        percentiles, scores = prospect_rank(pd.Series([0.4, 0.1, 0.3, 0.2]))
        assert percentiles.tolist() == pytest.approx([0.875, 0.125, 0.625, 0.375])
        assert scores.tolist() == [18, 3, 13, 8]

    def test_missing_medians_stay_missing_and_do_not_count(self):
        percentiles, scores = prospect_rank(pd.Series([0.1, np.nan, 0.3]))
        assert percentiles.tolist()[0] == pytest.approx(0.25) and percentiles.tolist()[2] == pytest.approx(0.75)
        assert pd.isna(scores.iloc[1])

    def test_ties_share_a_value_and_order_is_kept(self):
        medians = pd.Series([0.0, 0.0, 0.05, 0.2, 0.6])
        _, scores = prospect_rank(medians)
        assert scores.iloc[0] == scores.iloc[1]
        assert scores.is_monotonic_increasing

    def test_the_label(self):
        assert PA_LABEL == "rank among 2025/26 U21 prospects"


class TestLowEvidence:
    def test_the_threshold(self):
        assert config.PA_LOW_EVIDENCE_MINUTES == 450
        assert low_evidence(pd.Series([0, 449, 450, 3000])).tolist() == [True, True, False, False]


def raw_projections() -> pd.DataFrame:
    """Four players as build_pa has them before finish_pa: one under 17 with no history, three projected."""
    return pd.DataFrame({
        "fpl_code": [1, 2, 3, 4],
        "position": ["MID", "MID", "DEF", "FWD"],
        "age_completed": [16, 19, 20, 18],
        "history_seasons": [0, 2, 3, 1],
        "pl_minutes_to_date": [120, 1500, 3000, 40],
        "re_intercept": [0.0, 0.05, -0.02, 0.0],
        "re_slope": [0.0, 0.001, 0.002, 0.0],
        "peak_population": [0.2, 0.2, 0.25, 0.1],
        "peak_predicted": [0.2, 0.26, 0.24, 0.1],
        "peak_median": [0.2, 0.26, 0.24, 0.1],
        "peak_lower": [-0.1, -0.05, 0.1, -0.2],
        "peak_upper": [0.5, 0.6, 0.45, 0.4],
    })


def references() -> dict[str, pd.DataFrame]:
    """Small peak-age reference samples per position, under the sensitivity names."""
    sample = pd.DataFrame({"position": ["MID"] * 4 + ["DEF"] * 4 + ["FWD"] * 2,
                           "peak_share": [0.0, 0.1, 0.3, 0.6, 0.0, 0.2, 0.4, 0.7, 0.0, 0.5]})
    return {name: sample for name in SENSITIVITY_REFERENCES}


class TestFinishPa:
    @pytest.fixture
    def card(self) -> pd.DataFrame:
        return finish_pa(raw_projections(), references(), factor=1.4).set_index("fpl_code")

    def test_under_17s_get_no_pa_and_the_right_status(self, card):
        assert card.loc[1, "pa_status"] == PA_STATUS_TOO_YOUNG == "insufficient data: debut before 17"
        missing = ["peak_median", "peak_lower_calibrated", "peak_upper_calibrated", "peak_median_display",
                   "peak_lower_calibrated_display", "pa", "pa_pct", "pa_display", "pa_range_text", "re_intercept",
                   "pa_all_players_median", "pa_young_debutants_median"]
        assert card.loc[1, missing].isna().all()

    def test_everyone_else_is_experimental_with_a_pa(self, card):
        projected = card.loc[[2, 3, 4]]
        assert (projected["pa_status"] == PA_STATUS_EXPERIMENTAL).all()
        assert projected[["peak_median", "pa", "pa_pct", "pa_display", "pa_range_text"]].notna().all().all()
        assert (projected["pa_label"] == PA_LABEL).all()

    def test_pa_ranks_projected_medians_across_positions(self, card):
        # Medians 0.26 (MID), 0.24 (DEF), 0.10 (FWD): percentiles 5/6, 3/6, 1/6 -> 17, 11, 4.
        assert card.loc[[2, 3, 4], "pa"].tolist() == [17, 11, 4]
        assert card.loc[[2, 3, 4], "pa_pct"].tolist() == pytest.approx([5 / 6, 3 / 6, 1 / 6])

    def test_low_evidence_players_are_greyed(self, card):
        assert card.loc[[2, 3, 4], "pa_low_evidence"].tolist() == [False, False, True]
        assert card.loc[[2, 3, 4], "pa_display"].tolist() == [PA_DISPLAY_NORMAL, PA_DISPLAY_NORMAL, PA_DISPLAY_GREYED]
        assert bool(card.loc[1, "pa_low_evidence"]) and pd.isna(card.loc[1, "pa_display"])

    def test_the_card_has_no_1_to_20_bounds(self, card):
        assert not {"pa_lower", "pa_upper", "pa_median", "pa_lower_pct", "pa_upper_pct"} & set(card.columns)

    def test_both_references_are_kept_as_sensitivity_columns(self, card):
        for name in SENSITIVITY_REFERENCES:
            for end in ("median", "lower", "upper"):
                column = f"pa_{name}_{end}"
                assert card.loc[[2, 3, 4], column].between(config.ATTRIBUTE_MIN, config.ATTRIBUTE_MAX).all()
            assert card.loc[[2, 3, 4], f"pa_{name}_lower"].le(card.loc[[2, 3, 4], f"pa_{name}_upper"]).all()

    def test_calibration_widens_around_the_median_and_keeps_the_order(self, card):
        projected = card.loc[[2, 3, 4]]
        assert (projected["peak_lower_calibrated"] <= projected["peak_median"]).all()
        assert (projected["peak_median"] <= projected["peak_upper_calibrated"]).all()
        assert (projected["peak_upper_calibrated"] - projected["peak_median"]).to_numpy() == pytest.approx(
            1.4 * (projected["peak_upper"] - projected["peak_median"]).to_numpy())
        assert projected[["peak_lower_calibrated_display", "peak_upper_calibrated_display"]].stack().between(0, 1).all()

    def test_the_card_shows_the_raw_range_by_default(self, card):
        assert config.CARD_INTERVAL == "raw"
        projected = card.loc[[2, 3, 4]]
        assert (projected["card_interval"] == "raw").all()
        assert projected["pa_range_text"].tolist() == projected["pa_range_text_raw"].tolist()

    def test_the_calibrated_range_is_kept_and_can_be_chosen(self):
        card = finish_pa(raw_projections(), references(), factor=1.4, card_interval="calibrated").iloc[1:]
        assert card["pa_range_text"].tolist() == card["pa_range_text_calibrated"].tolist()
        assert card["pa_range_text_raw"].notna().all()

    def test_the_rank_does_not_depend_on_the_interval_or_the_reference(self, card):
        other = finish_pa(raw_projections(), {name: references()["all_players"].assign(peak_share=0.9)
                                              for name in SENSITIVITY_REFERENCES},
                          factor=1.0, card_interval="calibrated").set_index("fpl_code")
        assert other["pa"].tolist() == card["pa"].tolist()

    def test_an_unknown_card_interval_is_an_error(self):
        with pytest.raises(ValueError):
            finish_pa(raw_projections(), references(), factor=1.0, card_interval="wide")
        with pytest.raises(ValueError):
            card_bounds("wide")

    def test_a_young_player_with_history_is_an_error(self):
        bad = raw_projections()
        bad.loc[0, "history_seasons"] = 1
        with pytest.raises(ValueError):
            finish_pa(bad, references(), factor=1.0)


# ---------------------------------------------------------------------------
# PA sensitivity references
# ---------------------------------------------------------------------------

class TestPeakReference:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        """Player 1 first seen at 18 (seasons at 24 and 25), player 2 at 23 (season at 24), player 3 unknown."""
        return pd.DataFrame({
            "fpl_code": [1, 1, 2, 3],
            "season": ["2024-25", "2025-26", "2025-26", "2025-26"],
            "season_year": [2024, 2025, 2025, 2025],
            "position": ["MID", "MID", "MID", "DEF"],
            "age_completed": pd.array([24, 25, 24, 24], dtype="Int64"),
            "minutes_share": [0.2, 0.4, 0.9, 0.5],
            "first_seen_age": pd.array([18, 18, 23, pd.NA], dtype="Int64"),
        })

    def test_every_player_by_default(self, panel):
        reference = peak_reference(panel)
        assert sorted(reference["fpl_code"]) == [1, 2, 3]
        assert reference.set_index("fpl_code").loc[1, "peak_share"] == pytest.approx(0.3)

    def test_young_debutants_only(self, panel):
        assert peak_reference(panel, first_seen_max_age=20)["fpl_code"].tolist() == [1]

    def test_the_sensitivity_references(self):
        assert SENSITIVITY_REFERENCES == {"all_players": None, "young_debutants": config.FIRST_SEEN_YOUNG_MAX_AGE}


def test_percentile_against_a_reference_uses_mid_ranks():
    reference = np.array([0.0, 0.0, 0.2, 0.4])
    result = percentile_against(pd.Series([0.0, 0.3, 1.0]), reference)
    assert result.tolist() == pytest.approx([0.25, 0.75, 1.0])


# ---------------------------------------------------------------------------
# Range text and persistence
# ---------------------------------------------------------------------------

class TestRangeText:
    def test_rounding_to_the_nearest_five_percent(self):
        assert [nearest_five_percent(x) for x in (0.0, 0.024, 0.025, 0.123, 0.551, 1.0)] == [0, 0, 5, 10, 55, 100]

    def test_the_sentence(self):
        text = range_text(pd.Series([0.2, 0.3, np.nan]), pd.Series([0.55, 0.31, 0.5]))
        assert text.iloc[0] == "Projected to play about 20% to 55% of league minutes at 24 to 26"
        assert text.iloc[1] == "Projected to play about 30% of league minutes at 24 to 26"
        assert pd.isna(text.iloc[2])

    def test_a_range_from_zero_admits_dropping_out_and_names_the_ages(self):
        text = range_text(pd.Series([0.0, 0.02, 0.025]), pd.Series([0.55, 0.551, 0.55]))
        expected = "Range: from not playing regularly in the PL to about 55% of league minutes at 24 to 26"
        assert text.iloc[0] == expected and text.iloc[1] == expected
        assert text.iloc[2] == "Projected to play about 5% to 55% of league minutes at 24 to 26"  # 2.5% rounds to 5%
        assert "0%" not in " ".join(text.iloc[:2].str.replace("100%", ""))

    def test_a_range_entirely_at_zero(self):
        text = range_text(pd.Series([0.0]), pd.Series([0.02])).iloc[0]
        assert text == "Range: not playing regularly in the PL at 24 to 26"

    def test_every_text_names_the_ages(self):
        lower = pd.Series([0.0, 0.0, 0.1, 0.3, 0.5])
        upper = pd.Series([0.0, 0.4, 0.1, 0.8, 1.0])
        assert range_text(lower, upper).str.endswith(AGES).all()


def test_persistence_is_the_mean_share_so_far_including_filled_zeros():
    panel = pd.DataFrame({
        "fpl_code": [1, 1, 1, 2],
        "season_year": [2024, 2025, 2026, 2025],
        "minutes_share": [0.4, 0.0, 0.9, 0.1],  # 2026 is after the last season and must be ignored
    })
    shares = persistence_shares(panel, pd.Series([1, 2]), last_season="2025-26")
    assert shares.loc[1] == pytest.approx(0.2) and shares.loc[2] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# The real output
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_pa() -> pd.DataFrame:
    if not config.PA_PROJECTIONS_PATH.exists():
        pytest.skip("PA not built (python -m src.ability)")
    return pd.read_parquet(config.PA_PROJECTIONS_PATH)


def test_real_pa_intervals_are_ordered(real_pa):
    pa = real_pa[real_pa["pa_status"] == PA_STATUS_EXPERIMENTAL]
    for lower, upper in (("peak_lower", "peak_upper"), ("peak_lower_calibrated", "peak_upper_calibrated")):
        for suffix in ("", "_display"):
            assert (pa[f"{lower}{suffix}"] <= pa[f"peak_median{suffix}"]).all()
            assert (pa[f"peak_median{suffix}"] <= pa[f"{upper}{suffix}"]).all()
    assert (pa["peak_lower_calibrated"] <= pa["peak_lower"]).all() and (pa["peak_upper_calibrated"] >= pa["peak_upper"]).all()
    assert pa[["peak_lower_calibrated_display", "peak_upper_calibrated_display"]].stack().between(0, 1).all()


def test_real_pa_status_and_persistence(real_pa):
    pa = real_pa
    assert {"pa_status", "persistence_share", "pa_range_text", "pa_low_evidence"} <= set(pa.columns)
    assert pa[["pa_status", "persistence_share", "pa_low_evidence"]].notna().all().all()
    assert set(pa["pa_status"]) <= {PA_STATUS_EXPERIMENTAL, PA_STATUS_TOO_YOUNG}
    assert pa["persistence_share"].between(0, 1).all()
    too_young = pa["age_completed"] < config.MODEL_AGE_RANGE[0]
    assert (too_young == (pa["pa_status"] == PA_STATUS_TOO_YOUNG)).all()
    pa_fields = ["peak_median", "peak_lower_calibrated", "peak_upper_calibrated", "pa", "pa_pct", "pa_display",
                 "pa_range_text"]
    assert pa.loc[too_young, pa_fields].isna().all().all()
    assert pa.loc[~too_young, pa_fields].notna().all().all()


def test_real_pa_is_the_rank_among_this_seasons_prospects(real_pa):
    pa = real_pa[real_pa["pa_status"] == PA_STATUS_EXPERIMENTAL]
    _, expected = prospect_rank(pa["peak_median_display"])
    assert pa["pa"].tolist() == expected.tolist()
    assert pa["pa"].between(config.ATTRIBUTE_MIN, config.ATTRIBUTE_MAX).all()
    assert (pa["pa_label"] == PA_LABEL).all()
    # A higher projected median never gets a lower PA, whatever the position.
    ordered = pa.sort_values("peak_median_display")
    assert ordered["pa"].is_monotonic_increasing


def test_real_low_evidence_flag_and_display(real_pa):
    pa = real_pa
    assert (pa["pa_low_evidence"] == (pa["pl_minutes_to_date"] < config.PA_LOW_EVIDENCE_MINUTES)).all()
    projected = pa[pa["pa_status"] == PA_STATUS_EXPERIMENTAL]
    greyed = projected["pa_display"] == PA_DISPLAY_GREYED
    assert (greyed == projected["pa_low_evidence"]).all()


def test_real_card_fields_and_sensitivity_columns(real_pa):
    pa = real_pa[real_pa["pa_status"] == PA_STATUS_EXPERIMENTAL]
    assert not {"pa_lower", "pa_upper", "pa_median"} & set(pa.columns)
    version = config.CARD_INTERVAL
    assert (pa["card_interval"] == version).all()
    assert pa["pa_range_text"].tolist() == pa[f"pa_range_text_{version}"].tolist()
    assert pa[["pa_range_text_raw", "pa_range_text_calibrated"]].notna().all().all()
    assert pa["pa_range_text"].str.endswith(AGES).all()
    assert not pa["pa_range_text"].str.contains("about 0%").any()
    for name in SENSITIVITY_REFERENCES:
        columns = [f"pa_{name}_median", f"pa_{name}_lower", f"pa_{name}_upper"]
        assert pa[columns].notna().all().all()
        assert pa[columns].stack().between(config.ATTRIBUTE_MIN, config.ATTRIBUTE_MAX).all()


# ---------------------------------------------------------------------------
# CA against PA
# ---------------------------------------------------------------------------

def toy_ca_pa() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six U21s: five projected with a CA, one projected without a CA, one under 17 with a CA but no PA."""
    pa = pd.DataFrame({
        "fpl_code": [1, 2, 3, 4, 5, 6, 7], "web_name": list("ABCDEFG"), "position": ["DEF"] * 3 + ["MID"] * 4,
        "pa": pd.array([20, 15, 10, 5, 1, 12, pd.NA], dtype="Int64"),
        "peak_median_display": [0.6, 0.4, 0.3, 0.1, 0.0, 0.35, np.nan],
        "pa_status": [PA_STATUS_EXPERIMENTAL] * 6 + [PA_STATUS_TOO_YOUNG],
    })
    ca = pd.DataFrame({"fpl_code": [1, 2, 3, 4, 5, 7], "ca": [14.0, 12.0, 11.0, 9.0, 8.0, 13.0]})
    return ca, pa


def test_ca_pa_players_need_both_a_ca_and_a_projection():
    ca, pa = toy_ca_pa()
    from src.ability import ca_pa_players

    assert sorted(ca_pa_players(ca, pa)["fpl_code"]) == [1, 2, 3, 4, 5]


def test_ca_pa_correlation_on_a_perfect_ordering():
    ca, pa = toy_ca_pa()
    from src.ability import ca_pa_correlation

    result = ca_pa_correlation(ca, pa)
    assert result["n"] == 5
    assert result["spearman_ca_pa"] == pytest.approx(1.0)
    assert result["spearman_ca_projected_median"] == pytest.approx(1.0)
    assert math.isnan(result["interval_95"][0])  # no interval at a correlation of exactly 1
    assert "Spearman" in result["correlation"] and result["positions"] == {"DEF": 3, "MID": 2}
    assert [p["fpl_code"] for p in result["player_list"]] == [1, 2, 3, 4, 5]


def test_ca_pa_correlation_on_a_reversed_ordering():
    ca, pa = toy_ca_pa()
    from src.ability import ca_pa_correlation

    reversed_ca = ca.assign(ca=-ca["ca"])
    assert ca_pa_correlation(reversed_ca, pa)["spearman_ca_pa"] == pytest.approx(-1.0)


def test_spearman_interval_contains_the_estimate_and_narrows_with_n():
    from src.ability import spearman_interval

    low, high = spearman_interval(0.2, 31)
    assert low < 0.2 < high
    # Fisher z with variance 1.06 / (n - 3): half-width 1.96 x sqrt(1.06 / 28) on the z scale.
    assert np.arctanh(high) - np.arctanh(0.2) == pytest.approx(1.959964 * math.sqrt(1.06 / 28))
    wide, narrow = spearman_interval(0.2, 20), spearman_interval(0.2, 200)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])
    assert all(math.isnan(v) for v in spearman_interval(0.2, 4))


def test_real_ca_pa_correlation_file_matches_the_tables(real_pa):
    import json

    if not config.CA_PA_CORRELATION_PATH.exists() or not config.CA_PATH.exists():
        pytest.skip("CA-to-PA correlation not built (python -m src.ability)")
    from src.ability import ca_pa_correlation

    saved = json.loads(config.CA_PA_CORRELATION_PATH.read_text(encoding="utf-8"))
    fresh = ca_pa_correlation(pd.read_parquet(config.CA_PATH), real_pa)
    assert saved["n"] == fresh["n"] == len(saved["player_list"])
    assert saved["spearman_ca_pa"] == pytest.approx(fresh["spearman_ca_pa"])
    assert saved["interval_95"][0] <= saved["spearman_ca_pa"] <= saved["interval_95"][1]
