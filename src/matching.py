"""Match FPL players to Transfermarkt players.

The unit is the FPL `code`. The audit showed it is the Opta player ID and stays
the same across seasons, so each player is matched once and the match covers
every one of their FPL seasons, 2016-17 to 2025-26. This also matches players
who left the league before 2024-25, which the backtest needs: otherwise only
players who stayed in the league would have ages.

Rules, strongest first:
1. Birth date and name. FPL has birth dates from 2024-25. A Transfermarkt
   player with the same date of birth whose name scores at least
   config.MATCH_NAME_SCORE_WITH_BIRTH_DATE.
2. Premier League minutes and name. For players with no rule-1 match: a
   Transfermarkt player whose name scores at least
   config.MATCH_NAME_SCORE_WITHOUT_BIRTH_DATE and whose Premier League minutes
   agree with FPL's, season by season (see compare_minutes).
3. Unique full name. For players with PL minutes that rules 1 and 2 leave
   unmatched: a full name that exactly one Transfermarkt player has, with a
   plausible age and no Transfermarkt minutes that contradict FPL.

Transfermarkt's appearances miss some player-seasons entirely (Conor
Gallagher's 2021-22, for example, has no rows at all). So the minutes check
only compares seasons where Transfermarkt has data, and the gaps are counted
separately. Rule 3 exists for players whose Premier League seasons are all
missing from Transfermarkt.

If several candidates pass the same rule, the club breaks the tie: the
candidate who played for the matching Transfermarkt club in more of the
player's seasons wins. Which Transfermarkt club each FPL team is gets learned
from the clear-cut matches, so there is no hand-made club table. Name, then
minutes, break any remaining tie, but only by a clear margin; otherwise the
player is left for review. Hand-made fixes in data/manual_matches.csv are
applied after the rules and win any conflict with them.

Outputs:
- data/processed/fpl_tm_matches.csv: one row per matched FPL player.
- data/processed/unmatched_report.csv: everyone else, with the reason and the
  closest Transfermarkt names, to help fill in data/manual_matches.csv.
- data/processed/matching_report.md: match rates and checks (also printed).

Run from the repository root:
    python -m src.matching
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from src import config
from src.clean import (
    fpl_birth_dates,
    fpl_player_seasons,
    fpl_team_names,
    latest_team_names,
    normalise_name,
    transfermarkt_players,
    transfermarkt_season_minutes,
)
from src.load_data import read_fpl_gameweeks, read_fpl_players, read_transfermarkt, utc_now

RULE_BIRTH_DATE = "birth date and name"
RULE_MINUTES = "PL minutes and name"
RULE_UNIQUE_NAME = "unique full name"
RULE_MANUAL = "manual"
# Higher means stronger evidence; used when two FPL codes claim one Transfermarkt player.
RULE_STRENGTH = {RULE_MANUAL: 4, RULE_BIRTH_DATE: 3, RULE_MINUTES: 2, RULE_UNIQUE_NAME: 1}
PREMIER_LEAGUE = "GB1"
# The FPL seasons as Transfermarkt season years (2016 means 2016-17).
SEASON_YEARS: tuple[int, ...] = tuple(config.season_start_year(s) for s in config.FPL_SEASONS)
# Score for a one-word name (a nickname such as "paqueta") that is one of the
# words of the other name ("lucas paqueta"): strong, but below an exact match.
ONE_WORD_INSIDE_SCORE: float = 90.0
SUGGESTIONS: int = 3  # closest Transfermarkt names listed for each unmatched player
SCORED_COLUMNS = ["fpl_code", "player_id", "name_score", "birth_date_equal", "minutes_agreement",
                  "seasons_both_played", "tm_missing_seasons", "tm_only_minutes", "club_seasons"]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class Inputs:
    """Everything the matching needs, read and tidied once."""

    fpl_seasons: pd.DataFrame  # one row per FPL player-season (see clean.fpl_player_seasons)
    fpl_team_names: dict[int, str]  # team_code -> club name (from 2020-21 on)
    tm_players: pd.DataFrame  # see clean.transfermarkt_players
    tm_minutes: pd.DataFrame  # Premier League minutes per Transfermarkt player and season
    tm_club_names: dict[int, str]  # Transfermarkt club_id -> name


def load_inputs() -> Inputs:
    """Read the raw files and tidy them for matching."""
    players = {season: read_fpl_players(season) for season in config.FPL_SEASONS}
    gameweeks = {season: read_fpl_gameweeks(season) for season in config.FPL_SEASONS}
    fpl_seasons = fpl_player_seasons(players)
    fpl_seasons["season_year"] = fpl_seasons["season"].map(config.season_start_year)
    tm_minutes = transfermarkt_season_minutes(
        read_transfermarkt("appearances"), read_transfermarkt("games"), [PREMIER_LEAGUE]
    )
    clubs = read_transfermarkt("clubs")
    return Inputs(
        fpl_seasons=fpl_seasons,
        fpl_team_names=latest_team_names(fpl_team_names(players, gameweeks)),
        tm_players=transfermarkt_players(read_transfermarkt("players")),
        tm_minutes=tm_minutes[tm_minutes["season"].isin(SEASON_YEARS)],
        tm_club_names=dict(zip(clubs["club_id"], clubs["name"])),
    )


def fpl_player_table(fpl_seasons: pd.DataFrame, team_names: dict[int, str]) -> pd.DataFrame:
    """One row per FPL code: latest names and club, birth date, seasons listed and PL minutes."""
    ordered = fpl_seasons.sort_values("season")
    latest = ordered.drop_duplicates("code", keep="last").set_index("code")
    by_code = ordered.groupby("code")
    table = pd.DataFrame(
        {
            "fpl_name": latest["first_name"] + " " + latest["second_name"],
            "fpl_web_name": latest["web_name"],
            "fpl_position": latest["position"],
            "fpl_birth_date": fpl_birth_dates(fpl_seasons),
            "fpl_seasons": by_code["season"].agg(config.describe_seasons),
            "fpl_pl_minutes": by_code["minutes"].sum(),
            "latest_fpl_season": latest["season"],
            "latest_fpl_club": latest["team_code"].map(lambda code: team_names.get(code, f"team_code {code}")),
        }
    )
    return table.rename_axis("fpl_code").reset_index()


# ---------------------------------------------------------------------------
# Evidence about each player
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """What each source says about every player, held in dictionaries for quick lookups."""

    fpl_names: dict[int, list[str]]
    tm_names: dict[int, list[str]]
    fpl_minutes: dict[int, dict[int, int]]  # code -> season year -> PL minutes
    fpl_teams: dict[int, dict[int, int]]  # code -> season year -> team_code
    tm_minutes: dict[int, dict[int, int]]  # player_id -> season year -> PL minutes
    tm_clubs: dict[int, dict[int, tuple[int, ...]]]  # player_id -> season year -> club ids
    fpl_birth: dict[int, pd.Timestamp]
    tm_birth: dict[int, pd.Timestamp]


def fpl_name_variants(fpl_seasons: pd.DataFrame) -> dict[int, list[str]]:
    """Every normalised version of each FPL player's name, across all seasons.

    Uses the full name, web_name and known_name, plus the first word of the
    first name with the last word of the surname ("bruno fernandes" from
    "Bruno Miguel Borges Fernandes").
    """
    variants: dict[int, set[str]] = {}
    columns = ["code", "first_name", "second_name", "web_name", "known_name"]
    for code, first, second, web, known in fpl_seasons[columns].itertuples(index=False):
        first, second = normalise_name(first), normalise_name(second)
        names = variants.setdefault(int(code), set())
        names.update(n for n in (f"{first} {second}".strip(), normalise_name(web), normalise_name(known)) if n)
        if first and second:
            names.add(f"{first.split()[0]} {second.split()[-1]}")
    return {code: sorted(names) for code, names in variants.items()}


def tm_name_variants(tm_players: pd.DataFrame) -> dict[int, list[str]]:
    """Normalised versions of each Transfermarkt name: the display name, and first plus last name."""
    variants = {}
    columns = ["player_id", "name", "first_name", "last_name"]
    for player_id, name, first, last in tm_players[columns].itertuples(index=False):
        full = " ".join(part for part in (normalise_name(first), normalise_name(last)) if part)
        variants[int(player_id)] = sorted({n for n in (normalise_name(name), full) if n})
    return variants


def build_evidence(inputs: Inputs, fpl_table: pd.DataFrame) -> Evidence:
    """Gather names, minutes, clubs and birth dates from both sources."""
    fpl_minutes: dict[int, dict[int, int]] = {}
    fpl_teams: dict[int, dict[int, int]] = {}
    for code, year, minutes, team in inputs.fpl_seasons[
        ["code", "season_year", "minutes", "team_code"]
    ].itertuples(index=False):
        fpl_minutes.setdefault(int(code), {})[int(year)] = int(minutes)
        fpl_teams.setdefault(int(code), {})[int(year)] = int(team)
    tm_minutes: dict[int, dict[int, int]] = {}
    tm_clubs: dict[int, dict[int, tuple[int, ...]]] = {}
    for player_id, year, minutes, clubs in inputs.tm_minutes[
        ["player_id", "season", "minutes", "club_ids"]
    ].itertuples(index=False):
        tm_minutes.setdefault(int(player_id), {})[int(year)] = int(minutes)
        tm_clubs.setdefault(int(player_id), {})[int(year)] = clubs
    dated_fpl = fpl_table.dropna(subset=["fpl_birth_date"])
    dated_tm = inputs.tm_players.dropna(subset=["birth_date"])
    return Evidence(
        fpl_names=fpl_name_variants(inputs.fpl_seasons),
        tm_names=tm_name_variants(inputs.tm_players),
        fpl_minutes=fpl_minutes,
        fpl_teams=fpl_teams,
        tm_minutes=tm_minutes,
        tm_clubs=tm_clubs,
        fpl_birth=dict(zip(dated_fpl["fpl_code"], dated_fpl["fpl_birth_date"])),
        tm_birth=dict(zip(dated_tm["player_id"], dated_tm["birth_date"])),
    )


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------

def word_score(a: str, b: str) -> float:
    """Similarity of two normalised names, from 0 to 100.

    Two names of two or more words are compared as sets of words, so a shorter
    name inside a longer one scores 100 ("bruno fernandes" and "bruno miguel
    borges fernandes"). A one-word name scores ONE_WORD_INSIDE_SCORE if it is
    one of the other name's words, and is otherwise compared letter by letter,
    so a nickname cannot score 100 just by appearing inside a longer name.
    """
    if a == b:
        return 100.0
    if " " in a and " " in b:
        return fuzz.token_set_ratio(a, b)
    single, other = (a, b) if " " not in a else (b, a)
    if len(single) >= 3 and single in other.split():
        return ONE_WORD_INSIDE_SCORE
    return fuzz.ratio(a, b)


def name_score(fpl_names: Iterable[str], tm_names: Iterable[str]) -> float:
    """Best word_score between any FPL version and any Transfermarkt version of a name."""
    return max((word_score(a, b) for a in fpl_names for b in tm_names), default=0.0)


@dataclass(frozen=True)
class MinutesComparison:
    """How a Transfermarkt player's Premier League minutes line up with an FPL player's."""

    agreement: float  # 0 to 1 over the seasons where Transfermarkt has minutes; NaN if it has none
    seasons_both_played: int  # seasons where both sources have minutes
    tm_missing_seasons: int  # seasons where FPL has minutes but Transfermarkt has no rows
    tm_only_minutes: int  # Transfermarkt minutes in seasons where FPL has none


def compare_minutes(fpl: dict[int, int], tm: dict[int, int]) -> MinutesComparison:
    """Compare two Premier League minutes records, season by season, 2016-17 to 2025-26.

    The agreement is the sum of the smaller of the two numbers divided by the
    sum of the larger, over the seasons where Transfermarkt has minutes: 1 means
    identical. Seasons where only FPL has minutes are left out of it, because
    Transfermarkt misses some player-seasons entirely; they are counted as gaps
    instead. Transfermarkt minutes in a season where FPL has none do count
    against the agreement, since FPL's Premier League minutes are complete.
    """
    with_tm = [y for y in SEASON_YEARS if tm.get(y, 0) > 0]
    larger = sum(max(fpl.get(y, 0), tm[y]) for y in with_tm)
    smaller = sum(min(fpl.get(y, 0), tm[y]) for y in with_tm)
    return MinutesComparison(
        agreement=smaller / larger if larger else float("nan"),
        seasons_both_played=sum(1 for y in with_tm if fpl.get(y, 0) > 0),
        tm_missing_seasons=sum(1 for y in SEASON_YEARS if fpl.get(y, 0) > 0 and tm.get(y, 0) == 0),
        tm_only_minutes=sum(tm[y] for y in with_tm if fpl.get(y, 0) == 0),
    )


def club_seasons(
    fpl_teams: dict[int, int],
    tm_clubs: dict[int, tuple[int, ...]],
    club_map: dict[tuple[int, int], int],
) -> int:
    """Seasons in which the candidate played for the Transfermarkt club matching the FPL team."""
    return sum(1 for year, team in fpl_teams.items() if club_map.get((year, team)) in tm_clubs.get(year, ()))


def score_pairs(
    pairs: Iterable[tuple[int, int]],
    evidence: Evidence,
    club_map: dict[tuple[int, int], int],
) -> pd.DataFrame:
    """Measure the evidence for each candidate pair of FPL code and Transfermarkt player."""
    rows = []
    for code, player_id in pairs:
        minutes = compare_minutes(evidence.fpl_minutes.get(code, {}), evidence.tm_minutes.get(player_id, {}))
        fpl_born = evidence.fpl_birth.get(code)
        rows.append(
            {
                "fpl_code": code,
                "player_id": player_id,
                "name_score": round(
                    name_score(evidence.fpl_names.get(code, []), evidence.tm_names.get(player_id, [])), 1
                ),
                "birth_date_equal": fpl_born is not None and fpl_born == evidence.tm_birth.get(player_id),
                "minutes_agreement": round(minutes.agreement, 3),
                "seasons_both_played": minutes.seasons_both_played,
                "tm_missing_seasons": minutes.tm_missing_seasons,
                "tm_only_minutes": minutes.tm_only_minutes,
                "club_seasons": club_seasons(
                    evidence.fpl_teams.get(code, {}), evidence.tm_clubs.get(player_id, {}), club_map
                ),
            }
        )
    return pd.DataFrame(rows, columns=SCORED_COLUMNS)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def birth_date_pairs(evidence: Evidence) -> set[tuple[int, int]]:
    """FPL codes and Transfermarkt players that share a birth date (rule-1 candidates)."""
    by_date: dict[pd.Timestamp, list[int]] = {}
    for player_id, born in evidence.tm_birth.items():
        by_date.setdefault(born, []).append(player_id)
    return {(code, pid) for code, born in evidence.fpl_birth.items() for pid in by_date.get(born, [])}


def flatten(owners: Iterable[int], variants: dict[int, list[str]]) -> tuple[np.ndarray, list[str]]:
    """Lay out every name variant in one list, with an array saying whose variant each is."""
    owner_list: list[int] = []
    texts: list[str] = []
    for owner in owners:
        for text in variants.get(int(owner), []):
            owner_list.append(int(owner))
            texts.append(text)
    return np.array(owner_list, dtype=np.int64), texts


def minutes_pairs(inputs: Inputs, evidence: Evidence) -> set[tuple[int, int]]:
    """Rule-2 candidates: players with PL minutes in the same season and loosely similar names.

    Comparing every FPL player with every Transfermarkt player would be slow and
    noisy, so this compares players who both have Premier League minutes in the
    same season, and keeps pairs whose names pass a loose shortlist score.
    """
    pairs: set[tuple[int, int]] = set()
    played = inputs.fpl_seasons[inputs.fpl_seasons["minutes"] > 0]
    for year, rows in played.groupby("season_year"):
        player_ids = inputs.tm_minutes.loc[inputs.tm_minutes["season"] == year, "player_id"].unique()
        fpl_owner, fpl_text = flatten(rows["code"].unique(), evidence.fpl_names)
        tm_owner, tm_text = flatten(player_ids, evidence.tm_names)
        if not fpl_text or not tm_text:
            continue
        scores = process.cdist(
            fpl_text, tm_text, scorer=fuzz.token_set_ratio,
            score_cutoff=config.MATCH_SHORTLIST_SCORE, dtype=np.uint8, workers=-1,
        )
        hit_rows, hit_columns = np.nonzero(scores)
        pairs.update(zip(fpl_owner[hit_rows].tolist(), tm_owner[hit_columns].tolist()))
    return pairs


def apply_rules(scored: pd.DataFrame) -> pd.DataFrame:
    """Label each candidate pair with rule 1 or rule 2 if it passes ("" if not)."""
    rule1 = scored["birth_date_equal"] & (scored["name_score"] >= config.MATCH_NAME_SCORE_WITH_BIRTH_DATE)
    rule2 = (
        (scored["name_score"] >= config.MATCH_NAME_SCORE_WITHOUT_BIRTH_DATE)
        & (scored["seasons_both_played"] >= 1)
        & (scored["minutes_agreement"] >= config.MATCH_MIN_MINUTES_AGREEMENT)
    )
    return scored.assign(rule=np.select([rule1, rule2], [RULE_BIRTH_DATE, RULE_MINUTES], default=""))


# ---------------------------------------------------------------------------
# Choosing matches
# ---------------------------------------------------------------------------

def pick_one(pool: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Choose between candidates that pass the same rule.

    Returns the choice and a note, or None and the reason no choice was made.
    Club decides first, then name, then minutes; name and minutes only decide
    if the margin is clear.
    """
    ranked = pool.sort_values(["club_seasons", "name_score", "minutes_agreement"], ascending=False)
    best = ranked.iloc[0]
    if len(ranked) == 1:
        return best, ""
    second = ranked.iloc[1]
    if best["club_seasons"] > second["club_seasons"]:
        return best, f"tie between {len(ranked)} candidates broken by club"
    if best["name_score"] >= second["name_score"] + config.MATCH_DECISIVE_NAME_MARGIN:
        return best, f"tie between {len(ranked)} candidates broken by name"
    second_agreement = 0.0 if pd.isna(second["minutes_agreement"]) else second["minutes_agreement"]
    if best["minutes_agreement"] >= second_agreement + config.MATCH_DECISIVE_MINUTES_MARGIN:
        return best, f"tie between {len(ranked)} candidates broken by minutes"
    return None, f"ambiguous: {len(ranked)} candidates pass rule '{best['rule']}' and nothing separates them"


def pick_matches(scored: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    """Pick at most one Transfermarkt player per FPL code, using the strongest rule that applies."""
    chosen, reasons = [], {}
    for code, group in scored[scored["rule"] != ""].groupby("fpl_code"):
        pool = group[group["rule"] == RULE_BIRTH_DATE]
        best, note = pick_one(pool if not pool.empty else group)
        if best is None:
            reasons[int(code)] = note
        else:
            chosen.append({**best.to_dict(), "note": note})
    return pd.DataFrame(chosen, columns=[*scored.columns, "note"]), reasons


def learn_club_map(confident: pd.DataFrame, evidence: Evidence) -> dict[tuple[int, int], int]:
    """Which Transfermarkt club each FPL team is, season by season.

    For every FPL season and team_code, the Transfermarkt club that most of the
    team's confidently matched players played Premier League games for.
    """
    votes: dict[tuple[int, int], Counter[int]] = {}
    for code, player_id in zip(confident["fpl_code"], confident["player_id"]):
        clubs_by_year = evidence.tm_clubs.get(int(player_id), {})
        for year, team in evidence.fpl_teams.get(int(code), {}).items():
            for club in clubs_by_year.get(year, ()):
                votes.setdefault((year, team), Counter())[club] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in votes.items()}


def plausible_age(code: int, player_id: int, evidence: Evidence) -> bool:
    """Whether the Transfermarkt birth date fits the FPL player's first season with minutes."""
    born = evidence.tm_birth.get(player_id)
    played = [year for year, minutes in evidence.fpl_minutes.get(code, {}).items() if minutes > 0]
    if born is None or not played:
        return False
    age = (pd.Timestamp(min(played), 9, 1) - born).days / 365.25
    low, high = config.MATCH_PLAUSIBLE_AGE
    return low <= age <= high


def unique_name_matches(
    codes: list[int],
    evidence: Evidence,
    taken: set[int],
    club_map: dict[tuple[int, int], int],
) -> pd.DataFrame:
    """Rule 3: FPL players whose full name belongs to exactly one Transfermarkt player.

    Only names of two or more words count, and no other Transfermarkt player may
    score config.MATCH_UNIQUE_NAME_RUNNER_UP or more. The match must also have a
    plausible age and no Transfermarkt Premier League minutes in seasons where
    FPL has none. Transfermarkt players already matched are not reused.
    """
    multi_word = {code: [n for n in evidence.fpl_names.get(code, []) if " " in n] for code in codes}
    fpl_owner, fpl_text = flatten(codes, multi_word)
    tm_multi_word = {pid: [n for n in names if " " in n] for pid, names in evidence.tm_names.items()}
    tm_owner, tm_text = flatten(tm_multi_word, tm_multi_word)
    if not fpl_text or not tm_text:
        return pd.DataFrame(columns=[*SCORED_COLUMNS, "rule", "note"])
    scores = process.cdist(
        fpl_text, tm_text, scorer=fuzz.token_set_ratio,
        score_cutoff=config.MATCH_UNIQUE_NAME_RUNNER_UP, dtype=np.uint8, workers=-1,
    )
    hit_rows, hit_columns = np.nonzero(scores)
    hits = pd.DataFrame(
        {"fpl_code": fpl_owner[hit_rows], "player_id": tm_owner[hit_columns], "score": scores[hit_rows, hit_columns]}
    ).groupby(["fpl_code", "player_id"], as_index=False)["score"].max()
    pairs = []
    for code, group in hits.groupby("fpl_code"):
        if len(group) == 1 and group["score"].iat[0] == 100:
            player_id = int(group["player_id"].iat[0])
            if player_id not in taken and plausible_age(int(code), player_id, evidence):
                pairs.append((int(code), player_id))
    scored = score_pairs(pairs, evidence, club_map)
    scored = scored[scored["tm_only_minutes"] == 0]
    return scored.assign(rule=RULE_UNIQUE_NAME, note="only Transfermarkt player with this full name")


def resolve_shared_players(matches: pd.DataFrame, fpl_seasons: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, str]]:
    """Deal with a Transfermarkt player picked for more than one FPL code.

    Codes listed in the same season must be different people, so only the
    best-supported match is kept. Codes that never overlap are probably one
    player whose FPL code changed (the audit found such a case), so all are
    kept and flagged.
    """
    seasons_by_code = fpl_seasons.groupby("code")["season"].agg(set).to_dict()
    kept, dropped = [], {}
    for _, group in matches.groupby("player_id"):
        if len(group) == 1:
            kept.append(group)
            continue
        codes = [int(code) for code in group["fpl_code"]]
        if any(seasons_by_code[a] & seasons_by_code[b] for a, b in combinations(codes, 2)):
            ranked = (
                group.assign(_strength=group["rule"].map(RULE_STRENGTH))
                .sort_values(["_strength", "name_score", "minutes_agreement"], ascending=False)
                .drop(columns="_strength")
            )
            winner = int(ranked["fpl_code"].iat[0])
            kept.append(ranked.iloc[[0]])
            for code in ranked["fpl_code"].iloc[1:]:
                dropped[int(code)] = f"conflict: the same Transfermarkt player is a better match for FPL code {winner}"
        else:
            notes = []
            for code, note in zip(codes, group["note"]):
                others = ", ".join(str(c) for c in codes if c != code)
                extra = f"same Transfermarkt player as FPL code {others}: probably one player under several codes"
                notes.append(f"{note}; {extra}" if note else extra)
            kept.append(group.assign(note=notes))
    combined = pd.concat(kept, ignore_index=True) if kept else matches.iloc[0:0]
    return combined, dropped


def read_manual_matches() -> pd.DataFrame:
    """Read data/manual_matches.csv: one row per FPL code; a blank Transfermarkt id means 'no match'."""
    manual = pd.read_csv(
        config.MANUAL_MATCHES_PATH, dtype={"fpl_code": "Int64", "transfermarkt_player_id": "Int64"}
    )
    repeated = manual.loc[manual["fpl_code"].duplicated(), "fpl_code"]
    if not repeated.empty:
        raise ValueError(f"data/manual_matches.csv lists these FPL codes more than once: {repeated.tolist()}")
    return manual


def apply_manual_matches(
    matches: pd.DataFrame,
    reasons: dict[int, str],
    manual: pd.DataFrame,
    known_codes: set[int],
    known_players: set[int],
    evidence: Evidence,
    club_map: dict[tuple[int, int], int],
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Apply the hand-made fixes: a Transfermarkt id forces a match, a blank id forces no match."""
    bad_codes = [int(c) for c in manual["fpl_code"].dropna() if int(c) not in known_codes]
    bad_ids = [int(i) for i in manual["transfermarkt_player_id"].dropna() if int(i) not in known_players]
    if bad_codes or bad_ids:
        raise ValueError(
            f"data/manual_matches.csv has unknown FPL codes {bad_codes} or Transfermarkt ids {bad_ids}"
        )
    forced = []
    notes = manual["note"].fillna("").astype(str)
    for code, player_id, note in zip(manual["fpl_code"], manual["transfermarkt_player_id"], notes):
        code = int(code)
        matches = matches[matches["fpl_code"] != code]
        reasons.pop(code, None)
        if pd.isna(player_id):
            reasons[code] = "manual: no match" + (f" ({note})" if note else "")
        else:
            forced.append((code, int(player_id), note or "from data/manual_matches.csv"))
    if forced:
        scored = score_pairs([(code, pid) for code, pid, _ in forced], evidence, club_map)
        scored = scored.assign(rule=RULE_MANUAL, note=[note for _, _, note in forced])
        matches = pd.concat([matches, scored], ignore_index=True)
    return matches, reasons


def no_match_reason(birth_date: object, minutes: int) -> str:
    """Why a player with no passing candidate could not be matched."""
    if pd.isna(birth_date) and minutes == 0:
        return "nothing to match on: no FPL birth date and no PL minutes"
    return "no candidate passes any rule"


# ---------------------------------------------------------------------------
# Output tables
# ---------------------------------------------------------------------------

def latest_pl_club(clubs_by_year: dict[int, tuple[int, ...]], club_names: dict[int, str]) -> str:
    """Transfermarkt club(s) in the player's latest Premier League season, e.g. "Arsenal FC (2025-26)"."""
    if not clubs_by_year:
        return ""
    year = max(clubs_by_year)
    names = " / ".join(club_names.get(club, f"club {club}") for club in clubs_by_year[year])
    return f"{names} ({config.season_label(year)})"


def build_match_table(matches: pd.DataFrame, fpl_table: pd.DataFrame, inputs: Inputs, evidence: Evidence) -> pd.DataFrame:
    """One row per matched FPL player, with both sources' details side by side."""
    tm = inputs.tm_players.set_index("player_id")
    table = matches.merge(fpl_table, on="fpl_code", how="left")
    table["tm_name"] = table["player_id"].map(tm["name"])
    table["tm_birth_date"] = table["player_id"].map(tm["birth_date"])
    table["tm_position"] = table["player_id"].map(tm["position"])
    table["tm_sub_position"] = table["player_id"].map(tm["sub_position"])
    table["tm_latest_pl_club"] = [
        latest_pl_club(evidence.tm_clubs.get(int(pid), {}), inputs.tm_club_names) for pid in table["player_id"]
    ]
    both = table["fpl_birth_date"].notna() & table["tm_birth_date"].notna()
    same = table["fpl_birth_date"] == table["tm_birth_date"]
    table["birth_dates_agree"] = np.where(both, np.where(same, "yes", "no"), "")
    columns = [
        "fpl_code", "tm_player_id", "rule", "fpl_name", "tm_name", "fpl_web_name", "fpl_position",
        "tm_position", "tm_sub_position", "fpl_birth_date", "tm_birth_date", "birth_dates_agree",
        "fpl_seasons", "fpl_pl_minutes", "latest_fpl_club", "tm_latest_pl_club", "name_score",
        "minutes_agreement", "seasons_both_played", "tm_missing_seasons", "club_seasons", "note",
    ]
    return table.rename(columns={"player_id": "tm_player_id"})[columns].sort_values("fpl_code", ignore_index=True)


def describe_candidates(scored: pd.DataFrame, tm_players: pd.DataFrame) -> dict[int, str]:
    """The three strongest automatic candidates for each FPL code, as readable text."""
    names = tm_players.set_index("player_id")["name"]
    births = tm_players.set_index("player_id")["birth_date"]
    text: dict[int, str] = {}
    ordered = scored.sort_values(["fpl_code", "name_score", "minutes_agreement"], ascending=[True, False, False])
    for code, group in ordered.groupby("fpl_code"):
        parts = []
        for row in group.head(3).itertuples(index=False):
            born = births.get(row.player_id)
            born_text = f"{born:%Y-%m-%d}" if pd.notna(born) else "birth date unknown"
            agreement = "no season to compare" if pd.isna(row.minutes_agreement) else f"{row.minutes_agreement:.2f}"
            parts.append(
                f"{names.get(row.player_id, '?')} [{row.player_id}], born {born_text}, "
                f"name {row.name_score:.0f}, minutes {agreement}"
            )
        text[int(code)] = "; ".join(parts)
    return text


def name_suggestions(report: pd.DataFrame, tm_players: pd.DataFrame) -> list[str]:
    """The closest Transfermarkt names for each unmatched player, searched over every player."""
    if report.empty:
        return []
    choices = [normalise_name(name) for name in tm_players["name"]]
    queries = [normalise_name(name) for name in report["fpl_name"]]
    web_queries = [normalise_name(name) for name in report["fpl_web_name"]]
    scores = np.maximum(
        process.cdist(queries, choices, scorer=fuzz.token_sort_ratio, dtype=np.uint8, workers=-1),
        process.cdist(web_queries, choices, scorer=fuzz.ratio, dtype=np.uint8, workers=-1),
    )
    top = np.argsort(-scores.astype(np.int16), axis=1, kind="stable")[:, :SUGGESTIONS]
    ids, names = tm_players["player_id"].to_numpy(), tm_players["name"].to_numpy()
    births, clubs = tm_players["birth_date"].to_numpy(), tm_players["current_club_name"].to_numpy()
    suggestions = []
    for row_number, columns in enumerate(top):
        parts = []
        for column in columns:
            born = pd.Timestamp(births[column])
            born_text = f"{born:%Y-%m-%d}" if pd.notna(born) else "birth date unknown"
            parts.append(
                f"{names[column]} [{ids[column]}], born {born_text}, {clubs[column]}, "
                f"score {scores[row_number, column]}"
            )
        suggestions.append("; ".join(parts))
    return suggestions


def build_unmatched_report(
    reasons: dict[int, str], fpl_table: pd.DataFrame, scored: pd.DataFrame, inputs: Inputs
) -> pd.DataFrame:
    """Every unmatched FPL player, most PL minutes first, with clues for a manual fix."""
    report = fpl_table[fpl_table["fpl_code"].isin(reasons)].copy()
    report["reason"] = report["fpl_code"].map(reasons)
    report["best_candidates"] = report["fpl_code"].map(describe_candidates(scored, inputs.tm_players)).fillna("")
    report = report.sort_values(["fpl_pl_minutes", "fpl_code"], ascending=[False, True], ignore_index=True)
    report["name_suggestions"] = name_suggestions(report, inputs.tm_players)
    return report


# ---------------------------------------------------------------------------
# Match rates and checks
# ---------------------------------------------------------------------------

def match_rates(fpl_table: pd.DataFrame, fpl_seasons: pd.DataFrame, matched: set[int]) -> pd.DataFrame:
    """Share of FPL players matched, for the groups that matter most."""
    dated_seasons = sorted(fpl_seasons.loc[fpl_seasons["birth_date"].notna(), "season"].unique())
    recent = fpl_seasons[fpl_seasons["season"].isin(dated_seasons)]
    recent_codes = set(recent["code"])
    recent_played = set(recent.loc[recent["minutes"] > 0, "code"])
    played = set(fpl_table.loc[fpl_table["fpl_pl_minutes"] > 0, "fpl_code"])
    seasons_text = " or ".join(dated_seasons)
    groups = [
        (f"Listed in {seasons_text}", recent_codes),
        (f"Played in {seasons_text}", recent_played),
        ("Played, but only listed before those seasons", played - recent_codes),
        ("Played in any season, 2016-17 to 2025-26", played),
        ("Everyone listed, 2016-17 to 2025-26", set(fpl_table["fpl_code"])),
    ]
    return pd.DataFrame(
        [
            {
                "group": label,
                "players": len(codes),
                "matched": len(codes & matched),
                "match rate": f"{100 * len(codes & matched) / len(codes):.1f}%" if codes else "n/a",
            }
            for label, codes in groups
        ]
    )


def season_rates(fpl_seasons: pd.DataFrame, matched: set[int]) -> pd.DataFrame:
    """Per season: share of players with PL minutes who are matched, and share of minutes covered."""
    played = fpl_seasons[fpl_seasons["minutes"] > 0].assign(matched=lambda f: f["code"].isin(matched))
    rows = []
    for season, group in played.groupby("season"):
        covered = group.loc[group["matched"], "minutes"].sum() / group["minutes"].sum()
        rows.append(
            {
                "season": season,
                "players with minutes": len(group),
                "matched": int(group["matched"].sum()),
                "match rate": f"{100 * group['matched'].mean():.1f}%",
                "share of PL minutes covered": f"{100 * covered:.1f}%",
            }
        )
    return pd.DataFrame(rows)


def transfermarkt_gaps(matches: pd.DataFrame, evidence: Evidence) -> pd.DataFrame:
    """Per season: matched players with FPL minutes whose Transfermarkt record has no PL rows.

    FPL's Premier League minutes are complete, so this measures how many
    player-seasons Transfermarkt's appearances are missing.
    """
    rows = []
    pairs = list(zip(matches["fpl_code"].astype(int), matches["tm_player_id"].astype(int)))
    for year, season in zip(SEASON_YEARS, config.FPL_SEASONS):
        played = [(code, pid, evidence.fpl_minutes.get(code, {}).get(year, 0)) for code, pid in pairs]
        played = [(code, pid, minutes) for code, pid, minutes in played if minutes > 0]
        missing = [(code, pid, minutes) for code, pid, minutes in played if evidence.tm_minutes.get(pid, {}).get(year, 0) == 0]
        total_minutes = sum(minutes for _, _, minutes in played)
        rows.append(
            {
                "season": season,
                "matched players with minutes": len(played),
                "missing from Transfermarkt": len(missing),
                "share of players": f"{100 * len(missing) / len(played):.1f}%" if played else "n/a",
                "share of minutes": (
                    f"{100 * sum(m for _, _, m in missing) / total_minutes:.1f}%" if total_minutes else "n/a"
                ),
            }
        )
    return pd.DataFrame(rows)


def club_map_table(
    club_map: dict[tuple[int, int], int], team_names: dict[int, str], club_names: dict[int, str]
) -> pd.DataFrame:
    """One row per FPL team_code: the Transfermarkt club(s) it was mapped to, and in which seasons."""
    rows: dict[int, dict[int, list[int]]] = {}
    for (year, team), club in club_map.items():
        rows.setdefault(team, {}).setdefault(club, []).append(year)
    return pd.DataFrame(
        [
            {
                "team_code": team,
                "FPL club": team_names.get(team, f"team_code {team}"),
                "Transfermarkt club": "; ".join(
                    f"{club_names.get(club, club)} ({config.describe_seasons(config.season_label(y) for y in years)})"
                    for club, years in clubs.items()
                ),
                "_clubs": len(clubs),
            }
            for team, clubs in sorted(rows.items())
        ]
    )


@dataclass
class MatchingResult:
    """Everything the matching produces."""

    matches: pd.DataFrame
    unmatched: pd.DataFrame
    rates: pd.DataFrame
    season_rates: pd.DataFrame
    gaps: pd.DataFrame
    club_map: pd.DataFrame
    manual_rows: int


def run_matching(inputs: Inputs) -> MatchingResult:
    """Run the three rules, the club tie-break, the conflict checks and the manual fixes."""
    fpl_table = fpl_player_table(inputs.fpl_seasons, inputs.fpl_team_names)
    evidence = build_evidence(inputs, fpl_table)
    pairs = birth_date_pairs(evidence) | minutes_pairs(inputs, evidence)

    # First pass without clubs, to find clear-cut matches to learn the club map from.
    scored = apply_rules(score_pairs(pairs, evidence, club_map={}))
    first_pass, _ = pick_matches(scored)
    club_map = learn_club_map(first_pass[first_pass["note"] == ""], evidence)
    scored["club_seasons"] = [
        club_seasons(evidence.fpl_teams.get(code, {}), evidence.tm_clubs.get(pid, {}), club_map)
        for code, pid in zip(scored["fpl_code"], scored["player_id"])
    ]

    matches, reasons = pick_matches(scored)
    played = set(fpl_table.loc[fpl_table["fpl_pl_minutes"] > 0, "fpl_code"])
    still_open = sorted(played - set(matches["fpl_code"].astype(int)))
    rule3 = unique_name_matches(still_open, evidence, set(matches["player_id"].astype(int)), club_map)
    for code in rule3["fpl_code"]:
        reasons.pop(int(code), None)
    matches = pd.concat([matches, rule3], ignore_index=True)

    manual = read_manual_matches()
    matches, reasons = apply_manual_matches(
        matches, reasons, manual, set(fpl_table["fpl_code"]), set(inputs.tm_players["player_id"]),
        evidence, club_map,
    )
    # After the manual fixes, so a manual row cannot silently duplicate an automatic match.
    matches, conflicts = resolve_shared_players(matches, inputs.fpl_seasons)
    reasons.update(conflicts)
    matched = set(matches["fpl_code"].astype(int))
    for code, born, minutes in fpl_table[["fpl_code", "fpl_birth_date", "fpl_pl_minutes"]].itertuples(index=False):
        if code not in matched and code not in reasons:
            reasons[int(code)] = no_match_reason(born, int(minutes))

    table = build_match_table(matches, fpl_table, inputs, evidence)
    return MatchingResult(
        matches=table,
        unmatched=build_unmatched_report(reasons, fpl_table, scored, inputs),
        rates=match_rates(fpl_table, inputs.fpl_seasons, matched),
        season_rates=season_rates(inputs.fpl_seasons, matched),
        gaps=transfermarkt_gaps(table, evidence),
        club_map=club_map_table(club_map, inputs.fpl_team_names, inputs.tm_club_names),
        manual_rows=len(manual),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def markdown_table(frame: pd.DataFrame) -> str:
    """A DataFrame as a markdown table, without helper columns starting with '_'."""
    if frame.empty:
        return "_None._"
    return frame[[c for c in frame.columns if not str(c).startswith("_")]].to_markdown(index=False)


def build_report(result: MatchingResult) -> str:
    """The matching report: rules, match rates, checks and what is left to review."""
    matches, unmatched = result.matches, result.unmatched
    played = matches[matches["fpl_pl_minutes"] > 0]
    bins = pd.cut(
        played["minutes_agreement"], bins=[-0.01, 0.5, 0.8, 0.95, 1.0],
        labels=["below 0.5", "0.5 to 0.8", "0.8 to 0.95", "0.95 or more"],
    ).cat.add_categories("no season to compare").fillna("no season to compare")
    order = ["0.95 or more", "0.8 to 0.95", "0.5 to 0.8", "below 0.5", "no season to compare"]
    agreement = bins.value_counts().reindex(order, fill_value=0)
    weak = played[(played["rule"] != RULE_MINUTES) & (played["minutes_agreement"] < config.MATCH_MIN_MINUTES_AGREEMENT)]
    differing_dates = matches[matches["birth_dates_agree"] == "no"]
    rule3 = matches[matches["rule"] == RULE_UNIQUE_NAME]
    shared = matches[matches["note"].str.contains("same Transfermarkt player", na=False)]
    inconsistent_clubs = result.club_map[result.club_map["_clubs"] > 1]
    check_columns = ["fpl_code", "tm_player_id", "rule", "fpl_name", "tm_name", "fpl_birth_date",
                     "tm_birth_date", "fpl_seasons", "fpl_pl_minutes", "minutes_agreement",
                     "tm_missing_seasons", "note"]
    reason_counts = unmatched["reason"].str.split(":").str[0].value_counts()
    lines = [
        "# FPL to Transfermarkt matching",
        "",
        f"Generated by `src/matching.py` at {utc_now()} (UTC).",
        "",
        "## Rules",
        "",
        f"- **Rule 1, {RULE_BIRTH_DATE}:** FPL birth date equals the Transfermarkt date of birth, "
        f"and the name scores at least {config.MATCH_NAME_SCORE_WITH_BIRTH_DATE} out of 100.",
        f"- **Rule 2, {RULE_MINUTES}:** for players with no rule-1 match, the name scores at least "
        f"{config.MATCH_NAME_SCORE_WITHOUT_BIRTH_DATE}, both sources have minutes in at least one "
        f"season, and the Premier League minutes agree by at least {config.MATCH_MIN_MINUTES_AGREEMENT} "
        "(1 = identical) over the seasons where Transfermarkt has minutes.",
        f"- **Rule 3, {RULE_UNIQUE_NAME}:** for players with PL minutes still unmatched, exactly one "
        f"Transfermarkt player has the full name (no other scores {config.MATCH_UNIQUE_NAME_RUNNER_UP} "
        f"or more), aged {config.MATCH_PLAUSIBLE_AGE[0]} to {config.MATCH_PLAUSIBLE_AGE[1]} in the first "
        "season with minutes, with no Transfermarkt PL minutes in seasons where FPL has none.",
        "- **Ties** go to the candidate who played for the matching Transfermarkt club in more seasons, "
        f"then to a name score at least {config.MATCH_DECISIVE_NAME_MARGIN} points higher, then to "
        f"minutes agreement at least {config.MATCH_DECISIVE_MINUTES_MARGIN} higher. Otherwise the player "
        "is left for review.",
        f"- **Manual fixes:** {result.manual_rows} rows in `data/manual_matches.csv`.",
        "",
        "## Match rates",
        "",
        markdown_table(result.rates),
        "",
        "Players with PL minutes, season by season:",
        "",
        markdown_table(result.season_rates),
        "",
        "Matches by rule:",
        "",
        markdown_table(matches["rule"].value_counts().rename_axis("rule").reset_index(name="players")),
        "",
        "## Transfermarkt gaps",
        "",
        "Matched players with FPL minutes in a season but no Transfermarkt Premier League rows for it. "
        "FPL's minutes are complete, so these are seasons missing from Transfermarkt's appearances. The "
        "same gaps probably affect its other leagues, where there is nothing to check them against.",
        "",
        markdown_table(result.gaps),
        "",
        "## Checks on the matches",
        "",
        "Minutes agreement for matched players who played, over the seasons where Transfermarkt has "
        f"minutes (rule 2 requires at least {config.MATCH_MIN_MINUTES_AGREEMENT}; for the other rules "
        "it is an independent check):",
        "",
        markdown_table(agreement.rename_axis("minutes agreement").reset_index(name="players")),
        "",
        f"Rule-1 and rule-3 matches whose minutes agree by less than {config.MATCH_MIN_MINUTES_AGREEMENT} "
        f"({len(weak)}), worth a look:",
        "",
        markdown_table(weak[check_columns]),
        "",
        f"Rule-3 matches ({len(rule3)}), the weakest rule, worth a look:",
        "",
        markdown_table(rule3[check_columns]),
        "",
        f"Matches where the two sources give different birth dates ({len(differing_dates)}):",
        "",
        markdown_table(differing_dates[check_columns]),
        "",
        f"FPL codes sharing one Transfermarkt player ({len(shared)}):",
        "",
        markdown_table(shared[check_columns]),
        "",
        "## Club map",
        "",
        "Which Transfermarkt club each FPL team is, learned from the clear-cut matches. "
        f"Teams mapped to more than one Transfermarkt club across seasons: {len(inconsistent_clubs)}.",
        "",
        markdown_table(result.club_map),
        "",
        "## Not matched",
        "",
        markdown_table(reason_counts.rename_axis("reason").reset_index(name="players")),
        "",
        "The 15 unmatched players with the most PL minutes (all of them are in "
        "`data/processed/unmatched_report.csv`):",
        "",
        markdown_table(unmatched.head(15)[["fpl_code", "fpl_name", "fpl_birth_date", "fpl_seasons",
                                           "fpl_pl_minutes", "reason", "name_suggestions"]]),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Match, write the outputs and print the match rates."""
    config.ensure_directories()
    print("Reading and tidying the data ...")
    inputs = load_inputs()
    print("Matching ...")
    result = run_matching(inputs)
    # utf-8-sig adds a byte-order mark so Excel shows accented names correctly;
    # pandas reads these files back without any special options.
    result.matches.to_csv(config.MATCHES_PATH, index=False, encoding="utf-8-sig")
    result.unmatched.to_csv(config.UNMATCHED_REPORT_PATH, index=False, encoding="utf-8-sig")
    config.MATCHING_REPORT_PATH.write_text(build_report(result), encoding="utf-8")
    print()
    print(result.rates.to_string(index=False))
    print()
    print(result.matches["rule"].value_counts().to_string())
    print(f"\nMatches:   {config.MATCHES_PATH} ({len(result.matches)} rows)")
    print(f"Unmatched: {config.UNMATCHED_REPORT_PATH} ({len(result.unmatched)} rows)")
    print(f"Report:    {config.MATCHING_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
