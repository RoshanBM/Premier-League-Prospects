"""Data-quality audit of the raw FPL and Transfermarkt files.

Reads the files in data/raw/ exactly as downloaded (nothing is cleaned or
dropped first), runs the checks listed for Day 1, task 3 in CLAUDE.md and
writes data/processed/audit_report.md.

Every number in the report is computed here from the data. Statements from
CLAUDE.md (section 4, plus the defensive-stats limitation in section 9) are
listed at the top of the report with a status and the evidence, so the brief
can be corrected wherever the data disagrees with it.

Run from the repository root:
    python -m src.audit
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import combinations

import pandas as pd
from unidecode import unidecode

from src import config
from src.load_data import (
    detect_encoding,
    read_fpl_gameweeks,
    read_fpl_players,
    read_transfermarkt,
    utc_now,
)

# ---------------------------------------------------------------------------
# What CLAUDE.md says, and settings for the checks
# ---------------------------------------------------------------------------

# Section 4.1: each players_raw.csv field and the first season it should appear in.
EXPECTED_FIRST_SEASON: dict[str, str] = {
    **dict.fromkeys(
        ("minutes", "goals_scored", "assists", "clean_sheets", "bonus", "bps",
         "influence", "creativity", "threat", "ict_index"),
        "2016-17",
    ),
    **dict.fromkeys(("expected_goals", "expected_assists", "starts"), "2022-23"),
    **dict.fromkeys(("birth_date", "opta_code"), "2024-25"),
    **dict.fromkeys(
        ("tackles", "clearances_blocks_interceptions", "recoveries", "defensive_contribution"),
        "2025-26",
    ),
}
EXPECTED_PLAYERS_CURRENT: int = 841
EXPECTED_MISSING_BIRTH_DATES_CURRENT: int = 19
# Section 4.2. Transfermarkt seasons are start years, so 2012 means 2012/13.
EXPECTED_TM_SEASONS: tuple[int, int] = (2012, 2025)
EXPECTED_TM_LAST_GAME: str = "2026-07-06"
EXPECTED_TM_LAST_APPEARANCE: str = "2026-06-28"

FPL_POSITIONS: dict[int, str] = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
DEFENSIVE_STATS: tuple[str, ...] = ("tackles", "clearances_blocks_interceptions", "recoveries")
# Extra stats seen in the early gameweek files.
EARLY_GAMEWEEK_STATS: tuple[str, ...] = (
    *DEFENSIVE_STATS,
    "attempted_passes",
    "completed_passes",
    "key_passes",
    "big_chances_created",
    "dribbles",
)
# Season totals compared with the sum of the gameweek rows.
TOTAL_STATS: tuple[str, ...] = ("minutes", "goals_scored", "assists")

# Well-known players with long, moving or interrupted Premier League careers.
SPOT_CHECK_PLAYERS: tuple[str, ...] = (
    "Mohamed Salah", "Bukayo Saka", "Declan Rice", "Jordan Pickford", "James Milner",
    "Raheem Sterling", "Heung-Min Son", "Harry Kane", "Jamie Vardy", "Granit Xhaka",
)

PLAUSIBLE_AGE: tuple[int, int] = (15, 45)  # years; ages outside this range are flagged
EXTRA_TIME_MINUTES: int = 120  # 90 minutes plus 30 of extra time
# A Unicode replacement character, or UTF-8 text wrongly read as Latin-1 ("Ã©" for "é").
GARBLED_TEXT: re.Pattern[str] = re.compile("�|[ÂÃ][\u0080-¿]")
EXAMPLE_ROWS: int = 10

# Statuses used in the summary table.
CONFIRMED = "Confirmed"
WITH_EXCEPTIONS = "Confirmed, with exceptions"
CONTRADICTED = "Contradicted"
CHECKED = "Checked"  # an open question in CLAUDE.md (marked VERIFY), now answered


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FplData:
    """The raw FPL files for every season, read without changes."""

    players: dict[str, pd.DataFrame]
    gameweeks: dict[str, pd.DataFrame]
    player_encodings: dict[str, str]
    gameweek_encodings: dict[str, str]


@dataclass
class TransfermarktData:
    """The raw Transfermarkt tables, read without changes."""

    players: pd.DataFrame
    appearances: pd.DataFrame
    games: pd.DataFrame
    competitions: pd.DataFrame
    valuations: pd.DataFrame
    clubs: pd.DataFrame


def load_fpl() -> FplData:
    """Read players_raw.csv and gws/merged_gw.csv for every season."""
    seasons = config.FPL_SEASONS
    return FplData(
        players={season: read_fpl_players(season) for season in seasons},
        gameweeks={season: read_fpl_gameweeks(season) for season in seasons},
        player_encodings={
            season: detect_encoding(config.fpl_raw_path(season, "players_raw.csv"))
            for season in seasons
        },
        gameweek_encodings={
            season: detect_encoding(config.fpl_raw_path(season, "gws/merged_gw.csv"))
            for season in seasons
        },
    )


def load_transfermarkt() -> TransfermarktData:
    """Read the six Transfermarkt tables."""
    return TransfermarktData(
        players=read_transfermarkt("players"),
        appearances=read_transfermarkt("appearances"),
        games=read_transfermarkt("games"),
        competitions=read_transfermarkt("competitions"),
        valuations=read_transfermarkt("player_valuations"),
        clubs=read_transfermarkt("clubs"),
    )


# ---------------------------------------------------------------------------
# Report building blocks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Claim:
    """A statement from CLAUDE.md and what the data says about it."""

    statement: str
    status: str
    evidence: str


@dataclass
class Section:
    """One numbered part of the report, plus what it adds to the summary."""

    title: str
    parts: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)

    def subheading(self, text: str) -> None:
        self.parts.append(f"### {text}\n")

    def text(self, paragraph: str) -> None:
        self.parts.append(paragraph.strip() + "\n")

    def table(self, frame: pd.DataFrame) -> None:
        """Add a table, leaving out helper columns whose names start with '_'."""
        if frame.empty:
            self.parts.append("_None._\n")
            return
        shown = frame[[column for column in frame.columns if not str(column).startswith("_")]]
        self.parts.append(shown.to_markdown(index=False) + "\n")


def passed(ok: bool) -> str:
    """Status for a statement that is simply true or false."""
    return CONFIRMED if ok else CONTRADICTED


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def name_tokens(text: object) -> frozenset[str]:
    """Split a name into lower-case ASCII parts: "Heung-Min Son" -> {"heung", "min", "son"}."""
    plain = unidecode(str(text)).lower()
    return frozenset(part for part in re.split(r"[^a-z0-9]+", plain) if part)


def to_dates(values: pd.Series) -> pd.Series:
    """Parse ISO dates; anything unreadable becomes NaT."""
    return pd.to_datetime(values, errors="coerce", format="ISO8601")


def describe_fpl_seasons(seasons: Iterable[str]) -> str:
    """Describe FPL seasons as runs, e.g. "2016-17 to 2023-24, 2025-26"."""
    present = set(seasons)
    runs: list[list[str]] = []
    in_run = False
    for season in config.FPL_SEASONS:
        if season in present:
            if in_run:
                runs[-1][1] = season
            else:
                runs.append([season, season])
        in_run = season in present
    return ", ".join(a if a == b else f"{a} to {b}" for a, b in runs) or "none"


def describe_numbers(numbers: Iterable[int]) -> str:
    """Describe whole numbers as runs, e.g. Transfermarkt seasons "2012 to 2019, 2021"."""
    runs: list[list[int]] = []
    for number in sorted({int(n) for n in numbers}):
        if runs and number == runs[-1][1] + 1:
            runs[-1][1] = number
        else:
            runs.append([number, number])
    return ", ".join(str(a) if a == b else f"{a} to {b}" for a, b in runs) or "none"


def plural(count: int, word: str) -> str:
    """Return "1 player" or "2 players"."""
    return f"{count} {word}{'' if count == 1 else 's'}"


def describe_runs(seasons: Iterable[str], values: Iterable[str]) -> str:
    """Collapse repeats, e.g. "Sunderland (2016-17), Everton (2017-18 to 2025-26)"."""
    runs: list[list[str]] = []
    for season, value in zip(seasons, values):
        if runs and runs[-1][0] == value:
            runs[-1][2] = season
        else:
            runs.append([value, season, season])
    return ", ".join(f"{v} ({a})" if a == b else f"{v} ({a} to {b})" for v, a, b in runs)


# ---------------------------------------------------------------------------
# FPL helpers
# ---------------------------------------------------------------------------

def players_long(fpl: FplData) -> pd.DataFrame:
    """Stack players_raw for every season into one table of the columns the checks need."""
    columns = ["code", "id", "first_name", "second_name", "web_name", "element_type",
               "team_code", "minutes"]
    frames = []
    for season, players in fpl.players.items():
        frame = players[columns].copy()
        frame["birth_date"] = players["birth_date"] if "birth_date" in players.columns else None
        frame.insert(0, "season", season)
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)
    long["full_name"] = long["first_name"] + " " + long["second_name"]
    long["tokens"] = long["full_name"].map(name_tokens)
    # Order-free key, so "Heung-Min Son" and "Son Heung-min" count as one name.
    long["name_key"] = long["tokens"].map(lambda tokens: " ".join(sorted(tokens)))
    return long


def fpl_club_names(fpl: FplData) -> pd.DataFrame:
    """Name each (season, team_code) from the club names in the gameweek files.

    From 2020-21, merged_gw.csv has a club name on every row. A team_code's name
    is the most common name across its players' rows, so a player who changed
    club mid-season cannot mislabel a team.
    """
    rows = []
    for season, gameweeks in fpl.gameweeks.items():
        if "team" not in gameweeks.columns:
            continue
        codes = fpl.players[season][["id", "team_code"]].rename(columns={"id": "element"})
        joined = gameweeks[["element", "team"]].merge(codes, on="element", how="inner")
        for team_code, names in joined.groupby("team_code")["team"]:
            rows.append({"season": season, "team_code": int(team_code), "club": names.mode().iat[0]})
    return pd.DataFrame(rows)


def club_lookup(clubs: pd.DataFrame) -> dict[int, str]:
    """Map team_code to club name, using the latest season the code appears in."""
    latest = clubs.sort_values("season").drop_duplicates("team_code", keep="last")
    return dict(zip(latest["team_code"], latest["club"]))


def club_name(lookup: dict[int, str], team_code: int) -> str:
    """Return a club name, or the bare code for clubs only seen before 2020-21."""
    return lookup.get(team_code, f"team_code {team_code}")


def opta_code_check(fpl: FplData) -> pd.DataFrame:
    """Compare code with opta_code in the seasons that have opta_code."""
    rows = []
    for season, players in fpl.players.items():
        if "opta_code" not in players.columns:
            continue
        is_player = players["element_type"].isin(list(FPL_POSITIONS))
        opta = players["opta_code"].astype(str)
        matches = opta == "p" + players["code"].astype(str)
        rows.append(
            {
                "season": season,
                "players (GK to FWD)": int(is_player.sum()),
                "opta_code = 'p' + code": int((matches & is_player).sum()),
                "other rows": int((~is_player).sum()),
                "other rows with opta_code 'man...'": int((opta.str.startswith("man") & ~is_player).sum()),
            }
        )
    return pd.DataFrame(rows)


def spot_check(long: pd.DataFrame, lookup: dict[int, str]) -> pd.DataFrame:
    """Every season, code, id and club for each well-known player, found by name."""
    rows = []
    for player in SPOT_CHECK_PLAYERS:
        wanted = name_tokens(player)
        found = long[long["tokens"].map(lambda tokens: wanted <= tokens).astype(bool)]
        found = found.sort_values("season")
        if found.empty:
            rows.append({"player": player, "seasons listed": "not found", "codes": ""})
            continue
        seasons = list(found["season"])
        listed = describe_fpl_seasons(seasons)
        rows.append(
            {
                "player": player,
                "seasons listed": f"{len(set(seasons))}: {listed}",
                "codes": ", ".join(str(code) for code in sorted(found["code"].unique())),
                "id in each season": ", ".join(str(i) for i in found["id"]),
                "clubs": describe_runs(seasons, [club_name(lookup, c) for c in found["team_code"]]),
                "names used": "; ".join(dict.fromkeys(found["full_name"])),
                "_moved": found["team_code"].nunique() > 1,
                "_away": "," in listed,
            }
        )
    return pd.DataFrame(rows)


def same_name_different_codes(long: pd.DataFrame, lookup: dict[int, str]) -> pd.DataFrame:
    """List names that appear under more than one code, with a verdict for each name."""
    codes_per_name = long.groupby("name_key")["code"].nunique()
    rows = []
    for key in codes_per_name[codes_per_name > 1].index:
        entries = long[long["name_key"] == key]
        by_code = dict(tuple(entries.groupby("code")))
        seasons = {code: set(group["season"]) for code, group in by_code.items()}
        teams = {code: set(group["team_code"]) for code, group in by_code.items()}
        pairs = list(combinations(by_code, 2))
        if any(seasons[a] & seasons[b] for a, b in pairs):
            verdict = "different players: both listed in the same season"
        elif any(teams[a] & teams[b] for a, b in pairs):
            verdict = "possibly one player under two codes: same club, never in the same season"
        else:
            verdict = "probably different players: different clubs and seasons"
        for code, group in by_code.items():
            rows.append(
                {
                    "name": group["full_name"].iat[0],
                    "code": code,
                    "seasons": describe_fpl_seasons(group["season"]),
                    "clubs": ", ".join(dict.fromkeys(club_name(lookup, c) for c in group["team_code"])),
                    "minutes": int(group["minutes"].sum()),
                    "verdict": verdict,
                    "_name_key": key,
                }
            )
    return pd.DataFrame(rows)


def same_birth_date_different_codes(long: pd.DataFrame) -> pd.DataFrame:
    """Rows whose name and birth date appear under more than one code."""
    dated = long.dropna(subset=["birth_date"])
    codes = dated.groupby(["name_key", "birth_date"])["code"].nunique()
    flagged = codes[codes > 1].reset_index()[["name_key", "birth_date"]]
    found = dated.merge(flagged, on=["name_key", "birth_date"])
    return found[["season", "full_name", "birth_date", "code", "team_code", "name_key"]].rename(
        columns={"name_key": "_name_key"}
    )


def linked_code_groups(code_sets: Iterable[set[int]]) -> list[set[int]]:
    """Merge sets of codes that share a member, so {1, 2} and {2, 3} become {1, 2, 3}."""
    groups: list[set[int]] = []
    for codes in code_sets:
        merged = set(codes)
        for group in [g for g in groups if g & merged]:
            merged |= group
            groups.remove(group)
        groups.append(merged)
    return groups


def suspected_code_changes(same: pd.DataFrame, dated: pd.DataFrame) -> list[set[int]]:
    """Group the codes that probably belong to one player, from the name and birth-date checks."""
    code_sets: list[set[int]] = []
    if not same.empty:
        possible = same[same["verdict"].str.startswith("possibly")]
        code_sets += [set(rows["code"]) for _, rows in possible.groupby("_name_key")]
    if not dated.empty:
        code_sets += [set(rows["code"]) for _, rows in dated.groupby(["_name_key", "birth_date"])]
    return linked_code_groups(code_sets)


def describe_code_group(codes: set[int], long: pd.DataFrame) -> str:
    """Describe one suspected player: names used, codes and total PL minutes."""
    rows = long[long["code"].isin(codes)]
    names = " / ".join(dict.fromkeys(rows["full_name"]))
    return (
        f"{names} (codes {', '.join(str(code) for code in sorted(codes))}; "
        f"{int(rows['minutes'].sum())} PL minutes in total)"
    )


def name_changes_within_codes(long: pd.DataFrame) -> pd.DataFrame:
    """Codes whose name differs between seasons, with the name parts all versions share."""
    variants = long.groupby("code")["name_key"].nunique()
    rows = []
    for code in variants[variants > 1].index:
        versions = long[long["code"] == code].drop_duplicates("name_key")
        shared = frozenset.intersection(*versions["tokens"])
        rows.append(
            {
                "code": code,
                "names (season first used)": "; ".join(
                    f"{name} ({season})" for name, season in zip(versions["full_name"], versions["season"])
                ),
                "shared name parts": " ".join(sorted(shared)) or "(none)",
                "_shared": len(shared),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["code", "names (season first used)", "shared name parts", "_shared"])
    return pd.DataFrame(rows).sort_values(["_shared", "code"], ignore_index=True)


def club_code_consistency(clubs: pd.DataFrame) -> str:
    """Check that each team_code keeps one club name, and each club one team_code."""
    names_per_code = clubs.groupby("team_code")["club"].nunique()
    codes_per_name = clubs.groupby("club")["team_code"].nunique()
    seasons = describe_fpl_seasons(clubs["season"])
    if names_per_code.max() == 1 and codes_per_name.max() == 1:
        return (
            f"Club names come from the gameweek files, which have them for {seasons}. Across those "
            f"seasons each of the {len(names_per_code)} `team_code` values always carries the same "
            "club name, and each club always has the same `team_code`."
        )
    odd_codes = names_per_code[names_per_code > 1].index.tolist()
    odd_names = codes_per_name[codes_per_name > 1].index.tolist()
    return (
        f"Club names come from the gameweek files ({seasons}). team_code values with more than one "
        f"name: {odd_codes or 'none'}; club names with more than one team_code: {odd_names or 'none'}."
    )


def compare_totals(players: pd.DataFrame, gameweeks: pd.DataFrame) -> pd.DataFrame:
    """Put players_raw season totals next to summed gameweek rows, one row per player."""
    sums = (
        gameweeks.groupby("element", as_index=False)[list(TOTAL_STATS)].sum()
        .rename(columns={"element": "id"})
    )
    merged = players[["id", "web_name", *TOTAL_STATS]].merge(
        sums, on="id", how="outer", suffixes=("", "_gw"), indicator="source"
    )
    differs = pd.Series(False, index=merged.index)
    for stat in TOTAL_STATS:
        merged[f"{stat}_gw"] = merged[f"{stat}_gw"].fillna(0)
        differs = differs | (merged[stat] != merged[f"{stat}_gw"])
    merged["differs"] = differs
    return merged


def gameweek_name_style(names: pd.Series) -> str:
    """Describe how merged_gw.csv writes names in a season."""
    text = names.astype(str)
    if text.str.contains(r"_\d+$").mean() > 0.5:
        return "First_Last_id"
    if text.str.contains("_").mean() > 0.5:
        return "First_Last"
    return "First Last"


# ---------------------------------------------------------------------------
# FPL sections
# ---------------------------------------------------------------------------

def position_label_agreement(fpl: FplData) -> str:
    """Check the element_type mapping against the position labels in merged_gw."""
    results = []
    for season in config.FPL_SEASONS:
        gameweeks = fpl.gameweeks[season]
        if "position" not in gameweeks.columns:
            continue
        labels = gameweeks.groupby("element")["position"].agg(lambda s: s.mode().iat[0])
        mapped = fpl.players[season].set_index("id")["element_type"].map(FPL_POSITIONS)
        both = pd.concat({"mapped": mapped, "label": labels}, axis=1, join="inner").dropna()
        results.append(f"{season} {100 * (both['mapped'] == both['label']).mean():.1f}%")
    return (
        "The mapping 1 = GK, 2 = DEF, 3 = MID, 4 = FWD was checked against the `position` column "
        "of `merged_gw.csv`, which exists from 2020-21. Share of players where the two agree: "
        + ", ".join(results) + "."
    )


def section_fpl_overview(fpl: FplData) -> Section:
    """Rows, players, positions and file encodings for each FPL season."""
    section = Section("FPL: rows and players per season")
    rows = []
    for season in config.FPL_SEASONS:
        players, gameweeks = fpl.players[season], fpl.gameweeks[season]
        position = players["element_type"].map(FPL_POSITIONS).fillna("other")
        minutes = players["minutes"]
        rows.append(
            {
                "season": season,
                "players_raw rows": len(players),
                "distinct code": players["code"].nunique(),
                "minutes > 0": int((minutes > 0).sum()),
                f"{config.MIN_MINUTES_FOR_PER90}+ min": int((minutes >= config.MIN_MINUTES_FOR_PER90).sum()),
                **{label: int((position == label).sum()) for label in (*FPL_POSITIONS.values(), "other")},
                "gameweek rows": len(gameweeks),
                "gameweeks": f"{gameweeks['GW'].nunique()} ({gameweeks['GW'].min()} to {gameweeks['GW'].max()})",
                "encoding (players_raw / merged_gw)": (
                    f"{fpl.player_encodings[season]} / {fpl.gameweek_encodings[season]}"
                ),
            }
        )
    section.text(
        "`players_raw.csv` lists everyone in the FPL game that season, including players who never "
        "played, so players with minutes are counted separately. Positions come from `element_type`. "
        "`merged_gw.csv` has one row per player per fixture."
    )
    section.table(pd.DataFrame(rows))
    gaps = []
    for season, gameweeks in fpl.gameweeks.items():
        numbers = set(gameweeks["GW"].unique())
        skipped = [gw for gw in range(min(numbers), max(numbers) + 1) if gw not in numbers]
        if skipped:
            word = "gameweeks" if len(skipped) > 1 else "gameweek"
            gaps.append(f"{season} skips {word} {describe_numbers(skipped)}")
    if gaps:
        section.text("Gameweek numbering has gaps: " + "; ".join(gaps) + ".")
    section.text(position_label_agreement(fpl))

    odd_seasons = []
    for season, players in fpl.players.items():
        odd = players[~players["element_type"].isin(list(FPL_POSITIONS))]
        if odd.empty:
            continue
        values = ", ".join(str(value) for value in sorted(odd["element_type"].unique()))
        managers = 0
        if "opta_code" in odd.columns:
            managers = int(odd["opta_code"].astype(str).str.startswith("man").sum())
        manager_columns = [column for column in players.columns if column.startswith("mng_")]
        section.text(
            f"{season} has {len(odd)} rows with element_type {values}. {managers} of them have an "
            f"`opta_code` starting with 'man', together they have {int(odd['minutes'].sum())} minutes, "
            f"and the season has {len(manager_columns)} manager columns (`mng_*`). These rows are "
            "managers, not players, and must be left out of any player analysis."
        )
        odd_seasons.append(f"{season}: {len(odd)} rows with element_type {values} (managers)")

    current = fpl.players[config.CURRENT_SEASON]
    section.claims.append(
        Claim(
            f"{config.CURRENT_SEASON} players_raw.csv has {EXPECTED_PLAYERS_CURRENT} players",
            passed(len(current) == EXPECTED_PLAYERS_CURRENT == current["code"].nunique()),
            f"{len(current)} rows, {current['code'].nunique()} distinct codes",
        )
    )
    section.claims.append(
        Claim(
            "element_type gives GK / DEF / MID / FWD only",
            CONTRADICTED if odd_seasons else CONFIRMED,
            "; ".join(odd_seasons) or "only the values 1 to 4 in every season",
        )
    )
    latin = [season for season, encoding in fpl.gameweek_encodings.items() if encoding != "utf-8"]
    if latin:
        section.highlights.append(
            f"`merged_gw.csv` for {', '.join(latin)} is not UTF-8, so it is read as Latin-1."
        )
    return section


def section_fpl_fields(fpl: FplData) -> Section:
    """Which section 4.1 fields exist in each season, plus the extra early gameweek stats."""
    section = Section("FPL: fields by season")
    seasons = list(config.FPL_SEASONS)
    present = {f: [f in fpl.players[s].columns for s in seasons] for f in EXPECTED_FIRST_SEASON}
    first_seen = {
        f: next((s for s, has in zip(seasons, flags) if has), "never") for f, flags in present.items()
    }
    unbroken = {
        f: first_seen[f] != "never" and all(flags[seasons.index(first_seen[f]):])
        for f, flags in present.items()
    }
    table = pd.DataFrame(
        {
            "field": list(EXPECTED_FIRST_SEASON),
            "CLAUDE.md: from": list(EXPECTED_FIRST_SEASON.values()),
            "first seen": [first_seen[f] for f in EXPECTED_FIRST_SEASON],
            **{
                season: ["yes" if present[f][i] else "-" for f in EXPECTED_FIRST_SEASON]
                for i, season in enumerate(seasons)
            },
        }
    )
    section.text(
        "Columns of `players_raw.csv` named in CLAUDE.md section 4.1 ('yes' means present). "
        "Goals are in the column `goals_scored`."
    )
    section.table(table)
    for expected in dict.fromkeys(EXPECTED_FIRST_SEASON.values()):
        fields = [f for f, s in EXPECTED_FIRST_SEASON.items() if s == expected]
        ok = all(first_seen[f] == expected and unbroken[f] for f in fields)
        when = "in every season" if expected == seasons[0] else f"from {expected}"
        section.claims.append(
            Claim(
                f"players_raw.csv has {', '.join(fields)} {when}",
                passed(ok),
                "present from that season on, with no gaps"
                if ok else "; ".join(f"{f}: first seen {first_seen[f]}" for f in fields),
            )
        )

    section.subheading("Extra stats in the early gameweek files")
    early = [s for s in seasons if all(stat in fpl.gameweeks[s].columns for stat in EARLY_GAMEWEEK_STATS)]
    rows = []
    for stat in EARLY_GAMEWEEK_STATS:
        row = {"stat": stat}
        for season in seasons:
            gameweeks = fpl.gameweeks[season]
            if stat not in gameweeks.columns:
                row[season] = "-"
                continue
            played = gameweeks[gameweeks["minutes"] > 0]
            row[season] = f"{100 * (played[stat] != 0).mean():.0f}%"
        rows.append(row)
    section.text(
        f"`merged_gw.csv` for {describe_fpl_seasons(early)} has stats that later files dropped. "
        "Each cell is the share of gameweek rows with minutes > 0 where the stat is not zero "
        "('-' means there is no such column). Non-zero shares show the columns hold real values, "
        "not placeholders. Their definitions are not documented, so they should not be compared "
        "with the 2025-26 defensive stats without further checks."
    )
    section.table(pd.DataFrame(rows))

    defensive_before = [
        s for s in seasons
        if s < config.CURRENT_SEASON
        and any(stat in fpl.gameweeks[s].columns and (fpl.gameweeks[s][stat] != 0).any() for stat in DEFENSIVE_STATS)
    ]
    in_players_raw_only_from_current = all(first_seen[stat] == config.CURRENT_SEASON for stat in DEFENSIVE_STATS)
    if defensive_before:
        evidence = (
            f"players_raw.csv has them only from {config.CURRENT_SEASON}"
            if in_players_raw_only_from_current else "players_raw.csv has them earlier too"
        )
        evidence += f"; but merged_gw.csv for {describe_fpl_seasons(defensive_before)} has non-zero {', '.join(DEFENSIVE_STATS)}"
    else:
        evidence = "no gameweek file before 2025-26 has non-zero values"
    section.claims.append(
        Claim(
            "Defensive stats only exist from 2025/26 (CLAUDE.md section 9)",
            CONTRADICTED if defensive_before else CONFIRMED,
            evidence,
        )
    )
    later = [s for s in seasons if s not in early]
    early_only = [
        stat for stat in EARLY_GAMEWEEK_STATS
        if stat not in DEFENSIVE_STATS and not any(stat in fpl.gameweeks[s].columns for s in later)
    ]
    if early and early_only:
        section.highlights.append(
            f"The gameweek files for {describe_fpl_seasons(early)} also have passing and chance-creation "
            f"stats ({', '.join(early_only)}) that no later season has."
        )
    return section


def section_code_stability(fpl: FplData, long: pd.DataFrame, clubs: pd.DataFrame) -> Section:
    """Is FPL's code the same for the same player in every season?"""
    section = Section("FPL: is `code` stable across seasons?")
    lookup = club_lookup(clubs)
    section.text(
        "`id` is renumbered every season, so multi-season histories need a key that stays the same. "
        "CLAUDE.md asks whether `code` is that key. The checks below approach it from several sides."
    )

    section.subheading("One row per code in each season")
    duplicates = pd.DataFrame(
        [
            {
                "season": season,
                "repeated code": int(players["code"].duplicated().sum()),
                "repeated id": int(players["id"].duplicated().sum()),
            }
            for season, players in fpl.players.items()
        ]
    )
    unique = int(duplicates[["repeated code", "repeated id"]].to_numpy().sum()) == 0
    if unique:
        section.text("No season repeats a `code` or an `id`.")
    else:
        section.text("Some seasons repeat a `code` or an `id`:")
        section.table(duplicates)

    section.subheading("`code` against `opta_code`")
    opta = opta_code_check(fpl)
    section.table(opta)
    opta_ok = bool((opta["opta_code = 'p' + code"] == opta["players (GK to FWD)"]).all())
    if opta_ok:
        section.text(
            "In every season that has `opta_code`, it is 'p' followed by `code` for every player, so "
            "`code` is the Opta player ID. An Opta ID is meant to stay with a player for good; the "
            "checks below look for exceptions, especially in the seasons without `opta_code`."
        )
    else:
        section.text("`opta_code` and `code` disagree for some players (see the table).")

    section.subheading("Spot checks of well-known players")
    spot = spot_check(long, lookup)
    single = spot["codes"].astype(str).str.fullmatch(r"\d+")
    moved = int(spot.get("_moved", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    away = int(spot.get("_away", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    section.text(
        "Each player is found by name in every season, so a second code for the same player would "
        "show up here. `id` changes every season; the question is whether `code` does."
    )
    section.table(spot)
    section.text(
        f"{int(single.sum())} of {len(spot)} players have a single `code` across all their seasons. "
        f"{moved} of them changed club and {away} had seasons away from the league in between."
    )

    section.subheading("Same name, different codes")
    same = same_name_different_codes(long, lookup)
    section.text(
        f"{same['_name_key'].nunique() if not same.empty else 0} names (ignoring accents, punctuation "
        "and word order) appear under more than one code. Two people can share a name, so each case "
        "is judged: listed in the same season means two different players; never in the same season "
        "but at the same club suggests one player whose code changed."
    )
    section.table(same)

    section.subheading("Same name and birth date, different codes")
    dated = same_birth_date_different_codes(long)
    section.text(
        "Where FPL gives `birth_date`, a name plus a birth date identifies a player. "
        + ("No name and birth date appears under more than one code." if dated.empty
           else "These appear under more than one code:")
    )
    if not dated.empty:
        section.table(dated)

    suspects = [describe_code_group(codes, long) for codes in suspected_code_changes(same, dated)]
    verb = "appears" if len(suspects) == 1 else "appear"
    section.text(
        (f"Taken together, {plural(len(suspects), 'player')} {verb} under several codes: "
         + "; ".join(suspects) + ".")
        if suspects else "No player appears under several codes."
    )
    section.text(
        "These checks rely on names, plus birth dates from 2024-25, so a player whose code and name "
        "both changed at the same time before 2024-25 would not be caught."
    )

    section.subheading("Name changes under one code")
    variants = name_changes_within_codes(long)
    no_shared = variants[variants["shared name parts"] == "(none)"]
    section.text(
        f"{len(variants)} of {long['code'].nunique()} codes carry more than one version of the name "
        f"across seasons. In {len(variants) - len(no_shared)} of them every version shares at least "
        "one name part, which points to formatting changes (fuller legal names, accents, nicknames) "
        f"rather than different people. {len(no_shared)} share no part at all. The "
        f"{min(EXAMPLE_ROWS, len(variants))} with the fewest shared parts are shown for review."
    )
    section.table(variants.head(max(EXAMPLE_ROWS, len(no_shared))))

    section.subheading("Club codes")
    section.text(club_code_consistency(clubs))

    if not unique or not opta_ok:
        status = CONTRADICTED
    elif suspects or not single.all():
        status = WITH_EXCEPTIONS
    else:
        status = CONFIRMED
    section.claims.append(
        Claim(
            "FPL `code` is stable for the same player across seasons (VERIFY)",
            status,
            f"no repeated code within a season; opta_code is 'p' + code for all "
            f"{int(opta['players (GK to FWD)'].sum()):,} player rows in {', '.join(opta['season'])}; "
            f"{int(single.sum())} of {len(spot)} spot-checked players keep one code; "
            + (f"{plural(len(suspects), 'player')} found under several codes: {'; '.join(suspects)}"
               if suspects else "no player found under several codes"),
        )
    )
    return section


def section_birth_dates(fpl: FplData, long: pd.DataFrame) -> Section:
    """Missing, odd and inconsistent birth dates in the seasons that have them."""
    section = Section("FPL: birth dates")
    seasons = [s for s in config.FPL_SEASONS if "birth_date" in fpl.players[s].columns]
    players = long[long["season"].isin(seasons) & long["element_type"].isin(list(FPL_POSITIONS))].copy()
    players["parsed"] = to_dates(players["birth_date"])
    known = players.dropna(subset=["parsed"])
    codes_with_date = set(known["code"])
    low, high = PLAUSIBLE_AGE
    rows = []
    for season in seasons:
        sub = players[players["season"] == season]
        missing = sub[sub["birth_date"].isna()]
        age = (pd.Timestamp(config.season_start_year(season), 9, 1) - sub["parsed"]).dt.days / 365.25
        rows.append(
            {
                "season": season,
                "players (GK to FWD)": len(sub),
                "no birth_date": len(missing),
                "of which played": int((missing["minutes"] > 0).sum()),
                f"of which {config.MIN_MINUTES_FOR_PER90}+ min": int(
                    (missing["minutes"] >= config.MIN_MINUTES_FOR_PER90).sum()
                ),
                "missing by position": ", ".join(
                    f"{FPL_POSITIONS[t]} {n}" for t, n in missing["element_type"].value_counts().sort_index().items()
                ),
                "known from another season": int(missing["code"].isin(codes_with_date).sum()),
                "unreadable dates": int((sub["birth_date"].notna() & sub["parsed"].isna()).sum()),
                f"age under {low} or over {high} on 1 Sep": int(((age < low) | (age > high)).sum()),
            }
        )
    table = pd.DataFrame(rows)
    section.text(
        "Managers are left out. 'Known from another season' counts missing birth dates that the same "
        "`code` has in another season. Ages are measured on 1 September of the season's first year."
    )
    section.table(table)
    dates_per_code = known.groupby("code")["parsed"].nunique()
    changed = dates_per_code[dates_per_code > 1]
    section.text(
        f"Players whose birth date differs between seasons: {len(changed)}."
        + (f" Codes: {', '.join(map(str, changed.index[:EXAMPLE_ROWS]))}." if len(changed) else "")
    )

    current = table[table["season"] == config.CURRENT_SEASON]
    missing_now = int(current["no birth_date"].iat[0]) if not current.empty else -1
    section.claims.append(
        Claim(
            f"{EXPECTED_MISSING_BIRTH_DATES_CURRENT} players in {config.CURRENT_SEASON} have no birth_date",
            passed(missing_now == EXPECTED_MISSING_BIRTH_DATES_CURRENT),
            f"{missing_now} missing; {int(current['of which played'].iat[0])} of them played",
        )
    )
    for values in table[table["season"] != config.CURRENT_SEASON].to_dict("records"):
        section.highlights.append(
            f"{values['season']}: {values['no birth_date']} of {values['players (GK to FWD)']} players have no "
            f"birth_date ({values['of which played']} of them played); {values['known from another season']} "
            "can be filled in from another season through `code`."
        )
    return section


def section_gameweek_totals(fpl: FplData) -> Section:
    """Do summed gameweek rows match the season totals in players_raw?"""
    section = Section("FPL: gameweek rows against season totals")
    summary, differences, repeated = [], [], {}
    for season in config.FPL_SEASONS:
        players, gameweeks = fpl.players[season], fpl.gameweeks[season]
        totals = compare_totals(players, gameweeks)
        listed = totals[totals["source"] != "right_only"]
        deduplicated = compare_totals(players, gameweeks.drop_duplicates())
        repeats = gameweeks[gameweeks.duplicated(["element", "fixture"], keep=False)]
        if not repeats.empty:
            repeated[season] = repeats
        summary.append(
            {
                "season": season,
                "players": len(players),
                "no gameweek rows": int((totals["source"] == "left_only").sum()),
                "in gameweeks only": int((totals["source"] == "right_only").sum()),
                "repeated (element, fixture) rows": int(gameweeks.duplicated(["element", "fixture"]).sum()),
                **{f"{stat} differs": int((listed[stat] != listed[f"{stat}_gw"]).sum()) for stat in TOTAL_STATS},
                "players with a difference": int(listed["differs"].sum()),
                "after removing exact copies": int(
                    deduplicated.loc[deduplicated["source"] != "right_only", "differs"].sum()
                ),
            }
        )
        if listed["differs"].any():
            differences.append(listed[listed["differs"]].assign(season=season))
    table = pd.DataFrame(summary)
    section.text(
        "For each player, the gameweek rows for minutes, goals and assists are summed and compared "
        "with the season totals in `players_raw.csv`, joining `element` to `id`. CLAUDE.md asks for "
        f"{config.CURRENT_SEASON}; every season is shown because the trajectory model will use them all. "
        "A player can appear only once per fixture, so a repeated (element, fixture) pair needs "
        "explaining. The last column repeats the comparison after removing rows that are exact copies "
        "of another row."
    )
    section.table(table)

    if differences:
        section.subheading("Players whose totals differ")
        columns = ["season", "id", "web_name"] + [c for stat in TOTAL_STATS for c in (stat, f"{stat}_gw")]
        shown = pd.concat(differences)[columns]
        note = f"All {len(shown)} are shown." if len(shown) <= 30 else f"The first 30 of {len(shown)} are shown."
        section.text(f"Columns ending in `_gw` are the summed gameweek rows. {note}")
        section.table(shown.head(30))

    for season, repeats in repeated.items():
        identical = not repeats.drop_duplicates().duplicated(["element", "fixture"]).any()
        differing = [
            column for column in repeats.columns
            if column not in ("element", "fixture")
            and (repeats.groupby(["element", "fixture"])[column].nunique(dropna=False) > 1).any()
        ]
        names = list(dict.fromkeys(repeats["name"].astype(str)))
        shown_names = ", ".join(names[:EXAMPLE_ROWS]) + (
            f" and {len(names) - EXAMPLE_ROWS} more" if len(names) > EXAMPLE_ROWS else ""
        )
        if identical:
            detail = "The copies are identical in every column, so they are plain duplicates."
        else:
            listed_gameweeks = ", ".join(str(gw) for gw in sorted(repeats["GW"].unique()))
            detail = (
                f"The copies are not identical: they sit in gameweeks {listed_gameweeks} and differ in "
                f"{len(differing)} columns, including {', '.join(differing[:6])}."
            )
        changed = int(table.loc[table["season"] == season, "players with a difference"].iat[0])
        effect = (
            "Even so, summed minutes, goals and assists match players_raw for every player."
            if changed == 0 else f"They change the summed totals for {plural(changed, 'player')}."
        )
        section.subheading(f"Repeated gameweek rows in {season}")
        section.text(
            f"{int(repeats.duplicated(['element', 'fixture']).sum())} extra rows repeat an (element, fixture) "
            f"pair, for {plural(len(names), 'player')}: {shown_names}. {detail} {effect}"
        )
        columns = [c for c in ("element", "name", "GW", "fixture", "kickoff_time", "minutes",
                               "goals_scored", "assists", "modified") if c in repeats.columns]
        section.table(repeats.sort_values(["element", "fixture"])[columns].head(20))

    current = table[table["season"] == config.CURRENT_SEASON].iloc[0]
    differ_now = int(current["players with a difference"])
    after = int(current["after removing exact copies"])
    text = (
        f"{config.CURRENT_SEASON}: summed gameweek minutes, goals and assists match players_raw for "
        f"{int(current['players']) - differ_now} of {int(current['players'])} players"
    )
    if differ_now:
        text += (
            "; removing exact duplicate gameweek rows fixes every difference, so the season totals are "
            "right and the gameweek file has duplicates" if after == 0
            else f"; {plural(after, 'player')} still differ after removing exact duplicate rows"
        )
    section.highlights.append(text + ".")
    others = table[(table["season"] != config.CURRENT_SEASON) & (table["players with a difference"] > 0)]
    if not others.empty:
        earlier = pd.concat([d for d in differences if d["season"].iat[0] != config.CURRENT_SEASON])
        largest_gap = int((earlier["minutes"] - earlier["minutes_gw"]).abs().max())
        section.highlights.append(
            "Other seasons where gameweek sums and season totals differ: "
            + ", ".join(
                f"{season} ({plural(int(count), 'player')})"
                for season, count in zip(others["season"], others["players with a difference"])
            )
            + f"; the largest gap is {largest_gap} minutes and the cause is not visible in the data."
        )
    return section


def section_fpl_anomalies(fpl: FplData) -> Section:
    """Impossible values, name formats and garbled text in the FPL files."""
    section = Section("FPL: duplicates and other anomalies")
    rows = []
    for season in config.FPL_SEASONS:
        players, gameweeks = fpl.players[season], fpl.gameweeks[season]
        counts = [c for c in ("minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
                              "saves", "bonus", "yellow_cards", "red_cards") if c in players.columns]
        names = pd.concat(
            [players["first_name"], players["second_name"], players["web_name"], gameweeks["name"]]
        ).dropna().astype(str)
        rows.append(
            {
                "season": season,
                "repeated code or id": int(players["code"].duplicated().sum() + players["id"].duplicated().sum()),
                f"season minutes > {config.MAX_SEASON_MINUTES}": int((players["minutes"] > config.MAX_SEASON_MINUTES).sum()),
                "negative counts": int((players[counts] < 0).any(axis=1).sum()),
                "largest gameweek minutes": int(gameweeks["minutes"].max()),
                "gameweek name style": gameweek_name_style(gameweeks["name"]),
                "garbled names": int(names.str.contains(GARBLED_TEXT).sum()),
            }
        )
    table = pd.DataFrame(rows)
    section.text(
        f"Season minutes above {config.MAX_SEASON_MINUTES} (38 x 90) or negative counts would be impossible. "
        "'Garbled names' counts names with a Unicode replacement character or UTF-8 text read as "
        "Latin-1. Repeated gameweek rows are covered in the previous section."
    )
    section.table(table)
    styles = table.groupby("gameweek name style", sort=False)["season"].agg(describe_fpl_seasons)
    if len(styles) > 1:
        section.highlights.append(
            "`merged_gw.csv` writes names differently by season: "
            + "; ".join(f"{style} in {seasons}" for style, seasons in styles.items())
            + ". Join gameweek rows on `element`, not on name."
        )
    garbled = int(table["garbled names"].sum())
    if garbled:
        section.highlights.append(f"{garbled} FPL names contain garbled characters.")
    return section


# ---------------------------------------------------------------------------
# Transfermarkt sections
# ---------------------------------------------------------------------------

def appearance_seasons(tm: TransfermarktData) -> pd.Series:
    """The season of each appearance, taken from its game."""
    return tm.appearances["game_id"].map(tm.games.set_index("game_id")["season"])


def section_tm_tables(tm: TransfermarktData) -> Section:
    """Table sizes, keys, date ranges and the season convention."""
    section = Section("Transfermarkt: tables, seasons and dates")
    specs = (
        ("players", tm.players, ("player_id",), None),
        ("appearances", tm.appearances, ("appearance_id",), "date"),
        ("appearances", tm.appearances, ("player_id", "game_id"), None),
        ("games", tm.games, ("game_id",), "date"),
        ("competitions", tm.competitions, ("competition_id",), None),
        ("player_valuations", tm.valuations, ("player_id", "date"), "date"),
        ("clubs", tm.clubs, ("club_id",), None),
    )
    rows = []
    for name, frame, key, date_column in specs:
        row = {
            "table": name,
            "rows": len(frame),
            "key checked": " + ".join(key),
            "repeated keys": int(frame.duplicated(list(key)).sum()),
            "date range": "",
        }
        if date_column:
            dates = to_dates(frame[date_column])
            row["date range"] = f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"
        rows.append(row)
    section.table(pd.DataFrame(rows))

    section.subheading("What a season means")
    games = tm.games.assign(date=to_dates(tm.games["date"]))
    by_season = (
        games.groupby("season")
        .agg(first_game=("date", "min"), last_game=("date", "max"), games=("game_id", "size"),
             competitions=("competition_id", "nunique"))
        .reset_index()
    )
    for column in ("first_game", "last_game"):
        by_season[column] = by_season[column].dt.strftime("%Y-%m-%d")
    window_start = pd.to_datetime(games["season"].astype(str) + "-01-01")
    window_end = pd.to_datetime((games["season"] + 1).astype(str) + "-08-31")
    outside = games[(games["date"] < window_start) | (games["date"] > window_end)]
    section.text(
        "`season` is the year a season starts (2012 means 2012/13); leagues that run over a calendar "
        "year use that year. A game is flagged if its date falls outside 1 January of its season year "
        f"to 31 August of the next year. Flagged games: {len(outside)}."
    )
    section.table(by_season)
    if not outside.empty:
        breakdown = (
            outside.groupby("competition_id")
            .agg(flagged_games=("game_id", "size"), seasons=("season", describe_numbers))
            .reset_index()
            .sort_values(["flagged_games", "competition_id"], ascending=[False, True])
        )
        section.text("Flagged games by competition:")
        section.table(breakdown)
        shown = outside.sort_values("date")[["season", "date", "competition_id", "home_club_name",
                                             "away_club_name", "round"]].head(EXAMPLE_ROWS).copy()
        shown["date"] = shown["date"].dt.strftime("%Y-%m-%d")
        section.text(f"The {len(shown)} earliest-dated flagged games:")
        section.table(shown)
        with_appearances = set(tm.appearances["competition_id"].unique())
        section.highlights.append(
            f"{len(outside)} Transfermarkt games fall outside the usual date window for their season: "
            f"{int(outside['competition_id'].isin(with_appearances).sum())} of them in competitions with "
            f"player appearances, {int((outside['competition_id'] == 'GB1').sum())} in the Premier League."
        )

    national = games["competition_type"] == "national_team_competition"
    club_seasons = games.loc[~national, "season"]
    app_seasons = appearance_seasons(tm)
    earlier_national = games[national & (games["season"] < EXPECTED_TM_SEASONS[0])]
    club_ok = (int(club_seasons.min()), int(club_seasons.max())) == EXPECTED_TM_SEASONS
    apps_ok = (int(app_seasons.min()), int(app_seasons.max())) == EXPECTED_TM_SEASONS
    if club_ok and apps_ok:
        status = WITH_EXCEPTIONS if not earlier_national.empty else CONFIRMED
    else:
        status = CONTRADICTED
    section.claims.append(
        Claim(
            "Transfermarkt covers seasons 2012/13 to 2025/26",
            status,
            f"club games {int(club_seasons.min())} to {int(club_seasons.max())}; appearances "
            f"{int(app_seasons.min())} to {int(app_seasons.max())}"
            + (f"; national-team games go back to {int(earlier_national['season'].min())} "
               f"({', '.join(sorted(earlier_national['competition_id'].unique()))}), but those early "
               "tournaments have no appearances"
               if not earlier_national.empty else ""),
        )
    )
    last_game = f"{games['date'].max():%Y-%m-%d}"
    last_appearance = f"{to_dates(tm.appearances['date']).max():%Y-%m-%d}"
    section.claims.append(
        Claim("No Transfermarkt games after 6 July 2026", passed(last_game <= EXPECTED_TM_LAST_GAME),
              f"latest game: {last_game}")
    )
    section.claims.append(
        Claim("No Transfermarkt appearances after 28 June 2026",
              passed(last_appearance <= EXPECTED_TM_LAST_APPEARANCE), f"latest appearance: {last_appearance}")
    )
    future = int((games["season"] > EXPECTED_TM_SEASONS[1]).sum())
    section.claims.append(
        Claim("No 2026/27 Transfermarkt data", passed(future == 0), f"{future} games with season 2026 or later")
    )
    return section


def competition_coverage(tm: TransfermarktData) -> pd.DataFrame:
    """Seasons with games and with player appearances, for every competition id seen."""
    game_seasons = tm.games.groupby("competition_id")["season"].agg(lambda s: set(s.astype(int)))
    appearances = tm.appearances[["competition_id"]].assign(season=appearance_seasons(tm))
    appearance_seasons_by_id = (
        appearances.dropna(subset=["season"]).groupby("competition_id")["season"]
        .agg(lambda s: set(s.astype(int)))
    )
    appearance_counts = appearances.groupby("competition_id").size()
    game_types = tm.games.groupby("competition_id")["competition_type"].agg(
        lambda s: s.mode().iat[0] if s.notna().any() else "(unknown)"
    )
    described = tm.competitions.set_index("competition_id")
    rows = []
    for cid in sorted(set(game_seasons.index) | set(appearance_counts.index) | set(described.index)):
        known = cid in described.index
        country = described.at[cid, "country_name"] if known else None
        rows.append(
            {
                "competition_id": cid,
                "name": described.at[cid, "name"] if known else "(not in competitions.csv)",
                "type": described.at[cid, "type"] if known else game_types.get(cid, ""),
                "country": country if isinstance(country, str) else "",
                "seasons with games": describe_numbers(game_seasons.get(cid, set())),
                "seasons with appearances": describe_numbers(appearance_seasons_by_id.get(cid, set())),
                "appearances": int(appearance_counts.get(cid, 0)),
                "_described": known,
            }
        )
    return pd.DataFrame(rows).sort_values(["type", "competition_id"], ignore_index=True)


def section_tm_competitions(tm: TransfermarktData) -> Section:
    """Which seasons each competition covers, and what that means for English football."""
    section = Section("Transfermarkt: seasons covered per competition")
    coverage = competition_coverage(tm)
    section.text(
        "'Seasons with games' come from `games`; 'seasons with appearances' are seasons with at least "
        "one player row in `appearances`, which is what player minutes are built from."
    )
    section.table(coverage)

    undescribed = coverage[~coverage["_described"]]
    premier_league_clubs = set(tm.clubs.loc[tm.clubs["domestic_competition_id"] == "GB1", "club_id"])
    profiles = []
    for cid in undescribed["competition_id"]:
        games = tm.games[tm.games["competition_id"] == cid].sort_values("date")
        if games.empty:
            continue
        first = games.iloc[0]
        involves_pl = games["home_club_id"].isin(premier_league_clubs) | games["away_club_id"].isin(premier_league_clubs)
        profiles.append(
            {
                "competition_id": cid,
                "games": len(games),
                "most games in a season": int(games.groupby("season").size().max()),
                "games with a Premier League club": int(involves_pl.sum()),
                "commonest rounds": ", ".join(games["round"].value_counts().index[:3]),
                "first game": f"{first['home_club_name']} v {first['away_club_name']} ({first['date']})",
            }
        )
    profiles = pd.DataFrame(profiles)
    if not profiles.empty:
        section.subheading("Competition ids missing from competitions.csv")
        section.text(
            "Clues to what each one is. A 'Premier League club' is a club whose `domestic_competition_id` "
            "in `clubs` is GB1."
        )
        section.table(profiles)
        section.highlights.append(
            f"{len(profiles)} competition ids used in games are missing from competitions.csv "
            f"({', '.join(profiles['competition_id'])}); section 8 lists clues to what they are."
        )

    english = tm.competitions[tm.competitions["country_name"] == "England"]
    english_leagues = english[english["type"] == "domestic_league"]
    section.subheading("English competitions")
    section.table(english[["competition_id", "name", "type", "sub_type"]])
    championship_games = 24 * 23  # a 24-club league where every club plays the other 23 twice
    linked = profiles[profiles["games with a Premier League club"] > 0] if not profiles.empty else profiles
    evidence = (
        "English competitions in competitions.csv: "
        + ", ".join(f"{r.competition_id} ({r.name}, {r.type})" for r in english.itertuples(index=False))
    )
    if not linked.empty:
        evidence += (
            f"; the ids missing from that file that involve Premier League clubs "
            f"({', '.join(linked['competition_id'])}) have at most "
            f"{int(linked['most games in a season'].max())} games in a season, against "
            f"{championship_games} in a 24-club league"
        )
    no_efl_league = set(english_leagues["competition_id"]) <= {"GB1"} and (
        linked.empty or int(linked["most games in a season"].max()) < championship_games
    )
    section.claims.append(
        Claim("Transfermarkt has no Championship or other EFL league", passed(no_efl_league), evidence)
    )

    leagues = coverage[coverage["type"] == "domestic_league"]
    with_appearances = leagues[leagues["appearances"] > 0]
    games_only = leagues[leagues["appearances"] == 0]
    games_only_seasons = tm.games.loc[tm.games["competition_id"].isin(games_only["competition_id"]), "season"]
    section.claims.append(
        Claim(
            "Which seasons have data for each competition (VERIFY)",
            CHECKED,
            f"{len(with_appearances)} first-tier leagues have player appearances "
            f"({', '.join(with_appearances['competition_id'])}); {len(games_only)} more have games only "
            f"({', '.join(games_only['competition_id'])}), for seasons {describe_numbers(games_only_seasons)}",
        )
    )
    if not games_only.empty:
        section.highlights.append(
            f"Player minutes exist for only {len(with_appearances)} first-tier leagues. The other "
            f"{len(games_only)} leagues ({', '.join(games_only['competition_id'])}) have games but no "
            "appearances, so a loan there is as invisible as a loan to the EFL."
        )
    return section


def section_tm_positions(tm: TransfermarktData) -> Section:
    """The position fields in the players table."""
    section = Section("Transfermarkt: position fields")
    players = tm.players
    table = pd.crosstab(
        players["sub_position"].fillna("(blank)"), players["position"].fillna("(blank)"),
        margins=True, margins_name="total",
    ).reset_index()
    literal_missing = int((players["position"] == "Missing").sum())
    blank = int(players["position"].isna().sum())
    nested = bool((players.dropna(subset=["sub_position"]).groupby("sub_position")["position"].nunique() == 1).all())
    section.text(
        f"`players` has one row per player, so each player has one `position` and one `sub_position` "
        f"(the current one, with no history). `position` has {players['position'].nunique()} values; "
        f"'Missing' is stored as text for {literal_missing} players, and {blank} are blank. "
        f"`sub_position` has {players['sub_position'].nunique()} values"
        + (", each inside a single `position`." if nested else ".")
    )
    section.table(table)
    section.claims.append(
        Claim(
            "How detailed the Transfermarkt position fields are (VERIFY)",
            CHECKED,
            f"position: {', '.join(sorted(players['position'].dropna().unique()))}; "
            f"{players['sub_position'].nunique()} sub-positions; one value per player, no history",
        )
    )
    if literal_missing:
        section.highlights.append(
            f"Transfermarkt `position` stores 'Missing' as text for {literal_missing} players; treat it as missing."
        )
    return section


def section_tm_anomalies(tm: TransfermarktData) -> Section:
    """Broken links between tables, odd minutes and ages, and garbled text."""
    section = Section("Transfermarkt: duplicates and other anomalies")
    apps = tm.appearances
    birth = to_dates(tm.players.set_index("player_id")["date_of_birth"])
    age = (to_dates(apps["date"]) - apps["player_id"].map(birth)).dt.days / 365.25
    low, high = PLAUSIBLE_AGE
    odd_age = apps.assign(age=age.round(1))[(age < low) | (age > high)]
    long_games = apps[apps["minutes_played"] > EXTRA_TIME_MINUTES]
    checks = [
        ("appearances repeating a (player_id, game_id) pair", int(apps.duplicated(["player_id", "game_id"]).sum())),
        ("appearances whose game_id is not in games", int((~apps["game_id"].isin(tm.games["game_id"])).sum())),
        ("appearances whose player_id is not in players", int((~apps["player_id"].isin(tm.players["player_id"])).sum())),
        ("valuations whose player_id is not in players", int((~tm.valuations["player_id"].isin(tm.players["player_id"])).sum())),
        ("games whose competition_id is not in competitions",
         int((~tm.games["competition_id"].isin(tm.competitions["competition_id"])).sum())),
        ("appearances with 0 minutes", int((apps["minutes_played"] == 0).sum())),
        (f"appearances over {EXTRA_TIME_MINUTES} minutes", len(long_games)),
        ("players with no date_of_birth", int(tm.players["date_of_birth"].isna().sum())),
        (f"appearances by a player under {low} or over {high}", len(odd_age)),
    ]
    section.table(pd.DataFrame(checks, columns=["check", "rows"]))

    if not long_games.empty:
        section.subheading(f"Appearances over {EXTRA_TIME_MINUTES} minutes")
        section.text(
            "Extra time takes a game to 120 minutes, so longer appearances are suspect. Minutes "
            "seen: " + ", ".join(f"{m} ({n})" for m, n in long_games["minutes_played"].value_counts().sort_index().items())
            + "."
        )
        section.table(long_games.sort_values("minutes_played", ascending=False)[
            ["date", "player_name", "competition_id", "minutes_played"]].head(EXAMPLE_ROWS))
    if not odd_age.empty:
        section.subheading(f"Appearances by players under {low} or over {high}")
        section.table(odd_age.sort_values("age")[["date", "player_name", "competition_id", "age"]].head(EXAMPLE_ROWS))

    section.subheading("Garbled text")
    text_columns = (
        ("players", "name"), ("players", "city_of_birth"), ("players", "current_club_name"),
        ("appearances", "player_name"), ("games", "home_club_name"), ("clubs", "name"),
        ("clubs", "stadium_name"),
    )
    rows = []
    for table_name, column in text_columns:
        values = getattr(tm, table_name)[column].dropna().astype(str)
        unique = pd.Series(values.unique())
        bad = unique[unique.str.contains(GARBLED_TEXT)]
        rows.append(
            {
                "table": table_name,
                "column": column,
                "rows": len(values),
                "rows with garbled text": int(values.isin(set(bad)).sum()),
                "examples": "; ".join(bad.head(3)),
            }
        )
    garbled = pd.DataFrame(rows)
    section.text(
        "Counts values containing a Unicode replacement character, which means the original letters "
        "were lost before the data was published, or UTF-8 text read as Latin-1."
    )
    section.table(garbled)

    duplicates = int(apps.duplicated(["player_id", "game_id"]).sum())
    section.claims.append(
        Claim("Transfermarkt appearances have one row per player per game", passed(duplicates == 0),
              f"{duplicates} repeated (player_id, game_id) pairs in {len(apps):,} rows")
    )
    if not long_games.empty:
        section.highlights.append(
            f"{len(long_games)} Transfermarkt appearances are longer than {EXTRA_TIME_MINUTES} minutes "
            f"(up to {int(long_games['minutes_played'].max())})."
        )
    if not odd_age.empty:
        section.highlights.append(
            f"{len(odd_age)} Transfermarkt appearances are by players under {low} or over {high}; "
            "their birth dates are worth checking, although very young debuts do happen."
        )
    bad_rows = int(garbled["rows with garbled text"].sum())
    if bad_rows:
        worst = garbled.sort_values("rows with garbled text", ascending=False).iloc[0]
        section.highlights.append(
            f"{bad_rows:,} Transfermarkt text values contain garbled characters, most in "
            f"{worst['table']}.{worst['column']} ({int(worst['rows with garbled text']):,})."
        )
    return section


# ---------------------------------------------------------------------------
# Putting the report together
# ---------------------------------------------------------------------------

def build_report(sections: list[Section]) -> str:
    """Summary first (claims and other findings), then each numbered section."""
    claims = pd.DataFrame(
        [
            {"statement in CLAUDE.md": c.statement, "status": c.status, "evidence": c.evidence,
             "report section": n}
            for n, section in enumerate(sections, start=1)
            for c in section.claims
        ]
    )
    highlights = [
        f"- {text} (section {n})"
        for n, section in enumerate(sections, start=1)
        for text in section.highlights
    ]
    lines = [
        "# Data audit report",
        "",
        f"Generated by `src/audit.py` at {utc_now()} (UTC) from the raw files in `data/raw/`, as "
        "listed in `data/raw/manifest.json`. Every number is computed from those files, read exactly "
        "as downloaded, before any cleaning.",
        "",
        "## Summary",
        "",
        "### Statements in CLAUDE.md, checked against the data",
        "",
        claims.to_markdown(index=False),
        "",
        "Not checked here: the FPL archive's 2026-27 season (not downloaded) and StatsBomb Open Data "
        "(a separate week-2 module).",
        "",
        "### Other findings",
        "",
        *highlights,
        "",
    ]
    for number, section in enumerate(sections, start=1):
        lines += [f"## {number}. {section.title}", "", *section.parts]
    return "\n".join(lines)


def main() -> int:
    """Run every check and write the report."""
    config.ensure_directories()
    print("Reading the FPL files ...")
    fpl = load_fpl()
    print("Reading the Transfermarkt files ...")
    tm = load_transfermarkt()
    print("Running the checks ...")
    long = players_long(fpl)
    clubs = fpl_club_names(fpl)
    sections = [
        section_fpl_overview(fpl),
        section_fpl_fields(fpl),
        section_code_stability(fpl, long, clubs),
        section_birth_dates(fpl, long),
        section_gameweek_totals(fpl),
        section_fpl_anomalies(fpl),
        section_tm_tables(tm),
        section_tm_competitions(tm),
        section_tm_positions(tm),
        section_tm_anomalies(tm),
    ]
    config.AUDIT_REPORT_PATH.write_text(build_report(sections), encoding="utf-8")
    print(f"\nReport written to {config.AUDIT_REPORT_PATH}\n")
    for section in sections:
        for claim in section.claims:
            print(f"  [{claim.status}] {claim.statement}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
