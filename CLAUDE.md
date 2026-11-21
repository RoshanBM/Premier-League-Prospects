# Project context: Football Manager-style Premier League U21 Prospects Scout

This file is the standing brief for this repository. Read it fully before doing anything, and re-read the relevant section before starting each new task.

---

## 1. Who I am and why this exists

I'm Roshan, an MSc Applied AI graduate (University of Warwick). I'm building this as a portfolio project for academy data science roles at Manchester City (Academy Football Insights & Innovation) and Tottenham Hotspur (Academy Data Insights). Both roles emphasise:

- player profiling, development pathways and trajectories
- longitudinal / hierarchical models (mixed-effects)
- robust, explainable, validated work in reproducible, version-controlled code
- clear insight for non-technical staff (coaches, recruitment, sport science)
- interactive apps and dashboards
- football data (StatsBomb is named explicitly)

Timeline: a working first version in 2 to 3 days, then polish before interviews.

My skills: Python (pandas, NumPy, scikit-learn, PyTorch), SQL, statsmodels, Plotly/Flask, Git, Docker. I have **not** used StatsBomb event data or mixed-effects models in a project before, so explain those parts as we go, briefly and concretely.

---

## 2. How to work with me (always apply)

### 2.1 Step by step
- Work on **one step at a time**. A step is one numbered task (for example, Day 1 task 2) or a smaller piece of one if it is large.
- Before starting a step, tell me in a few lines what you are about to do and which files you expect to touch.
- After finishing a step, **stop and wait for me to reply** before moving to the next one. Do not chain several steps together.

### 2.2 Report changed files after every step
End every step with a section titled **"Files changed in this step"** in this format:

```
Files changed in this step
  Created:   src/config.py         (paths, seasons, thresholds, URLs)
  Modified:  requirements.txt      (added rapidfuzz, unidecode)
  Deleted:   (none)
  Data written (gitignored): data/raw/fpl/2025-26/players_raw.csv
```

- List every file you created, modified or deleted, each with a one-line description of what it is or what changed.
- List data files separately, so it is clear they are gitignored and won't be committed.
- If nothing changed in a category, write `(none)`.
- Then give me a suggested commit message for the step.

### 2.3 Never commit
- **Never run `git init`, `git commit`, `git push`, `git add`, `git stash`, `git reset`, `git checkout`, `git merge`, `git rebase` or any other command that changes git state.** I commit everything manually.
- Read-only commands such as `git status`, `git diff` and `git log` are fine if they help you report changes.
- This applies even if a step seems finished, even if I say "looks good", and even if a tool or template suggests committing.

---

## 3. Ground rules (always apply)

1. **Never invent data, statistics or results.** If a dataset doesn't contain something, say so. If you haven't run something, don't report a number for it.
2. **Verify before relying.** When a fact about a dataset matters (a column, a season, an ID being stable), check it in the data and show me the check.
3. **Keep scope realistic.** If we're falling behind, propose cuts rather than rushing.
4. **Clean, modular, commented code** that I can explain in an interview. Type hints, docstrings, small functions, no notebooks for core logic.
5. **Give me complete files, not diffs or fragments**, when you create or rewrite a file.
6. **Never commit or change git state** (see section 2.3). I make every commit myself.
7. **Raw data must never be committed.** Everything under `data/` stays gitignored for licence reasons (see section 4), apart from `data/manual_matches.csv`.
8. **Writing style for docs, README and comments:** UK English, plain language, no em dashes.
9. Be honest about limitations in anything written for the README.
10. Use a virtual environment and pin versions in `requirements.txt`. Tell me before adding a new dependency.

---

## 4. Data sources: what has already been verified (20 Nov 2026)

These checks were done by reading the actual files. Treat them as facts, but re-check anything marked **VERIFY**.

Where the Day 1 checks of the downloaded files (the audit and the matching step, 20 Nov 2026) disagreed, the item was corrected in place on 21 Nov 2026 and marked *corrected by Day 1 audit, see audit_report.md*, with the original wording kept after "Previously". The VERIFY items have their answers added. Both reports are in `data/processed/` (gitignored; rebuild them with `python -m src.audit` and `python -m src.matching`).

### 4.1 Fantasy Premier League (core source for player cards)

- Archive: `vaastav/Fantasy-Premier-League` on GitHub (code is MIT; the data itself is the Premier League's).
- URL pattern:
  - `https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/players_raw.csv`
  - `https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv`
  - `{season}` looks like `2025-26`. Seasons `2016-17` to `2026-27` exist.
- Fields by season:
  - all seasons: minutes, goals, assists, clean sheets, bonus, bps, influence, creativity, threat, ICT index
  - from 2022-23: `expected_goals`, `expected_assists`, `starts`
  - from 2024-25: `birth_date`, `opta_code`
  - from 2025-26: `tackles`, `clearances_blocks_interceptions`, `recoveries`, `defensive_contribution`
    - That is true of `players_raw.csv`, but the gameweek files (`gws/merged_gw.csv`) for 2016-17 to 2018-19 also have `tackles`, `clearances_blocks_interceptions` and `recoveries`, plus `attempted_passes`, `completed_passes`, `key_passes`, `big_chances_created` and `dribbles`. 2019-20 to 2024-25 have none of them. The early definitions are not documented, so check them before comparing them with 2025-26. *(Corrected by Day 1 audit, see audit_report.md section 2. Previously this line read as if these stats exist only from 2025-26.)*
- 2025-26 `players_raw.csv`: 841 players, 19 with missing `birth_date`.
- The archive's 2026-27 data only runs to GW6 (12 Oct 2026). The live API (`https://fantasy.premierleague.com/api/bootstrap-static/`) has the latest, but it is unofficial and undocumented.
- Positions: `element_type` 1 to 4 give GK / DEF / MID / FWD. 2024-25 also has 20 rows with `element_type` 5: managers (`opta_code` starting 'man', no minutes, and the season has `mng_*` columns), who are left out of all player analysis. *(Corrected by Day 1 audit, see audit_report.md section 1. Previously: "`element_type` gives GK / DEF / MID / FWD only.")*
- Terms: the Premier League site terms allow private and personal use, prohibit commercial use, and prohibit creating a database from site material. So this is a non-commercial portfolio project, raw data is never committed, and the README credits the source.
- **VERIFY:** that FPL's `code` field is stable for the same player across seasons (the per-season `id` is not). We need this to build multi-season histories.
  - *Answered by Day 1 audit, see audit_report.md section 3:* stable. `code` is the Opta player ID (`opta_code` is 'p' + `code` for every player in 2024-25 and 2025-26). One exception: Kaine Kesler-Hayden appears under codes 465390, 490098 and 537043, with 5 PL minutes in total.
- FPL's influence / creativity / threat indices are composites with an unpublished formula. Prefer raw counts where they exist.

### 4.2 Transfermarkt dataset (ages, non-PL minutes, market value)

- Repo `dcaribou/transfermarkt-datasets`, also on Kaggle as `davidcariboo/player-scores`.
- Tables (csv.gz): `players`, `appearances`, `games`, `competitions`, `player_valuations`, `clubs`, `club_games`, `game_events`, `game_lineups`, `transfers`, `countries`, `national_teams`. Also published as a single DuckDB file.
- Download URL pattern from the README: `https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/{table}.csv.gz` (or download manually from Kaggle into `data/raw/transfermarkt/`).
- Seasons: 2012/13 to 2025/26. **Updates are paused**: no games after 6 July 2026, no appearances after 28 June 2026, no 2026/27 squads.
- Competitions: top flights in Europe and elsewhere, domestic cups, European club competitions, national-team tournaments. **There is no Championship or other EFL league.** A loan to the Championship shows up as a gap. This is a major limitation for English prospects and must be stated.
  - Games exist for all of these, but player appearances, and so minutes, only for some. Appearances exist for 14 first-tier leagues from 2012/13 to 2025/26 (BE1, DK1, ES1, FR1, GB1, GR1, IT1, L1, NL1, PO1, RU1, SC1, TR1, UKR1), for the 10 domestic cups in `competitions.csv` (including the FA Cup), and for the Champions League, Europa League and their qualifiers. 17 more leagues have games but no appearances, for the 2024 and 2025 seasons only (Japan: 2024): Austria, Argentina, Australia, Brazil, Switzerland, Japan, Croatia, Mexico, MLS, Norway, Poland, Romania, South Korea, Saudi Arabia, Sweden, Serbia and the Czech Republic. A loan to one of them is as invisible as a loan to the EFL. The Conference League has appearances for its qualifiers only, and of the national-team tournaments only AFCON and the World Cup in season 2025 have them. *(Corrected by Day 1 audit, see audit_report.md section 8. Previously this line read as if every competition listed had player data.)*
- `appearances`: one row per player per game, with minutes, goals, assists, cards.
  - Whole player-seasons are missing. Checked against FPL for the Premier League: some matched players have FPL minutes in a season but no Transfermarkt rows for it, from 1 to 29 players a season in 2017-18 to 2021-22, and none in 2016-17 or from 2022-23. The worst season is 2021-22: 29 of 537 players (5.4%), 0.8% of PL minutes. The same gaps probably affect other leagues, where nothing can check them, so non-PL minutes are a lower bound. *(Corrected by Day 1 matching, see matching_report.md, "Transfermarkt gaps". The matching step found this, not the audit. Previously this line read as if appearances were complete.)*
- Licence: the repo is CC0, but the data was scraped from Transfermarkt, whose terms prohibit automated collection. Non-commercial use only, never redistribute raw data, credit Transfermarkt and the dataset author.
- **VERIFY:** which seasons actually have data for each competition (some leagues may have been added later), and how detailed the position fields in `players` are.
  - *Answered by Day 1 audit, see audit_report.md sections 8 and 9:* seasons per competition as in the Competitions correction above. `position` has Attack, Defender, Goalkeeper and Midfield, plus 'Missing' stored as text for 586 players; `sub_position` has 13 values. Each player has one current value, with no history.

### 4.3 StatsBomb Open Data (week-2 module, not the core)

- Same dataset in `statsbomb/open-data` and `hudl/open-data` (identical `competitions.json` and licence). Load with `statsbombpy`.
- Premier League: only 2003/04 (Arsenal matches only) and 2015/16 (all 380 matches). The other 2015/16 top-five leagues are also complete.
- Recent men's data is mostly single-team or tournament data: Bundesliga 2023/24 (34 matches, Leverkusen only), Ligue 1 2021/22 and 2022/23 (PSG only), Euro 2024 (all 51 matches), Copa América 2024, AFCON 2023, World Cup 2022, MLS 2023.
- Lineups have no date of birth.
- Terms (Public Data User Agreement, 8 Sept 2023): no redistribution, no commercial use, and the StatsBomb logo on any published analysis. Sharing analysis publicly is allowed.
- Use: a separate event-data deep dive (event-derived attributes and `mplsoccer` radars on 2015/16 PL and Euro 2024). It does **not** feed the main player cards, because stat definitions differ.

### 4.4 Not used

- **FBref:** advanced Opta stats were removed after Opta ended the feed. Not a source for progressive passes and similar stats.
- **EA FC / FIFA ratings (Kaggle, stefanoleone992):** FIFA 15 to FC 24, data to Sept 2023, scraped from sofifa, licence not confirmed. Optional in week 2 only, as a benchmark in the backtest (FIFA "potential" against how players actually developed), clearly labelled as video-game ratings.

---

## 5. Agreed design decisions

### 5.1 Cohort
- "Current" season = **2025/26** (the last complete season in every source).
- U21 = aged under 21 on **1 Sep 2025**.
- Per-90 stats only for player-seasons with **at least 450 PL minutes**. List players below that threshold separately, without attributes.

### 5.2 Attributes (1 to 20, FM-style)
These are the only attributes the data supports for 2025/26. Do not add others.

| Attribute | Inputs (per 90) |
| --- | --- |
| Finishing | goals; goals minus xG |
| Shot Threat | xG |
| Chance Creation | xA; assists |
| Defending | tackles + clearances/blocks/interceptions |
| Ball Recovery | recoveries |
| Goalkeeping (GK only) | saves; goals conceded minus xGC |

Passing, Progression, Dribbling, Aerial and Pressing are **not available** in any open source for recent PL seasons. The README must say this.

Mapping:
- Compute percentiles within position against **all** PL players at that position with at least 450 minutes (not only U21s), so ratings are absolute, as in FM.
- `attribute = min(20, 1 + floor(percentile * 20))`, with percentile in [0, 1].
- Multi-input attributes: average the input percentiles (state this choice; revisit later).
- Document the mapping in the README.

### 5.3 Current Ability (CA)
Start with transparent, hand-chosen weights per position over the attributes above. Show each attribute's contribution to the score. Learned weights are an optional later comparison.

### 5.4 Potential Ability (PA): the core technical piece
- History: FPL seasons 2016-17 to 2025-26, joined to Transfermarkt for date of birth (seasons before 2024-25) and top-flight minutes outside the PL.
- Primary outcome: **PL minutes share** per season (minutes / (38 x 90)). Secondary outcome: per-90 output.
- Model: `statsmodels` MixedLM, outcome ~ age + age², with a random intercept and random age slope per player. Explain the model and its assumptions to me as you build it (for example, that minutes share is bounded, so a Gaussian model is an approximation).
- Output: projected peak (ages 24 to 26) **with an uncertainty range**, never a single number.

### 5.5 Validation
- Backtest on players who were U21 in earlier seasons, predicting with only the data available at that time. Time-based splits only, no leakage.
- Compare against baselines: "no change" and a population age curve.
- Report error metrics honestly, including if the model doesn't beat the baselines.

### 5.6 Similar players
Standardise per-90 stats within position, then use cosine similarity or kNN for current and historical comparisons. Only compare like with like (same stat set).

### 5.7 App
Streamlit app: player search, FM-style card (attributes, CA, PA range), trajectory chart, similar players, shortlist with filters (age, position, minutes, CA, PA), a short plain-English scouting report. Plotly for charts.

---

## 6. Repository structure

```
football-youth-prospects/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore            (data/, .venv/, __pycache__/, .env)
├── data/
│   ├── raw/              (fpl/, transfermarkt/, statsbomb/; gitignored)
│   ├── processed/        (gitignored)
│   └── manual_matches.csv  (hand-made fixes for player matching; the one exception that is committed)
├── src/
│   ├── config.py         (paths, seasons, thresholds, URLs in one place)
│   ├── load_data.py
│   ├── audit.py          (data-quality checks, writes a report)
│   ├── clean.py
│   ├── matching.py       (FPL to Transfermarkt player matching)
│   ├── cohort.py         (who is in the sample: U21 on a cut-off date, for any season; added on Day 1)
│   ├── features.py       (per-90, percentiles, attributes)
│   ├── ability.py        (CA and PA)
│   ├── trajectory.py     (mixed-effects model, projections)
│   ├── similarity.py
│   └── validate.py       (backtesting)
├── notebooks/            (exploration only)
├── app/streamlit_app.py
└── tests/
```

---

## 7. Timeline

**Day 1: data and profiles** (detailed tasks in section 8)

**Day 2: trajectories and potential**
- Age curves, mixed-effects model, PA projections with uncertainty.
- First backtest against the baselines.
- Similar-player comparisons.

**Day 3: app, write-up, apply**
- Minimal Streamlit app (card, trajectory, similar players, shortlist).
- README for a coach or academy lead: the question, method in plain English, two or three insights, limitations, how to run it.

**Week 2:** StatsBomb deep-dive module, better visuals and backtest, scouting-report text, deployment (Streamlit Community Cloud, with licences in mind), more tests, optional EA FC benchmark.

If Day 1 slips, the Transfermarkt matching (task 4) can move to Day 2 morning, since only the trajectory model needs it.

---

## 8. Day 1 tasks (one at a time, following section 2)

Do these in order. After each task, show me the result and the "Files changed in this step" section, give a suggested commit message, then stop and wait for me. Do not commit.

1. **Scaffold.** Create the structure in section 6, `requirements.txt`, `.gitignore`, and `src/config.py` holding all paths, season lists, thresholds (450 minutes, U21 cut-off date) and URLs.

2. **`load_data.py`.**
   - Download FPL `players_raw.csv` and `gws/merged_gw.csv` for 2016-17 to 2025-26 into `data/raw/fpl/{season}/`.
   - Download Transfermarkt `players`, `appearances`, `games`, `competitions`, `player_valuations`, `clubs` into `data/raw/transfermarkt/`. If the R2 URL fails, print clear instructions for downloading manually from Kaggle.
   - Skip files that already exist unless `--force` is passed. Be polite: sequential requests, a timeout, retries with backoff.
   - Write `data/raw/manifest.json` with source URL, download timestamp, file size and SHA-256 for each file.

3. **`audit.py`.** Write a markdown report to `data/processed/audit_report.md` covering:
   - rows and players per FPL season; which of the fields in section 4.1 are present in each season
   - whether FPL `code` is stable across seasons (spot-check known players across several seasons and report what you find)
   - missing `birth_date` counts
   - for 2025-26: whether summed gameweek minutes, goals, assists match the season totals in `players_raw`
   - Transfermarkt: seasons covered per competition code, date range of `appearances`, position field values
   - duplicates and obvious anomalies

4. **`matching.py`.** Match FPL players (2024-25 and 2025-26 first, where FPL has `birth_date`) to Transfermarkt players on date of birth plus a normalised name (`unidecode`, `rapidfuzz`), using club as a tie-breaker. Apply overrides from `data/manual_matches.csv`. Output the matched table, `data/processed/unmatched_report.csv`, and the match rate. Then extend to earlier FPL seasons via FPL `code` (if the audit confirms it is stable).

5. **Cohort.** Build the 2025/26 U21 cohort using section 5.1. Report the real counts: U21 players with any PL minutes, and with at least 450.

6. **`features.py`.** Per-90 stats, position groups, within-position percentiles against all qualifying PL players, and 1 to 20 attributes per section 5.2. Write `data/processed/player_season_2025_26.parquet` and `data/processed/u21_cards_2025_26.parquet`.

7. **Tests** in `tests/`: per-90 arithmetic, 1 to 20 bounds and monotonicity of the mapping, no duplicate player-seasons, U21 age logic around the cut-off date.

8. **Sanity check.** Print the cards for three U21 players of my choosing (ask me which) so I can judge whether they look right.

End of Day 1: give me a short summary of what exists, what the audit found (especially anything that contradicts section 4), and a combined list of every file created or modified during the day. Do not commit; I will.

---

## 9. Limitations to carry into the README

- No academy or youth-league data: only players who already reached PL minutes (survivorship bias).
- No Championship or other EFL data, so loan spells in England are invisible.
- Limited attribute set: no passing, progression, dribbling, aerial or pressing data for recent PL seasons in open sources.
- Defensive stats only exist from 2025/26, so historical trajectories use minutes and attacking output.
- Small samples and noisy per-90 stats for low-minute players.
- Minutes and output depend on team, manager and role, not just ability.
- "Potential" is a statistical projection with wide uncertainty, not a verdict on a player.
- Data is used for a non-commercial portfolio project under each source's terms. Raw data isn't redistributed.
