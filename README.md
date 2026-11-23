# Premier League U21 Prospects Scout

A Football Manager-style scouting tool for Premier League players aged under 21. It uses open data to build player cards (1 to 20 attributes, a Current Ability score and a Potential Ability range), development trajectories and similar-player comparisons, presented in a Streamlit app.

This is a non-commercial portfolio project and a work in progress. The sections below will grow as each part is built.

## Setting up

Developed with Python 3.14 on Windows.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell (macOS or Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Run everything from the repository root. To check the configuration:

```powershell
python -m src.config
```

## Getting the data

```powershell
python -m src.load_data           # downloads about 108 MB; files already present are skipped
python -m src.load_data --force   # downloads everything again
```

This fetches FPL `players_raw.csv` and `gws/merged_gw.csv` for 2016-17 to 2025-26, plus six Transfermarkt tables, into `data/raw/`. Use `--source fpl` or `--source transfermarkt` to fetch one source only. Each file is recorded in `data/raw/manifest.json` with its source URL, download time, size and SHA-256. If the Transfermarkt download fails, the script prints instructions for getting the same tables from Kaggle by hand.

To check the downloaded files:

```powershell
python -m src.audit
```

This writes `data/processed/audit_report.md`. It compares the data with what this project assumes about it, and lists duplicates and other anomalies.

To match FPL players to Transfermarkt players:

```powershell
python -m src.matching
```

This writes `data/processed/fpl_tm_matches.csv` (one row per matched player), `data/processed/unmatched_report.csv` (everyone else, with the closest Transfermarkt names) and `data/processed/matching_report.md` (match rates and checks). Hand-made fixes go in `data/manual_matches.csv`: one row per FPL `code`, with a Transfermarkt `player_id` to force a match, or a blank one to force no match.

To build the U21 cohort of a season (players aged under 21 on 1 September at the start of the season who played Premier League minutes in it):

```powershell
python -m src.cohort                    # 2025/26: under 21 on 1 September 2025
python -m src.cohort --season 2018-19   # an earlier season: under 21 on 1 September 2018
```

This writes `data/processed/u21_cohort_2025_26.csv` (or the other season's file) and prints the counts. Age is in completed years, so a player born on 1 September 2004 is 21 on the 2025/26 cut-off date and not in the cohort. `--cutoff YYYY-MM-DD` measures age on a different date. For an earlier season, minutes and positions come from that season's files only. Birth dates come from later files where needed, which is safe because a birth date does not change.

To work out per-90 stats and attributes:

```powershell
python -m src.features
```

This writes `data/processed/player_season_2025_26.parquet` (every player in the 2025/26 season) and `data/processed/u21_cards_2025_26.parquet` (the U21 cohort). To print players' cards, with the numbers behind each attribute:

```powershell
python -m src.features --show Diouf Estêvão Kroupi
```

## Running the tests

```powershell
python -m pytest
```

The tests cover the per-90 arithmetic, the 1 to 20 mapping (its bounds, that it never goes down as a percentile rises, and exact boundaries), the U21 age rule around 1 September 2025, and that there is one row per player per season. Tests that check the real data files skip themselves if the data has not been downloaded and processed yet.

## How the attributes are calculated

Every player with at least 450 Premier League minutes in 2025/26 gets Football Manager-style attributes on a 1 to 20 scale. Players with fewer minutes are listed without attributes, because per-90 numbers from a handful of appearances are too noisy.

| Attribute | Who gets it | Inputs (per 90 minutes) |
| --- | --- | --- |
| Finishing | outfield players | goals; goals minus expected goals (xG) |
| Shot Threat | outfield players | expected goals (xG) |
| Chance Creation | outfield players | expected assists (xA); assists |
| Defending | outfield players | tackles plus clearances, blocks and interceptions |
| Ball Recovery | outfield players | recoveries |
| Goalkeeping | goalkeepers | saves; expected goals conceded minus goals conceded |

How a season total becomes an attribute:

1. **Per 90:** the season total divided by minutes played, times 90.
2. **Percentile within position:** the player is compared with every Premier League player in the same FPL position (goalkeeper, defender, midfielder or forward) who also played at least 450 minutes, not only with under-21s. Ratings are therefore absolute, as in Football Manager: a 15 means better than about 70% of the league's players in that position this season. The percentile is the share of the group below the player plus half the share level with them, so players with the same number get the same percentile. In 2025/26 the groups are 28 goalkeepers, 147 defenders, 181 midfielders and 44 forwards.
3. **Attribute:** 1 + (percentile x 20, rounded down), capped at 20. The bottom 5% of a group get 1 and the top 5% get 20.
4. **Two inputs:** the attribute uses the average of the two percentiles.

Limits of this approach:

- Passing, progression, dribbling, aerial ability and pressing are not in any open data source for recent Premier League seasons, so there are no attributes for them.
- FPL has only four positions, so a holding midfielder is compared with attacking midfielders, and a winger that FPL lists as a forward is compared with strikers.
- Attributes are relative to this season's Premier League players, and 450 minutes (five full matches) is still a small sample.
- Saves per 90 partly reflects how busy a goalkeeper is, not only how good.
- Many players share the same small totals (most defenders score no goals), which bunches them on the same attribute value.

## Repository layout

```
├── README.md
├── requirements.txt
├── pytest.ini               test settings
├── data/                    gitignored, apart from manual_matches.csv
│   ├── raw/                 downloaded files (fpl/, transfermarkt/, statsbomb/)
│   ├── processed/           pipeline outputs
│   └── manual_matches.csv   hand-made fixes for player matching
├── src/
│   ├── config.py            paths, seasons, thresholds and URLs
│   ├── load_data.py         downloads the raw files
│   ├── audit.py             data-quality checks
│   ├── clean.py             shared cleaning steps
│   ├── matching.py          FPL to Transfermarkt player matching
│   ├── cohort.py            the U21 cohort of any season (who is in the sample)
│   ├── features.py          per-90 stats, percentiles and attributes
│   ├── ability.py           Current and Potential Ability
│   ├── trajectory.py        mixed-effects age-curve model
│   ├── similarity.py        similar-player comparisons
│   └── validate.py          backtesting
├── app/streamlit_app.py     the app
├── notebooks/               exploration only
└── tests/
```

## Data sources and terms

No data is included in this repository. The pipeline downloads it into `data/`, which is gitignored apart from `data/manual_matches.csv`, a small file of hand-made matching fixes.

- **Fantasy Premier League**, via the [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) archive. The archive's code is MIT licensed, but the data belongs to the Premier League, whose terms allow private and personal use and prohibit commercial use. It is used here for a non-commercial project and the raw data is not redistributed.
- **Transfermarkt**, via [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) (also on Kaggle as `davidcariboo/player-scores`). The repository is released under CC0, but the data was collected from Transfermarkt, so it is used for non-commercial purposes only and is not redistributed. Credit to Transfermarkt and to the dataset's author.
- **StatsBomb Open Data** (planned as a separate module), under the StatsBomb Public Data User Agreement.

Please check each source's terms yourself before reusing anything from this project.

## Known limitations

- Only players who have already played Premier League minutes are covered. There is no academy or youth-league data, so the sample has survivorship bias.
- There is no Championship or other EFL data, so loan spells in England are invisible.
- The attribute set is limited: open sources have no passing, progression, dribbling, aerial or pressing data for recent Premier League seasons.
- Defensive stats only exist from 2025/26, so historical trajectories use minutes and attacking output.
- Per-90 stats are noisy for players with few minutes.
- Minutes and output depend on team, manager and role, not just ability.
- "Potential" is a statistical projection with wide uncertainty, not a verdict on a player.
