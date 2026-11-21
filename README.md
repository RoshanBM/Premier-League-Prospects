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

To build the 2025/26 U21 cohort (players aged under 21 on 1 September 2025 who played Premier League minutes that season):

```powershell
python -m src.cohort
```

This writes `data/processed/u21_cohort_2025_26.csv` and prints the counts. Age is in completed years, so a player born on 1 September 2004 is 21 on the cut-off date and not in the cohort.

## Repository layout

```
├── README.md
├── requirements.txt
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
