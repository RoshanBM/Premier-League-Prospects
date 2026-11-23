"""Run the whole pipeline in order with one command, from the raw data to the charts.

Stages, in order (each is a call to the module's main function, not a shell
command):

    load        download the raw FPL and Transfermarkt files     (src/load_data.py)
    audit       data-quality checks and report                    (src/audit.py)
    matching    FPL to Transfermarkt player matching              (src/matching.py)
    cohort      the 2025/26 U21 cohort                            (src/cohort.py)
    features    per-90 stats and 1 to 20 attributes               (src/features.py)
    panel       one row per player per PL season, 2016-17 on      (src/panel.py)
    trajectory  the mixed-effects age curve                       (src/trajectory.py)
    validate    backtest, baselines and interval calibration      (src/validate.py)
    ability     Current Ability and Potential Ability             (src/ability.py)
    similarity  current and historical similar players           (src/similarity.py)
    figures     trajectory charts for a few example players       (src/figures.py)

The order matters: ability needs the calibration factor that validate writes,
and figures needs the PA table that ability writes.

Each stage prints one status line with its time. What the stage itself prints
goes to data/processed/pipeline_log.txt (gitignored), so the console stays
readable; --verbose shows it on screen instead. If a stage fails, the run stops
and says which stage failed, why, and how to resume.

Usage, from the repository root:
    python run_pipeline.py                        # everything, downloading missing raw files
    python run_pipeline.py --skip-download        # use the raw data already in data/raw/
    python run_pipeline.py --from trajectory      # resume from a stage (earlier outputs must exist)
    python run_pipeline.py --skip-download --verbose
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import sys
import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

import pandas as pd

from src import config

LOG_PATH: Path = config.PROCESSED_DIR / "pipeline_log.txt"
# Lines of a failed stage's own output to repeat on screen.
FAILURE_TAIL_LINES: int = 15


@dataclass(frozen=True)
class Stage:
    """One pipeline stage: a module whose main function does the work.

    Attributes:
        name: the name used on the command line (--from).
        module: the module to import, such as "src.audit".
        description: a few words for the status line.
        takes_argv: True if main() takes an argument list. It is then called
            with an empty list, so it does not read this script's own options.
    """

    name: str
    module: str
    description: str
    takes_argv: bool = False

    def run(self) -> int:
        """Import the module and call its main function. Returns its exit code (0 means success)."""
        main = importlib.import_module(self.module).main
        result = main([]) if self.takes_argv else main()
        return 0 if result is None else int(result)


STAGES: tuple[Stage, ...] = (
    Stage("load", "src.load_data", "download raw files", takes_argv=True),
    Stage("audit", "src.audit", "data-quality report"),
    Stage("matching", "src.matching", "FPL to Transfermarkt matching"),
    Stage("cohort", "src.cohort", "2025/26 U21 cohort", takes_argv=True),
    Stage("features", "src.features", "per-90 stats and attributes", takes_argv=True),
    Stage("panel", "src.panel", "multi-season player panel"),
    Stage("trajectory", "src.trajectory", "mixed-effects age curve"),
    Stage("validate", "src.validate", "backtest and calibration"),
    Stage("ability", "src.ability", "Current and Potential Ability"),
    Stage("similarity", "src.similarity", "similar players"),
    Stage("figures", "src.figures", "trajectory charts", takes_argv=True),
)
STAGE_NAMES: tuple[str, ...] = tuple(stage.name for stage in STAGES)

# Main outputs, counted at the end of a run so a rerun can be checked at a glance.
KEY_OUTPUTS: tuple[Path, ...] = (
    config.MATCHES_PATH,
    config.U21_COHORT_PATH,
    config.PLAYER_SEASON_PATH,
    config.U21_CARDS_PATH,
    config.PANEL_PATH,
    config.BACKTEST_RESULTS_PATH,
    config.BACKTEST_PLAYERS_PATH,
    config.CA_PATH,
    config.PA_PROJECTIONS_PATH,
    config.SIMILAR_PLAYERS_PATH,
)


# ---------------------------------------------------------------------------
# Choosing what to run
# ---------------------------------------------------------------------------

def stages_to_run(start: str | None, skip_download: bool) -> list[Stage]:
    """The stages to run, in order: from `start` (or the first) to the end, without load if skip_download.

    Raises:
        ValueError: if `start` is not a stage name.
    """
    if start is not None and start not in STAGE_NAMES:
        raise ValueError(f"unknown stage {start!r}; choose one of: {', '.join(STAGE_NAMES)}")
    first = STAGE_NAMES.index(start) if start is not None else 0
    chosen = list(STAGES[first:])
    if skip_download:
        chosen = [stage for stage in chosen if stage.name != "load"]
    return chosen


def missing_raw_files() -> list[Path]:
    """Raw files the pipeline needs that are not on disk (a hand-copied Transfermarkt .csv counts)."""
    from src.load_data import find_local_copy, fpl_files, transfermarkt_files

    return [remote.destination for remote in [*fpl_files(), *transfermarkt_files()] if find_local_copy(remote) is None]


# ---------------------------------------------------------------------------
# Running one stage
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def stage_output(log: TextIO | None) -> Iterator[None]:
    """Send everything a stage prints (including warnings) to the log file, or leave it on screen if log is None."""
    if log is None:
        yield
        return
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        yield


def log_tail(path: Path, lines: int = FAILURE_TAIL_LINES) -> str:
    """The last few lines of the log file, indented, for a failure message."""
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").rstrip().splitlines()
    return "\n".join(f"    {line}" for line in text[-lines:])


def run_stage(stage: Stage, log: TextIO | None) -> tuple[bool, float, str]:
    """Run one stage and time it.

    Returns:
        (succeeded, seconds taken, a short reason if it failed).
    """
    started = time.perf_counter()
    reason = ""
    with stage_output(log):
        if log is not None:
            print(f"\n{'=' * 80}\n{stage.name}: {stage.description} ({datetime.now():%Y-%m-%d %H:%M:%S})\n{'=' * 80}")
        try:
            code = stage.run()
            if code != 0:
                reason = f"{stage.module}.main() returned exit code {code}"
        except SystemExit as stop:  # a stage that calls sys.exit() itself
            if stop.code not in (0, None):
                reason = f"{stage.module} exited with code {stop.code}"
        except Exception as error:  # report any failure clearly, then stop the run
            traceback.print_exc()
            reason = f"{type(error).__name__}: {error}"
        if log is not None:
            log.flush()
    return reason == "", time.perf_counter() - started, reason


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def status_line(number: int, total: int, stage: Stage, status: str, seconds: float | None = None) -> str:
    """One line per stage, for example "[ 3/10] matching    OK       12.3 s  FPL to Transfermarkt matching"."""
    timing = f"{seconds:7.1f} s" if seconds is not None else " " * 9
    return f"[{number:2d}/{total}] {stage.name:<11} {status:<8} {timing}  {stage.description}"


def count_rows(path: Path) -> str:
    """Row count of a parquet or CSV output, or "missing"."""
    if not path.exists():
        return "missing"
    table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return f"{len(table):,} rows"


def output_summary() -> str:
    """Row counts of the main outputs."""
    width = max(len(path.name) for path in KEY_OUTPUTS)
    return "\n".join(f"  {path.name:<{width}}  {count_rows(path)}" for path in KEY_OUTPUTS)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(
        description="Run the whole pipeline, from the raw data to the app's inputs.",
        epilog="Stages, in order: " + ", ".join(STAGE_NAMES),
    )
    parser.add_argument("--from", dest="start", choices=STAGE_NAMES, metavar="STAGE",
                        help="start from this stage; the outputs of earlier stages must already exist")
    parser.add_argument("--skip-download", action="store_true",
                        help="do not run the load stage; use the raw files already in data/raw/")
    parser.add_argument("--verbose", action="store_true",
                        help=f"show each stage's own output on screen instead of writing it to {LOG_PATH.name}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the chosen stages in order. Returns 0 if every stage succeeded, else 1."""
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # player names may not fit the console's encoding
    config.ensure_directories()
    stages = stages_to_run(args.start, args.skip_download)
    runs_load = any(stage.name == "load" for stage in stages)

    if not runs_load:
        missing = missing_raw_files()
        if missing:
            print(f"Stopped: {len(missing)} raw file(s) are missing, for example {missing[0]}.")
            print("Run python run_pipeline.py without --skip-download (and from the start) to download them.")
            return 1
        print("Raw data: every FPL and Transfermarkt file is present; the load stage is not run.")

    print(f"Running {len(stages)} stage(s): {', '.join(stage.name for stage in stages)}")
    if not args.verbose:
        print(f"Each stage's own output goes to {LOG_PATH}")
    total_started = time.perf_counter()
    log_file = None if args.verbose else LOG_PATH.open("w", encoding="utf-8", errors="replace")
    try:
        # On an interactive console, show "running" and overwrite it with the result.
        show_running = sys.stdout.isatty() and not args.verbose
        for number, stage in enumerate(stages, start=1):
            if show_running:
                print(status_line(number, len(stages), stage, "running"), end="\r", flush=True)
            succeeded, seconds, reason = run_stage(stage, log_file)
            print(status_line(number, len(stages), stage, "OK" if succeeded else "FAILED", seconds))
            if not succeeded:
                print(f"\nStopped: stage '{stage.name}' failed. {reason}")
                if log_file is not None:
                    print(f"Last lines of its output ({LOG_PATH}):\n{log_tail(LOG_PATH)}")
                print(f"\nFix the problem, then resume with: python run_pipeline.py --from {stage.name}"
                      + (" --skip-download" if args.skip_download else ""))
                return 1
    finally:
        if log_file is not None:
            log_file.close()

    print(f"\nAll {len(stages)} stage(s) finished in {time.perf_counter() - total_started:.1f} s.")
    print(f"Main outputs in {config.PROCESSED_DIR}:\n{output_summary()}")
    print("\nNext: streamlit run app/streamlit_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
