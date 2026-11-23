"""Tests for run_pipeline.py: stage order, --from and --skip-download, and how a failing stage is reported.

The stages here are fake modules, so nothing in data/ is read or written.
"""

import sys
import types

import pytest

import run_pipeline
from run_pipeline import STAGE_NAMES, Stage, run_stage, stages_to_run, status_line


def test_stages_are_in_the_dependency_order():
    assert STAGE_NAMES == ("load", "audit", "matching", "cohort", "features", "panel", "trajectory", "validate",
                           "ability", "similarity", "figures")


def test_validate_runs_before_ability_and_ability_before_figures():
    # ability needs the calibration factor from validate; figures needs the PA table from ability.
    assert STAGE_NAMES.index("validate") < STAGE_NAMES.index("ability") < STAGE_NAMES.index("figures")


def test_every_stage_module_has_a_main_function():
    import importlib

    for stage in run_pipeline.STAGES:
        assert callable(importlib.import_module(stage.module).main), stage.module


def test_all_stages_by_default():
    assert [stage.name for stage in stages_to_run(None, skip_download=False)] == list(STAGE_NAMES)


def test_skip_download_drops_only_the_load_stage():
    assert [stage.name for stage in stages_to_run(None, skip_download=True)] == list(STAGE_NAMES[1:])


def test_from_starts_at_the_named_stage():
    assert [stage.name for stage in stages_to_run("validate", skip_download=False)] == [
        "validate", "ability", "similarity", "figures"]


def test_from_an_unknown_stage_is_an_error():
    with pytest.raises(ValueError, match="unknown stage"):
        stages_to_run("nonsense", skip_download=True)


@pytest.fixture
def fake_module(monkeypatch):
    """Register a throwaway module whose main() does whatever the test sets."""
    module = types.ModuleType("fake_stage_module")
    monkeypatch.setitem(sys.modules, "fake_stage_module", module)
    return module


def test_a_successful_stage(fake_module, tmp_path):
    fake_module.main = lambda: 0
    with (tmp_path / "log.txt").open("w", encoding="utf-8") as log:
        succeeded, seconds, reason = run_stage(Stage("fake", "fake_stage_module", "fake"), log)
    assert succeeded and reason == "" and seconds >= 0


def test_a_stage_that_raises_is_reported_and_its_traceback_logged(fake_module, tmp_path):
    def broken() -> int:
        raise FileNotFoundError("missing input")

    fake_module.main = broken
    path = tmp_path / "log.txt"
    with path.open("w", encoding="utf-8") as log:
        succeeded, _, reason = run_stage(Stage("fake", "fake_stage_module", "fake"), log)
    assert not succeeded
    assert reason == "FileNotFoundError: missing input"
    assert "Traceback" in path.read_text(encoding="utf-8")


def test_a_non_zero_exit_code_is_a_failure(fake_module, tmp_path):
    fake_module.main = lambda: 1
    with (tmp_path / "log.txt").open("w", encoding="utf-8") as log:
        succeeded, _, reason = run_stage(Stage("fake", "fake_stage_module", "fake"), log)
    assert not succeeded and "exit code 1" in reason


def test_stages_that_take_arguments_get_an_empty_list(fake_module, tmp_path):
    received = []
    fake_module.main = lambda argv: received.append(argv) or 0
    with (tmp_path / "log.txt").open("w", encoding="utf-8") as log:
        run_stage(Stage("fake", "fake_stage_module", "fake", takes_argv=True), log)
    assert received == [[]]


def test_stage_output_goes_to_the_log_not_the_screen(fake_module, tmp_path, capsys):
    fake_module.main = lambda: print("stage chatter") or 0
    path = tmp_path / "log.txt"
    with path.open("w", encoding="utf-8") as log:
        run_stage(Stage("fake", "fake_stage_module", "fake"), log)
    assert "stage chatter" not in capsys.readouterr().out
    assert "stage chatter" in path.read_text(encoding="utf-8")


def test_status_line_names_the_stage_and_time():
    line = status_line(3, 10, Stage("matching", "src.matching", "FPL to Transfermarkt matching"), "OK", 12.34)
    assert line.startswith("[ 3/10] matching") and "OK" in line and "12.3 s" in line
