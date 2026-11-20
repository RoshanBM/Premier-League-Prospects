"""Download the raw FPL and Transfermarkt files into data/raw/, and read them back.

What it downloads:
- FPL: ``players_raw.csv`` and ``gws/merged_gw.csv`` for every season in
  ``config.FPL_SEASONS``, from the vaastav/Fantasy-Premier-League archive.
- Transfermarkt: the tables in ``config.TRANSFERMARKT_TABLES`` as ``.csv.gz``,
  from the download location in the dcaribou/transfermarkt-datasets README.
  If that fails, it prints instructions for getting the same tables from
  Kaggle by hand.

How it behaves:
- Files that already exist are skipped unless ``--force`` is passed. A
  Transfermarkt table copied in by hand as plain ``.csv`` also counts.
- Requests are made one at a time, with a pause between downloads, a timeout,
  and up to three retries with exponential backoff for temporary failures
  (dropped connections, timeouts, rate limiting, server errors). Errors that
  retrying will not fix, such as a 404, fail straight away.
- Each file is streamed into a temporary ``.part`` file, then renamed into
  place only once it is complete and has passed a basic content check. An
  interrupted run therefore never leaves a half-written file for a later run
  to skip.
- ``data/raw/manifest.json`` records every file (source URL, time in UTC, size
  in bytes and SHA-256). It is saved after each file and at the end of a run.

Reading the files back: ``read_fpl_players``, ``read_fpl_gameweeks`` and
``read_transfermarkt`` return the raw tables unchanged, decoded with the right
text encoding, for the audit and the later pipeline steps.

Usage, from the repository root:
    python -m src.load_data                          # everything
    python -m src.load_data --source fpl             # FPL only
    python -m src.load_data --source transfermarkt   # Transfermarkt only
    python -m src.load_data --force                  # download existing files again
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src import config

# Files are written and hashed in 1 MiB pieces, so a large file is never held
# in memory all at once.
CHUNK_BYTES: int = 1024 * 1024

# HTTP statuses worth retrying: rate limiting and temporary server problems.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Network errors worth retrying. ChunkedEncodingError covers a connection that
# drops part-way through a download.
RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

# Every gzip file starts with these two bytes.
GZIP_SIGNATURE: bytes = b"\x1f\x8b"
# Some CSV files start with a byte-order mark before the header.
UTF8_BOM: bytes = b"\xef\xbb\xbf"

MANIFEST_NOTE: str = (
    "Raw files fetched by src/load_data.py. The data belongs to its sources "
    "(the Premier League, via the vaastav/Fantasy-Premier-League archive; "
    "Transfermarkt, via dcaribou/transfermarkt-datasets) and must not be "
    "redistributed."
)

# One manifest entry per file, keyed by its path relative to data/raw/.
Manifest = dict[str, dict]


class DownloadError(Exception):
    """A file could not be downloaded, so nothing was saved for it."""


class RetryableDownloadError(DownloadError):
    """A temporary failure, such as a timeout, that is worth retrying."""


@dataclass(frozen=True)
class RemoteFile:
    """One file to fetch: where it comes from and where it is saved.

    ``manual_copy`` is an optional second local file that also counts as
    already having the file (a Transfermarkt table downloaded by hand from
    Kaggle), and ``manual_source`` records where such a copy comes from.
    """

    url: str
    destination: Path
    manual_copy: Path | None = None
    manual_source: str | None = None


@dataclass(frozen=True)
class FileRecord:
    """One manifest entry.

    ``downloaded_at`` is the UTC download time, or the file's last-modified
    time for a file that was already on disk (``method == "found on disk"``).
    """

    source_url: str
    method: str
    downloaded_at: str
    size_bytes: int
    sha256: str


@dataclass
class FetchResult:
    """What happened to one batch of files."""

    downloaded: int = 0
    already_present: int = 0
    failed: list[RemoteFile] = field(default_factory=list)


# ---------------------------------------------------------------------------
# What to fetch
# ---------------------------------------------------------------------------

def fpl_files() -> list[RemoteFile]:
    """List the FPL files to fetch: each file in ``FPL_SEASON_FILES`` for each season."""
    return [
        RemoteFile(url=config.fpl_url(season, name), destination=config.fpl_raw_path(season, name))
        for season in config.FPL_SEASONS
        for name in config.FPL_SEASON_FILES
    ]


def transfermarkt_files() -> list[RemoteFile]:
    """List the Transfermarkt tables to fetch, each accepting a manual Kaggle copy."""
    return [
        RemoteFile(
            url=config.transfermarkt_url(table),
            destination=config.transfermarkt_raw_path(table),
            manual_copy=config.transfermarkt_manual_path(table),
            manual_source=config.TRANSFERMARKT_KAGGLE_URL,
        )
        for table in config.TRANSFERMARKT_TABLES
    ]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string, to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def manifest_key(path: Path) -> str:
    """Return a path relative to data/raw/ with forward slashes, e.g. ``fpl/2025-26/players_raw.csv``."""
    return path.relative_to(config.RAW_DIR).as_posix()


def format_size(size_bytes: int) -> str:
    """Return a size in megabytes, e.g. ``4.6 MB``."""
    return f"{size_bytes / 1_000_000:.1f} MB"


def sha256_of_file(path: Path) -> str:
    """Return the SHA-256 of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_error(error: BaseException) -> str:
    """Return a short, readable description of a network error.

    requests wraps lower-level errors inside each other, which makes their
    text repeat itself, so this reports the innermost message only.
    """
    detail = error
    while detail.args and isinstance(detail.args[0], BaseException):
        detail = detail.args[0]
    text = str(detail.args[0]) if detail.args else str(detail)
    return f"{type(error).__name__}: {text[:150]}"


def check_content(path: Path, final_name: str) -> None:
    """Raise DownloadError if a downloaded file is clearly not what was asked for.

    This is a light check, not validation. A ``.gz`` file must start with the
    gzip signature, and a ``.csv`` file must start with a header line that
    contains commas (rather than, say, an HTML error page).
    """
    with path.open("rb") as handle:
        start = handle.read(64 * 1024)
    if not start:
        raise DownloadError("the file is empty")
    if final_name.endswith(".gz") and not start.startswith(GZIP_SIGNATURE):
        raise DownloadError("expected a gzip file, but the content is not gzip")
    if final_name.endswith(".csv"):
        header = start.removeprefix(UTF8_BOM).split(b"\n", 1)[0]
        if header.lstrip().startswith(b"<") or b"," not in header:
            raise DownloadError("expected a CSV file, but the first line is not a CSV header")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest() -> Manifest:
    """Return the manifest entries for files that still exist on disk.

    Entries for files deleted since the last run are dropped, so the manifest
    only describes files that are actually there.
    """
    if not config.MANIFEST_PATH.exists():
        return {}
    entries: Manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))["files"]
    return {key: entry for key, entry in entries.items() if (config.RAW_DIR / key).exists()}


def save_manifest(manifest: Manifest) -> None:
    """Write the manifest, replacing the old one in a single step."""
    payload = {
        "note": MANIFEST_NOTE,
        "updated_at": utc_now(),
        "files": dict(sorted(manifest.items())),
    }
    temp_path = config.MANIFEST_PATH.with_name(config.MANIFEST_PATH.name + ".part")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, config.MANIFEST_PATH)


# ---------------------------------------------------------------------------
# Files already on disk
# ---------------------------------------------------------------------------

def find_local_copy(remote: RemoteFile) -> tuple[Path, str] | None:
    """Return ``(path, source)`` for an existing local copy, or None if there is none."""
    if remote.destination.exists():
        return remote.destination, remote.url
    if remote.manual_copy is not None and remote.manual_copy.exists():
        return remote.manual_copy, remote.manual_source or "copied in by hand"
    return None


def record_local_file(path: Path, source: str) -> FileRecord:
    """Build a manifest record for a file that is already on disk (hashes the whole file)."""
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return FileRecord(
        source_url=source,
        method="found on disk",
        downloaded_at=modified.isoformat(timespec="seconds"),
        size_bytes=stat.st_size,
        sha256=sha256_of_file(path),
    )


def note_existing_file(path: Path, source: str, manifest: Manifest) -> str:
    """Make sure a file already on disk has an up-to-date manifest entry.

    Returns a short message for the progress output. A file with no entry, or
    whose size no longer matches its entry (for example, replaced by hand),
    gets a fresh entry.
    """
    key = manifest_key(path)
    entry = manifest.get(key)
    if entry is not None and entry.get("size_bytes") == path.stat().st_size:
        return "already present, skipped"
    manifest[key] = asdict(record_local_file(path, source))
    save_manifest(manifest)
    if entry is None:
        return f"already present as {path.name}, added to the manifest"
    return f"already present as {path.name} but its size has changed, manifest entry updated"


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    """Return an HTTP session with a generic client name and no on-the-fly compression.

    Asking for ``identity`` encoding means the bytes received are exactly the
    published file, so their size can be checked against Content-Length.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": config.USER_AGENT, "Accept-Encoding": "identity"})
    return session


def download_once(session: requests.Session, url: str, temp_path: Path) -> tuple[int, str]:
    """Make one attempt to stream ``url`` into ``temp_path``.

    Returns the number of bytes written and their SHA-256. Raises
    RetryableDownloadError for temporary problems and DownloadError for
    problems that retrying will not fix.
    """
    digest = hashlib.sha256()
    size = 0
    expected_size: str | None = None
    try:
        with session.get(url, stream=True, timeout=config.REQUEST_TIMEOUT_SECONDS) as response:
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise RetryableDownloadError(f"HTTP {response.status_code}")
            if response.status_code != 200:
                raise DownloadError(f"HTTP {response.status_code}")
            encoding = response.headers.get("Content-Encoding", "identity").lower()
            if encoding != "identity":
                # We asked for the file exactly as published. If a server
                # compresses it anyway, requests unpacks it on the fly, so a
                # .csv.gz would be saved as plain text and the size check
                # below would no longer mean anything.
                raise DownloadError(f"the server compressed the response ({encoding})")
            expected_size = response.headers.get("Content-Length")
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
    except RETRYABLE_EXCEPTIONS as error:
        raise RetryableDownloadError(describe_error(error)) from error
    if expected_size is not None and size != int(expected_size):
        raise RetryableDownloadError(f"incomplete download, {size} of {expected_size} bytes")
    return size, digest.hexdigest()


def download_with_retries(session: requests.Session, url: str, temp_path: Path) -> tuple[int, str]:
    """Call ``download_once``, retrying temporary failures with exponential backoff.

    With the default settings that is up to 4 attempts, with waits of 2, 4 and
    8 seconds in between.
    """
    for attempt in range(1, config.REQUEST_MAX_RETRIES + 1):
        try:
            return download_once(session, url, temp_path)
        except RetryableDownloadError as error:
            wait = config.REQUEST_BACKOFF_SECONDS * 2 ** (attempt - 1)
            print(f"      attempt {attempt} failed for {url} ({error}); retrying in {wait:g} s")
            time.sleep(wait)
    try:
        return download_once(session, url, temp_path)  # the last attempt
    except RetryableDownloadError as error:
        attempts = config.REQUEST_MAX_RETRIES + 1
        raise DownloadError(f"{error}; gave up after {attempts} attempts") from error


def download_file(session: requests.Session, remote: RemoteFile) -> FileRecord:
    """Download one file, with retries, and return its manifest record.

    The download goes to ``<name>.part`` and is renamed into place only after
    it is complete and has passed ``check_content``. If anything fails, the
    ``.part`` file is removed and any existing file at the destination is left
    as it was.
    """
    remote.destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = remote.destination.with_name(remote.destination.name + ".part")
    try:
        size, sha256 = download_with_retries(session, remote.url, temp_path)
        check_content(temp_path, remote.destination.name)
        os.replace(temp_path, remote.destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return FileRecord(
        source_url=remote.url,
        method="downloaded",
        downloaded_at=utc_now(),
        size_bytes=size,
        sha256=sha256,
    )


def fetch_all(
    session: requests.Session,
    files: list[RemoteFile],
    manifest: Manifest,
    force: bool,
) -> FetchResult:
    """Fetch or skip each file in turn, keeping the manifest up to date."""
    result = FetchResult()
    made_request = False
    for number, remote in enumerate(files, start=1):
        label = f"  [{number:>2}/{len(files)}] {manifest_key(remote.destination)}"
        local_copy = None if force else find_local_copy(remote)
        if local_copy is not None:
            print(f"{label}: {note_existing_file(*local_copy, manifest)}")
            result.already_present += 1
            continue
        if made_request:
            time.sleep(config.REQUEST_PAUSE_SECONDS)  # a polite pause between requests
        made_request = True
        try:
            record = download_file(session, remote)
        except DownloadError as error:
            print(f"{label}: FAILED ({error})")
            result.failed.append(remote)
            continue
        manifest[manifest_key(remote.destination)] = asdict(record)
        save_manifest(manifest)
        print(f"{label}: downloaded ({format_size(record.size_bytes)})")
        result.downloaded += 1
    return result


# ---------------------------------------------------------------------------
# Reading the raw files back
# ---------------------------------------------------------------------------

def detect_encoding(path: Path) -> str:
    """Return ``"utf-8"`` if the whole file is valid UTF-8, otherwise ``"latin-1"``.

    The FPL gameweek files for 2016-17 to 2018-19 are not UTF-8: accented
    letters are stored as single Latin-1 bytes. Latin-1 can decode any byte,
    so it is a safe fallback, and the audit reports which files needed it.
    """
    try:
        path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return "latin-1"
    return "utf-8"


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read a raw CSV file exactly as it is, in its own text encoding."""
    return pd.read_csv(path, encoding=detect_encoding(path), low_memory=False)


def read_fpl_players(season: str) -> pd.DataFrame:
    """Read one season's ``players_raw.csv``: one row per player in the FPL game."""
    return read_raw_csv(config.fpl_raw_path(season, "players_raw.csv"))


def read_fpl_gameweeks(season: str) -> pd.DataFrame:
    """Read one season's ``gws/merged_gw.csv``: one row per player per fixture."""
    return read_raw_csv(config.fpl_raw_path(season, "gws/merged_gw.csv"))


def find_transfermarkt_file(table: str) -> Path:
    """Return the local file for a Transfermarkt table, preferring the ``.csv.gz`` download."""
    for path in (config.transfermarkt_raw_path(table), config.transfermarkt_manual_path(table)):
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No local copy of the Transfermarkt table {table!r}. "
        "Run: python -m src.load_data --source transfermarkt"
    )


def read_transfermarkt(table: str) -> pd.DataFrame:
    """Read one Transfermarkt table exactly as it is (pandas unpacks ``.csv.gz`` itself)."""
    return pd.read_csv(find_transfermarkt_file(table), low_memory=False)


# ---------------------------------------------------------------------------
# Reporting and command line
# ---------------------------------------------------------------------------

def kaggle_instructions(failed: list[RemoteFile]) -> str:
    """Return step-by-step instructions for getting Transfermarkt tables by hand."""
    tables = [remote.destination.name.removesuffix(".csv.gz") for remote in failed]
    lines = [
        "",
        "The Transfermarkt download did not work for: " + ", ".join(tables),
        "You can get the same tables from Kaggle instead:",
        f"  1. Sign in to Kaggle and open {config.TRANSFERMARKT_KAGGLE_URL}",
        "  2. Download the dataset and unzip it.",
        f"  3. Copy these files into {config.TRANSFERMARKT_RAW_DIR}",
        "       " + ", ".join(f"{table}.csv" for table in tables),
        "     Plain .csv files are fine, and so are .csv.gz files.",
        "  4. Run this again: python -m src.load_data --source transfermarkt",
        "     It finds the files, skips the download and records them in the manifest.",
    ]
    return "\n".join(lines)


def print_summary(results: list[FetchResult], manifest: Manifest) -> None:
    """Print the totals for this run and for the manifest as a whole."""
    downloaded = sum(result.downloaded for result in results)
    present = sum(result.already_present for result in results)
    failed = [remote for result in results for remote in result.failed]
    total_bytes = sum(entry["size_bytes"] for entry in manifest.values())
    print(f"\nThis run: {downloaded} downloaded, {present} already present, {len(failed)} failed.")
    for remote in failed:
        print(f"  failed: {remote.url}")
    print(
        f"Manifest: {config.MANIFEST_PATH} "
        f"({len(manifest)} files, {format_size(total_bytes)} in total)"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(
        description="Download the raw FPL and Transfermarkt files into data/raw/."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="download files again even if they already exist",
    )
    parser.add_argument(
        "--source",
        choices=("all", "fpl", "transfermarkt"),
        default="all",
        help="which source to download (default: all)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the downloader. Returns 0 if every requested file is present afterwards, else 1."""
    args = parse_args(argv)
    config.ensure_directories()
    manifest = load_manifest()
    session = make_session()
    results: list[FetchResult] = []

    if args.source in ("all", "fpl"):
        print(
            f"FPL archive: {len(config.FPL_SEASONS)} seasons "
            f"({config.FPL_SEASONS[0]} to {config.FPL_SEASONS[-1]})"
        )
        results.append(fetch_all(session, fpl_files(), manifest, args.force))

    if args.source in ("all", "transfermarkt"):
        print(f"\nTransfermarkt: {len(config.TRANSFERMARKT_TABLES)} tables")
        transfermarkt = fetch_all(session, transfermarkt_files(), manifest, args.force)
        results.append(transfermarkt)
        if transfermarkt.failed:
            print(kaggle_instructions(transfermarkt.failed))

    save_manifest(manifest)
    print_summary(results, manifest)
    return 1 if any(result.failed for result in results) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Files finished so far are kept and recorded in the manifest.")
        sys.exit(130)
