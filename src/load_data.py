"""Download the raw FPL and Transfermarkt files into data/raw/.

Planned for Day 1, task 2:
- FPL ``players_raw.csv`` and ``gws/merged_gw.csv`` for every season in
  ``config.FPL_SEASONS``.
- The Transfermarkt tables in ``config.TRANSFERMARKT_TABLES``, with clear
  instructions for downloading them from Kaggle if the direct link fails.
- Existing files are skipped unless ``--force`` is passed. Requests are made
  one at a time, with a timeout and retries with backoff.
- Every file is recorded in ``data/raw/manifest.json`` (source URL, download
  time, size and SHA-256).
"""
