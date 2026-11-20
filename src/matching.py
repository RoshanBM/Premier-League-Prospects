"""Match FPL players to Transfermarkt players.

Planned for Day 1, task 4: match on date of birth plus a normalised name
(``unidecode`` and ``rapidfuzz``), with club as a tie-breaker, then apply the
hand-made fixes in ``data/manual_matches.csv``. Writes the matched table and
``data/processed/unmatched_report.csv``.
"""
