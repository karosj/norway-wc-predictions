"""
Orchestrate data collection: produce all four canonical input files (spec 2A-2D).

* matches.csv / elo_ratings.csv  <- eloratings.net (cached TSVs)
* fifa_rankings.csv              <- supplied year-end ranks
* flashscore_stats.csv          <- raw Flashscore extraction

Idempotent: existing inputs are reused unless ``force=True``. Network is only
needed the first time the eloratings cache is populated.
"""

from __future__ import annotations

from . import collect_elo, prepare_fifa, prepare_flashscore
from ..utils.config import ELO_CSV, FIFA_CSV, FLASHSCORE_CSV, MATCHES_CSV
from ..utils.logging_utils import get_logger

log = get_logger(__name__)


def build_all(force: bool = False) -> None:
    """Build every input CSV. Skips files that already exist unless ``force``."""
    if force or not (MATCHES_CSV.exists() and ELO_CSV.exists()):
        log.info("collecting match results + Elo from eloratings.net ...")
        collect_elo.collect()
    else:
        log.info("matches.csv + elo_ratings.csv present - skipping eloratings")

    if force or not FIFA_CSV.exists():
        prepare_fifa.build()
    else:
        log.info("fifa_rankings.csv present - skipping")

    if force or not FLASHSCORE_CSV.exists():
        prepare_flashscore.build()
    else:
        log.info("flashscore_stats.csv present - skipping")


if __name__ == "__main__":
    build_all(force=True)
