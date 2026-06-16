"""
Build the FIFA rankings input file (spec 2C) from the supplied year-end ranks.

FIFA does not expose a clean free bulk endpoint, so the user supplies official
year-end ranks for the focus teams (see ``config.FIFA_RANK``). We materialise
them as ``data/input/fifa_rankings.csv`` with one row per team per year, dated
31 December so the feature layer can pick "the most recent rank before a match".

Columns: ranking_date, team, team_code, fifa_rank, fifa_points. ``fifa_points``
is left blank (not supplied); downstream code treats it as optional/NaN.
"""

from __future__ import annotations

import pandas as pd

from ..utils.config import FIFA_CSV, FIFA_RANK, NAME2CODE
from ..utils.logging_utils import get_logger
from ..utils.team_names import standardize_team_name

log = get_logger(__name__)


def build() -> pd.DataFrame:
    rows = []
    for team, year_rank in FIFA_RANK.items():
        canon = standardize_team_name(team)
        code = NAME2CODE.get(team, "")
        for year, rank in sorted(year_rank.items()):
            rows.append({
                "ranking_date": pd.Timestamp(year, 12, 31).date(),
                "team": canon,
                "team_code": code,
                "fifa_rank": int(rank),
                "fifa_points": pd.NA,  # not supplied
            })
    df = pd.DataFrame(rows).sort_values(["team", "ranking_date"]).reset_index(drop=True)
    df.to_csv(FIFA_CSV, index=False)
    log.info("wrote %s (%d rows, %d teams)",
             FIFA_CSV.name, len(df), df["team"].nunique())
    return df


if __name__ == "__main__":
    build()
