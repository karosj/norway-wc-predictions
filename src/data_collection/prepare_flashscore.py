"""
Build the Flashscore advanced-stats input file (spec 2B).

The raw extraction (``data/raw/flashscore_team_stats_raw.csv``, one row per team
per match) carries ~100 columns with source-specific names. We project it onto
the tidy schema the pipeline expects, resolving two quirks:

* the raw ``passes`` column is actually the pass-accuracy %, while the real pass
  counts live in ``passes_completed`` / ``passes_attempted``;
* "count" metrics for passes/crosses/tackles are stored as ``*_attempted``.

Rows without a parseable date are dropped (they cannot be joined to a match).
Any missing advanced column is simply left absent — the feature layer treats all
advanced stats as optional.
"""

from __future__ import annotations

import pandas as pd

from ..utils.config import FLASHSCORE_CSV, FLASHSCORE_RAW_CSV
from ..utils.logging_utils import get_logger
from ..utils.team_names import standardize_team_name

log = get_logger(__name__)

# schema-B column -> ordered raw-column candidates (first present wins).
_DIRECT = {
    "xg": ["xg"],
    "xgot": ["xgot"],
    "xa": ["xa"],
    "possession_pct": ["possession_pct"],
    "total_shots": ["total_shots"],
    "shots_on_target": ["shots_on_target"],
    "shots_off_target": ["shots_off_target"],
    "blocked_shots": ["blocked_shots"],
    "shots_inside_box": ["shots_inside_box"],
    "shots_outside_box": ["shots_outside_box"],
    "big_chances": ["big_chances"],
    "corners": ["corners"],
    "passes": ["passes_attempted"],
    "long_passes": ["long_passes_attempted", "long_passes"],
    "final_third_passes": ["final_third_passes_attempted", "final_third_passes"],
    "crosses": ["crosses_attempted", "crosses"],
    "fouls": ["fouls"],
    "tackles": ["tackles_attempted", "tackles"],
    "duels_won": ["duels_won"],
    "clearances": ["clearances"],
    "interceptions": ["interceptions"],
    "goalkeeper_saves": ["goalkeeper_saves"],
    "xgot_faced": ["xgot_faced"],
    "goals_prevented": ["goals_prevented"],
}


def _pick(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="float64")


def build() -> pd.DataFrame:
    if not FLASHSCORE_RAW_CSV.exists():
        log.warning("no raw Flashscore file at %s; writing empty stats file",
                    FLASHSCORE_RAW_CSV)
        empty = pd.DataFrame(columns=["date", "team", "opponent", "is_home"]
                             + list(_DIRECT) + ["pass_accuracy"])
        empty.to_csv(FLASHSCORE_CSV, index=False)
        return empty

    raw = pd.read_csv(FLASHSCORE_RAW_CSV, low_memory=False)

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw.get("date"), errors="coerce")
    out["team"] = raw["team"].map(standardize_team_name)
    out["opponent"] = raw["opponent"].map(standardize_team_name)
    out["is_home"] = pd.to_numeric(raw.get("is_home"), errors="coerce").astype("Int64")

    for col, candidates in _DIRECT.items():
        out[col] = _pick(raw, candidates)

    # pass_accuracy: derive from completed/attempted (robust), else the raw % col.
    completed = _pick(raw, ["passes_completed"])
    attempted = _pick(raw, ["passes_attempted"])
    acc = (completed / attempted * 100).where(attempted > 0)
    out["pass_accuracy"] = acc.fillna(_pick(raw, ["pass_accuracy", "passes"]))

    before = len(out)
    out = out[out["date"].notna()].copy()
    out["date"] = out["date"].dt.date
    out = (out.drop_duplicates(subset=["team", "date"])
              .sort_values(["team", "date"]).reset_index(drop=True))

    out.to_csv(FLASHSCORE_CSV, index=False)
    log.info("wrote %s (%d rows; dropped %d undated; %d teams)",
             FLASHSCORE_CSV.name, len(out), before - len(out), out["team"].nunique())
    return out


if __name__ == "__main__":
    build()
