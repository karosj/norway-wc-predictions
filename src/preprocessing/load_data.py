"""
Preprocessing: load the four input files and reshape match results into
team-perspective rows, attaching leakage-safe pre-match Elo / FIFA and the
same-match Flashscore stats (the latter are only ever consumed via shift(1)
rolling in the feature layer).
"""

from __future__ import annotations

import pandas as pd

from ..utils.config import (
    ELO_CSV, FIFA_CSV, FLASHSCORE_CSV, MATCHES_CSV,
)
from ..utils.logging_utils import get_logger
from ..utils.team_names import standardize_team_name

log = get_logger(__name__)

# Flashscore "for" stats kept per match (team's own performance).
_FOR_RENAME = {
    "xg": "xg_for", "xgot": "xgot_for", "total_shots": "shots_for",
    "shots_on_target": "shots_on_target_for", "big_chances": "big_chances_for",
    "corners": "corners_for", "possession_pct": "possession_for",
    "pass_accuracy": "pass_accuracy_for", "goals_prevented": "goals_prevented_for",
}
# Flashscore "against" stats: the opponent's own numbers in the same match.
_AGAINST_RENAME = {
    "xg": "xg_against", "xgot": "xgot_against", "total_shots": "shots_against",
    "shots_on_target": "shots_on_target_against", "big_chances": "big_chances_against",
    "corners": "corners_against",
}


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_matches() -> pd.DataFrame:
    """Load match results (spec 2A); parse dates, standardize team names."""
    df = pd.read_csv(MATCHES_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["home_team"] = df["home_team"].map(standardize_team_name)
    df["away_team"] = df["away_team"].map(standardize_team_name)
    df["neutral"] = df["neutral"].astype(bool)
    log.info("loaded %d matches (%s -> %s)", len(df),
             df["date"].min().date(), df["date"].max().date())
    return df


def load_flashscore() -> pd.DataFrame:
    """Load Flashscore advanced stats (spec 2B). Empty frame if file absent."""
    if not FLASHSCORE_CSV.exists():
        log.warning("no Flashscore file - advanced features will be skipped")
        return pd.DataFrame(columns=["date", "team", "opponent"])
    df = pd.read_csv(FLASHSCORE_CSV)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df["team"] = df["team"].map(standardize_team_name)
    df["opponent"] = df["opponent"].map(standardize_team_name)
    log.info("loaded %d Flashscore team-match rows (%d teams)",
             len(df), df["team"].nunique())
    return df


def load_fifa() -> pd.DataFrame:
    """Load FIFA rankings (spec 2C). Empty frame if file absent."""
    if not FIFA_CSV.exists():
        log.warning("no FIFA file - ranking features will be NaN")
        return pd.DataFrame(columns=["ranking_date", "team", "fifa_rank", "fifa_points"])
    df = pd.read_csv(FIFA_CSV)
    df["ranking_date"] = pd.to_datetime(df["ranking_date"])
    df["team"] = df["team"].map(standardize_team_name)
    df["fifa_points"] = pd.to_numeric(df.get("fifa_points"), errors="coerce")
    return df.sort_values(["team", "ranking_date"]).reset_index(drop=True)


def load_elo() -> pd.DataFrame:
    """Load Elo ratings (spec 2D). Empty frame if file absent (optional source)."""
    if not ELO_CSV.exists():
        log.warning("no Elo file - Elo features will be NaN")
        return pd.DataFrame(columns=["date", "team", "elo"])
    df = pd.read_csv(ELO_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["team"] = df["team"].map(standardize_team_name)
    df["elo"] = pd.to_numeric(df["elo"], errors="coerce")
    return df.sort_values(["team", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Team-perspective reshape (spec section 5)
# --------------------------------------------------------------------------- #

def to_team_perspective(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode each match into two team-perspective rows.

    ``Norway 2-1 Senegal`` becomes Norway(gf=2,ga=1,win) and Senegal(gf=1,ga=2,loss).
    A neutral-venue match gives is_home=0 to both sides.
    """
    home = pd.DataFrame({
        "match_id": matches["match_id"], "date": matches["date"],
        "team": matches["home_team"], "opponent": matches["away_team"],
        "goals_for": matches["home_score"], "goals_against": matches["away_score"],
        "is_home": (~matches["neutral"]).astype(int),
        "is_neutral": matches["neutral"].astype(int),
        "competition": matches["competition"],
        "competition_type": matches["competition_type"],
    })
    away = pd.DataFrame({
        "match_id": matches["match_id"], "date": matches["date"],
        "team": matches["away_team"], "opponent": matches["home_team"],
        "goals_for": matches["away_score"], "goals_against": matches["home_score"],
        "is_home": 0,
        "is_neutral": matches["neutral"].astype(int),
        "competition": matches["competition"],
        "competition_type": matches["competition_type"],
    })
    persp = pd.concat([home, away], ignore_index=True)

    persp["goal_diff"] = persp["goals_for"] - persp["goals_against"]
    persp["result"] = persp["goal_diff"].apply(
        lambda d: "win" if d > 0 else ("draw" if d == 0 else "loss"))
    persp["points"] = persp["result"].map({"win": 3, "draw": 1, "loss": 0})
    persp["is_win"] = (persp["result"] == "win").astype(int)
    persp["is_draw"] = (persp["result"] == "draw").astype(int)
    persp["is_loss"] = (persp["result"] == "loss").astype(int)
    persp["clean_sheet"] = (persp["goals_against"] == 0).astype(int)
    persp["failed_to_score"] = (persp["goals_for"] == 0).astype(int)

    persp = persp.sort_values(["team", "date"]).reset_index(drop=True)
    log.info("team-perspective rows: %d (%d matches x 2 sides)",
             len(persp), len(matches))
    return persp


# --------------------------------------------------------------------------- #
# Attach leakage-safe pre-match Elo + FIFA
# --------------------------------------------------------------------------- #

def attach_ratings(persp: pd.DataFrame, elo: pd.DataFrame,
                   fifa: pd.DataFrame) -> pd.DataFrame:
    """Add ``elo_before`` and ``fifa_rank_before`` / ``fifa_points_before``.

    Both use ``merge_asof`` with ``allow_exact_matches=False`` so only ratings
    strictly *before* the match date are used (no leakage). Elo's timeline stores
    post-match values, so the most recent prior entry is the pre-match rating.
    """
    persp = persp.sort_values("date").reset_index(drop=True)

    if not elo.empty:
        elo_sorted = elo.sort_values("date")
        persp = pd.merge_asof(
            persp, elo_sorted.rename(columns={"elo": "elo_before"}),
            on="date", by="team", direction="backward", allow_exact_matches=False,
        )
    else:
        persp["elo_before"] = pd.NA

    if not fifa.empty:
        fifa_sorted = fifa.sort_values("ranking_date").rename(
            columns={"ranking_date": "date", "fifa_rank": "fifa_rank_before",
                     "fifa_points": "fifa_points_before"})
        persp = pd.merge_asof(
            persp.sort_values("date"),
            fifa_sorted[["date", "team", "fifa_rank_before", "fifa_points_before"]],
            on="date", by="team", direction="backward", allow_exact_matches=False,
        )
    else:
        persp["fifa_rank_before"] = pd.NA
        persp["fifa_points_before"] = pd.NA

    return persp.sort_values(["team", "date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Repair Flashscore team attribution using eloratings as ground truth
# --------------------------------------------------------------------------- #

def repair_flashscore(flashscore: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Fix mis-attributed Flashscore stat rows against true scorelines.

    The source extraction labels which side a stat row belongs to incorrectly in
    ~half of matches (verified: ``is_home`` agrees with eloratings only ~49% of
    the time, and many rows have goals > their own shots-on-target, which is
    impossible). Since a team cannot score more goals than it had shots on
    target, we use eloratings goals to detect and fix the swap:

    * both rows present  -> if swapping the two teams' labels removes more
      impossibilities than it creates, swap them;
    * single row present -> if it is impossible, null its stats so it cannot
      poison rolling averages.
    """
    if flashscore.empty or "shots_on_target" not in flashscore.columns:
        return flashscore

    goals: dict[tuple, int] = {}
    for _, r in matches.iterrows():
        goals[(r["home_team"], r["date"])] = r["home_score"]
        goals[(r["away_team"], r["date"])] = r["away_score"]

    fs = flashscore.copy()
    stat_cols = [c for c in fs.columns if c not in ("date", "team", "opponent", "is_home")]
    pair = [tuple(sorted((t, o))) for t, o in zip(fs["team"], fs["opponent"])]
    fs["_pair"] = pair

    swapped = nulled = checked = 0
    for (date, _), idx in fs.groupby(["date", "_pair"]).groups.items():
        idx = list(idx)
        if len(idx) == 2:
            i1, i2 = idx
            r1, r2 = fs.loc[i1], fs.loc[i2]
            g1, g2 = goals.get((r1["team"], date)), goals.get((r2["team"], date))
            s1, s2 = r1["shots_on_target"], r2["shots_on_target"]
            if g1 is None or g2 is None or pd.isna(s1) or pd.isna(s2):
                continue
            checked += 1
            current = int(g1 > s1) + int(g2 > s2)
            swap = int(g1 > s2) + int(g2 > s1)
            if swap < current:
                fs.loc[i1, ["team", "opponent"]] = [r2["team"], r1["team"]]
                fs.loc[i2, ["team", "opponent"]] = [r1["team"], r2["team"]]
                swapped += 1
        else:
            i1 = idx[0]
            r1 = fs.loc[i1]
            g1, s1 = goals.get((r1["team"], date)), r1["shots_on_target"]
            if g1 is not None and pd.notna(s1) and g1 > s1:
                fs.loc[i1, stat_cols] = pd.NA
                nulled += 1

    fs = fs.drop(columns="_pair")
    log.info("Flashscore repair: checked %d paired matches, swapped %d, "
             "nulled %d unfixable single rows", checked, swapped, nulled)
    return fs


# --------------------------------------------------------------------------- #
# Attach same-match Flashscore stats (for / against)
# --------------------------------------------------------------------------- #

def attach_flashscore(persp: pd.DataFrame, flashscore: pd.DataFrame) -> pd.DataFrame:
    """Left-join the team's and the opponent's same-match Flashscore numbers.

    These are *same-match* values and would leak if used directly; the feature
    layer only ever reads them through shift(1) rolling windows.
    """
    if flashscore.empty:
        return persp

    have = [c for c in _FOR_RENAME if c in flashscore.columns]
    for_df = (flashscore[["date", "team"] + have]
              .rename(columns={k: _FOR_RENAME[k] for k in have}))
    persp = persp.merge(for_df, on=["date", "team"], how="left")

    have_a = [c for c in _AGAINST_RENAME if c in flashscore.columns]
    against_df = (flashscore[["date", "team"] + have_a]
                  .rename(columns={"team": "opponent",
                                   **{k: _AGAINST_RENAME[k] for k in have_a}}))
    persp = persp.merge(against_df, on=["date", "opponent"], how="left")

    n_cov = persp["xg_for"].notna().sum() if "xg_for" in persp.columns else 0
    log.info("attached Flashscore stats (xg coverage: %d/%d perspective rows)",
             n_cov, len(persp))
    return persp.sort_values(["team", "date"]).reset_index(drop=True)
