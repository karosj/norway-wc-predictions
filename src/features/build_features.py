"""
Feature engineering: leakage-safe rolling pre-match features + matchup dataset.

THE GOLDEN RULE (spec section 4): every feature for a match is computed from
matches *strictly before* it. We enforce this with ``groupby('team').shift(1)``
*before* every ``rolling`` window, so the current match never contributes to its
own features.

Pipeline:
    team-perspective rows  --add_rolling_features-->  per-team pre-match features
    per-team features       --build_matchup-->        one row per team-vs-opponent
                                                      with team_/opponent_/_diff

The same machinery produces prediction rows for future fixtures: we append a
placeholder perspective row per fixture, run the *identical* rolling code, and
read the placeholder back — so fixtures and training data are built the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..preprocessing import load_data as L
from ..utils.config import (
    COMP_TYPE_CODES, COMP_WEIGHTS, FOCUS_TEAMS, RESULT_CLASSES,
    ROLLING_MIN_PERIODS, ROLLING_WINDOWS,
)
from ..utils.logging_utils import get_logger

log = get_logger(__name__)

# (per-match source column, output base). Rolling mean over the window => the
# feature named ``{base}_last_{w}``. Form metrics exist for every match.
FORM_SPEC = [
    ("is_win", "win_rate"), ("is_draw", "draw_rate"), ("is_loss", "loss_rate"),
    ("points", "points_per_game"),
    ("goals_for", "avg_goals_for"), ("goals_against", "avg_goals_against"),
    ("goal_diff", "goal_diff_avg"),
    ("clean_sheet", "clean_sheet_rate"), ("failed_to_score", "failed_to_score_rate"),
]
# Advanced (Flashscore) metrics — sparse; min_periods=1 so any history counts.
ADV_SPEC = [
    ("xg_for", "xg_for"), ("xg_against", "xg_against"),
    ("xgot_for", "xgot_for"), ("xgot_against", "xgot_against"),
    ("shots_for", "shots_for"), ("shots_against", "shots_against"),
    ("shots_on_target_for", "shots_on_target_for"),
    ("shots_on_target_against", "shots_on_target_against"),
    ("big_chances_for", "big_chances_for"), ("big_chances_against", "big_chances_against"),
    ("corners_for", "corners_for"), ("corners_against", "corners_against"),
    ("possession_for", "possession_avg"), ("pass_accuracy_for", "pass_accuracy_avg"),
    ("goals_prevented_for", "goals_prevented"),
]

# Difference-feature name overrides (everything else => "{base}_diff").
_DIFF_NAME = {
    "elo_before": "elo_diff",
    "fifa_rank_before": "fifa_rank_diff",
    "fifa_points_before": "fifa_points_diff",
    "days_since_last_match": "rest_days_diff",
}


# --------------------------------------------------------------------------- #
# Rolling features
# --------------------------------------------------------------------------- #

def _weighted_rolling_mean(df: pd.DataFrame, src: str, window: int,
                           min_periods: int) -> pd.Series:
    """Competition-weighted shift(1)+rolling mean per team.

    Each past match contributes `value × comp_weight` to the window; friendlies
    (weight=0) are excluded entirely. Returns NaN when the weight sum is zero
    (all-friendly stretch) — SimpleImputer handles this downstream.
    """
    results = []
    for _, grp in df.groupby("team", sort=False):
        v = grp[src].shift(1)
        w = grp["comp_weight"].shift(1)
        wv = (v * w).rolling(window, min_periods=min_periods).sum()
        ws = w.rolling(window, min_periods=min_periods).sum()
        results.append(wv / ws.replace(0, np.nan))
    return pd.concat(results).sort_index()


def add_rolling_features(persp: pd.DataFrame) -> pd.DataFrame:
    """Add all rolling-form, advanced and context pre-match features."""
    df = persp.sort_values(["team", "date"]).reset_index(drop=True)

    # days since the team's previous match (context feature, unweighted).
    df["days_since_last_match"] = df.groupby("team")["date"].diff().dt.days

    # competition_type -> stable integer code (model categorical feature).
    df["comp_type_code"] = (df["competition_type"].map(COMP_TYPE_CODES)
                            .fillna(COMP_TYPE_CODES["other"]).astype(int))

    # competition_type -> rolling weight (0 = friendly, 4 = World Cup).
    df["comp_weight"] = (df["competition_type"].map(COMP_WEIGHTS)
                         .fillna(COMP_WEIGHTS["other"]).astype(float))

    for src, base in FORM_SPEC:
        for w in ROLLING_WINDOWS:
            df[f"{base}_last_{w}"] = _weighted_rolling_mean(df, src, w, ROLLING_MIN_PERIODS)

    for src, base in ADV_SPEC:
        if src in df.columns:
            for w in ROLLING_WINDOWS:
                df[f"{base}_last_{w}"] = _weighted_rolling_mean(df, src, w, 1)

    # within-team net xG (for - against), per spec's xg_diff_last_*.
    for w in ROLLING_WINDOWS:
        if f"xg_for_last_{w}" in df.columns:
            df[f"xg_diff_last_{w}"] = df[f"xg_for_last_{w}"] - df[f"xg_against_last_{w}"]

    return df


def rolling_feature_columns(df: pd.DataFrame) -> list[str]:
    """All per-team feature columns that get the team_/opponent_/diff treatment."""
    rolling = [c for c in df.columns
               if c.endswith("_last_5") or c.endswith("_last_10")]
    base = ["elo_before", "fifa_rank_before", "fifa_points_before",
            "days_since_last_match"]
    return [c for c in base if c in df.columns] + rolling


# --------------------------------------------------------------------------- #
# Matchup assembly (team vs opponent, with difference features)
# --------------------------------------------------------------------------- #

def build_matchup(persp_feat: pd.DataFrame, focus_only: bool = True) -> pd.DataFrame:
    """Pair each team-perspective row with its opponent's row (same match).

    Produces ``team_<f>`` / ``opponent_<f>`` / ``<f>_diff`` for every feature,
    plus context and the 3-class label. ``focus_only`` keeps only rows whose
    *team* side is a focus team (the rows we train/predict on).
    """
    feats = rolling_feature_columns(persp_feat)
    keep = (["match_id", "date", "team", "opponent", "is_home", "is_neutral",
             "competition_type", "comp_type_code", "goals_for", "goals_against",
             "result"] + feats)
    base = persp_feat[keep].copy()

    merged = base.merge(base, on="match_id", suffixes=("", "_OPP"))
    merged = merged[merged["team_OPP"] == merged["opponent"]].copy()
    if focus_only:
        merged = merged[merged["team"].isin(FOCUS_TEAMS)].copy()

    cols: dict[str, pd.Series] = {
        "match_id": merged["match_id"], "date": merged["date"],
        "team": merged["team"], "opponent": merged["opponent"],
        "competition_type": merged["competition_type"],
        "is_home": merged["is_home"].astype(int),
        "is_neutral": merged["is_neutral"].astype(int),
        "comp_type_code": merged["comp_type_code"].astype(int),
        "goals_for": merged["goals_for"], "goals_against": merged["goals_against"],
    }
    for f in feats:
        team_col, opp_col = merged[f], merged[f"{f}_OPP"]
        cols[f"team_{f}"] = team_col
        cols[f"opponent_{f}"] = opp_col
        cols[_DIFF_NAME.get(f, f"{f}_diff")] = team_col - opp_col

    # Labels (NaN for fixtures, which have no result yet).
    cols["target_result"] = merged["result"]
    cols["target_win"] = pd.Series(
        np.where(merged["result"].isna(), np.nan,
                 (merged["result"] == "win").astype(float)),
        index=merged.index)

    # Build all columns at once to avoid DataFrame fragmentation.
    out = pd.concat(cols, axis=1)
    return out.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Fixture placeholder rows
# --------------------------------------------------------------------------- #

def _fixture_perspective_rows(fixtures: list[dict]) -> pd.DataFrame:
    """Two placeholder perspective rows per fixture (team view + opponent view)."""
    from ..utils.team_names import standardize_team_name
    from ..utils.config import FIXTURE_COMPETITION_TYPE

    rows = []
    for fx in fixtures:
        team = standardize_team_name(fx["team"])
        opp = standardize_team_name(fx["opponent"])
        date = pd.Timestamp(fx["date"])
        mid = f"FIX_{team}_{opp}_{date.date()}"
        is_neutral = int(fx.get("is_neutral", 0))
        # Home advantage only applies when the venue is not neutral.
        team_home = int(fx.get("is_home", 0)) and not is_neutral
        common = {"match_id": mid, "date": date, "is_neutral": is_neutral,
                  "competition_type": FIXTURE_COMPETITION_TYPE,
                  "goals_for": np.nan, "goals_against": np.nan, "goal_diff": np.nan,
                  "result": np.nan, "points": np.nan,
                  "is_win": np.nan, "is_draw": np.nan, "is_loss": np.nan,
                  "clean_sheet": np.nan, "failed_to_score": np.nan}
        rows.append({**common, "team": team, "opponent": opp, "is_home": int(team_home)})
        rows.append({**common, "team": opp, "opponent": team, "is_home": 0})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #

def engineer(matches: pd.DataFrame, elo: pd.DataFrame, fifa: pd.DataFrame,
             flashscore: pd.DataFrame, fixtures: list[dict] | None = None
             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build (team_perspective_features, model_dataset, fixture_dataset).

    Historical perspective rows and fixture placeholders are concatenated and run
    through the *same* rolling code, guaranteeing identical, leakage-safe feature
    construction for training and prediction.
    """
    persp = L.to_team_perspective(matches)
    persp = L.attach_ratings(persp, elo, fifa)
    persp = L.attach_flashscore(persp, flashscore)

    # Historical rolling features (no fixtures present -> nothing to contaminate).
    persp_feat = add_rolling_features(persp)
    model_dataset = build_matchup(persp_feat, focus_only=True)
    all_team_dataset = build_matchup(persp_feat, focus_only=False)

    if fixtures:
        # Compute each fixture in ISOLATION against history. A team (e.g. Norway)
        # plays in several fixtures; if all placeholder rows shared one frame, an
        # earlier fixture's placeholder (is_win=NaN but comp_weight>0) would land
        # in a later fixture's rolling lookback and deflate that team's weighted
        # form. Isolating each fixture guarantees a clean, placeholder-free window.
        fx_rows = []
        for fx in fixtures:
            ph = L.attach_ratings(_fixture_perspective_rows([fx]), elo, fifa)
            combined = add_rolling_features(pd.concat([persp, ph], ignore_index=True))
            mask = combined["match_id"].astype(str).str.startswith("FIX_")
            fx_rows.append(combined[mask])
        fixture_dataset = build_matchup(pd.concat(fx_rows, ignore_index=True),
                                        focus_only=True)
    else:
        fixture_dataset = pd.DataFrame()

    log.info("model dataset: %d focus rows | %d all-team rows | %d fixture rows",
             len(model_dataset), len(all_team_dataset), len(fixture_dataset))
    return persp_feat, model_dataset, all_team_dataset, fixture_dataset
