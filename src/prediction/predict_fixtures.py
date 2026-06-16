"""
Predict the three World Cup fixtures and write the output CSV + a readable
summary.

Each fixture is scored from both perspectives (so Iraq-Norway reports both
P(Iraq ...) and P(Norway ...)) by logistic regression, by Poisson, and by the
ensemble (mean of logistic + Poisson).
"""

from __future__ import annotations

import pandas as pd

from ..models import poisson_model as P
from ..models.logistic_model import proba_frame
from ..utils.config import (
    FIXTURES, PREDICTIONS_CSV, RESULT_CLASSES,
)
from ..utils.logging_utils import get_logger
from ..utils.team_names import standardize_team_name

log = get_logger(__name__)

ENSEMBLE_MEMBERS = ["logistic", "poisson"]


def _fixture_label(fx: dict) -> str:
    """Home team listed first: is_home=0 means the opponent hosts."""
    team, opp = standardize_team_name(fx["team"]), standardize_team_name(fx["opponent"])
    return f"{team} vs {opp}" if fx.get("is_home") else f"{opp} vs {team}"


def _classifier_probs(fitted: dict, row: pd.DataFrame) -> dict:
    """{model_name: {win,draw,loss}} for one perspective row."""
    out = {}
    for name, (model, feats) in fitted.items():
        p = proba_frame(model, row[feats]).iloc[0]
        out[name] = {c: float(p[c]) for c in RESULT_CLASSES}
    return out


def predict(fixture_dataset: pd.DataFrame, fitted: dict,
            strengths: pd.DataFrame, mu: float) -> pd.DataFrame:
    """Build the predictions table for every fixture and perspective."""
    records = []
    for fx in FIXTURES:
        team = standardize_team_name(fx["team"])
        opp = standardize_team_name(fx["opponent"])
        date = pd.Timestamp(fx["date"])
        mid = f"FIX_{team}_{opp}_{date.date()}"
        label = _fixture_label(fx)
        is_neutral = int(fx.get("is_neutral", 0))

        for persp_team, persp_opp in [(team, opp), (opp, team)]:
            row = fixture_dataset[(fixture_dataset["match_id"] == mid)
                                  & (fixture_dataset["team"] == persp_team)]
            if row.empty:
                log.warning("no fixture row for %s vs %s", persp_team, persp_opp)
                continue

            probs = _classifier_probs(fitted, row)
            # Poisson uses this perspective's home flag.
            is_home = int(persp_team == team and fx.get("is_home", 0))
            pois = P.predict(strengths, mu, persp_team, persp_opp,
                             is_home=is_home, is_neutral=is_neutral)
            probs["poisson"] = {c: float(pois[c]) for c in RESULT_CLASSES}

            # Ensemble = mean over available members.
            members = [probs[m] for m in ENSEMBLE_MEMBERS if m in probs]
            probs["ensemble"] = {c: sum(m[c] for m in members) / len(members)
                                 for c in RESULT_CLASSES}

            for name, wdl in probs.items():
                pred_result = max(wdl, key=wdl.get)
                rec = {
                    "fixture": label, "date": date.date(),
                    "team": persp_team, "opponent": persp_opp,
                    "predicted_win_probability": round(wdl["win"], 4),
                    "predicted_draw_probability": round(wdl["draw"], 4),
                    "predicted_loss_probability": round(wdl["loss"], 4),
                    "predicted_result": pred_result, "model_name": name,
                }
                if name == "poisson":
                    rec["likely_scoreline"] = f"{pois['score'][0]}-{pois['score'][1]}"
                records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(PREDICTIONS_CSV, index=False)
    log.info("wrote %s (%d rows)", PREDICTIONS_CSV, len(df))
    return df


# --------------------------------------------------------------------------- #
# Readable summary (spec section 11)
# --------------------------------------------------------------------------- #

def print_summary(pred: pd.DataFrame) -> None:
    """Norway-centric readable report for each fixture."""
    print("\n" + "=" * 70)
    print("WORLD CUP FIXTURE PREDICTIONS  (neutral venue)")
    print("=" * 70)

    for fx in FIXTURES:
        team = standardize_team_name(fx["team"])      # always Norway here
        opp = standardize_team_name(fx["opponent"])
        label = _fixture_label(fx)
        block = pred[(pred["fixture"] == label) & (pred["team"] == team)]
        if block.empty:
            continue

        print(f"\n{label}   ({fx['date']})")
        for _, r in block.iterrows():
            tag = r["model_name"]
            extra = f"   likely {r['likely_scoreline']}" if pd.notna(
                r.get("likely_scoreline")) else ""
            print(f"  {tag:<9} {team} win {r['predicted_win_probability']*100:4.0f}%"
                  f"   draw {r['predicted_draw_probability']*100:4.0f}%"
                  f"   {opp} win {r['predicted_loss_probability']*100:4.0f}%{extra}")

        ens = block[block["model_name"] == "ensemble"].iloc[0]
        verdict = {"win": team, "draw": "draw", "loss": opp}[ens["predicted_result"]]
        print(f"  -> consensus: {verdict} "
              f"({team} win {ens['predicted_win_probability']*100:.0f}% / "
              f"draw {ens['predicted_draw_probability']*100:.0f}% / "
              f"{opp} win {ens['predicted_loss_probability']*100:.0f}%)")

    print("\n(Full both-sides table in "
          f"{PREDICTIONS_CSV.relative_to(PREDICTIONS_CSV.parents[2])})")
