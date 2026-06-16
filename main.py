"""
Norway 2026 World Cup prediction pipeline — end-to-end entry point.

    py main.py                  # full run (collect if needed, train, predict)
    py main.py --force-collect  # re-download eloratings + rebuild all inputs
    py main.py --skip-eval      # skip the time-split evaluation section

Stages: collect inputs -> load + repair -> engineer features -> evaluate
(time-based split) -> fit models -> predict the three fixtures.
"""

from __future__ import annotations

import argparse

from src.data_collection import build_inputs
from src.features import build_features as F
from src.models import logistic_model as LM
from src.models import poisson_model as P
from src.prediction import predict_fixtures as PF
from src.preprocessing import load_data as L
from src.utils.config import (
    FIXTURES, MODEL_DATASET_CSV, PROCESSED_DIR, TEAM_PERSPECTIVE_CSV,
)
from src.utils.logging_utils import get_logger

log = get_logger("main")


def run(force_collect: bool = False, skip_eval: bool = False) -> None:
    # 1) Inputs --------------------------------------------------------------- #
    log.info("STAGE 1/5  collecting / verifying input data")
    build_inputs.build_all(force=force_collect)

    # 2) Load + reshape + repair ---------------------------------------------- #
    log.info("STAGE 2/5  loading and preprocessing")
    matches = L.load_matches()
    flashscore = L.repair_flashscore(L.load_flashscore(), matches)
    fifa = L.load_fifa()
    elo = L.load_elo()

    # 3) Feature engineering -------------------------------------------------- #
    log.info("STAGE 3/5  engineering leakage-safe features")
    persp_feat, model_dataset, all_team_dataset, fixture_dataset = F.engineer(
        matches, elo, fifa, flashscore, FIXTURES)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    persp_feat.to_csv(TEAM_PERSPECTIVE_CSV, index=False)
    model_dataset.to_csv(MODEL_DATASET_CSV, index=False)
    log.info("saved %s and %s", TEAM_PERSPECTIVE_CSV.name, MODEL_DATASET_CSV.name)

    # 4) Evaluate + fit ------------------------------------------------------- #
    # Logistic trains on all-team rows (CV showed −0.038 log-loss improvement).
    if not skip_eval:
        log.info("STAGE 4/5  evaluating logistic (time-based split)")
        LM.evaluate_time_split(model_dataset, train_override=all_team_dataset)
    else:
        log.info("STAGE 4/5  skipped evaluation")
    fitted = LM.fit_final(model_dataset, train_override=all_team_dataset)
    mu, strengths = P.compute_strengths(persp_feat)

    # 5) Predict fixtures ----------------------------------------------------- #
    log.info("STAGE 5/5  predicting fixtures")
    predictions = PF.predict(fixture_dataset, fitted, strengths, mu)
    PF.print_summary(predictions)


def main() -> None:
    ap = argparse.ArgumentParser(description="Norway WC 2026 prediction pipeline")
    ap.add_argument("--force-collect", action="store_true",
                    help="re-download eloratings data and rebuild all inputs")
    ap.add_argument("--skip-eval", action="store_true",
                    help="skip the time-based evaluation section")
    args = ap.parse_args()
    run(force_collect=args.force_collect, skip_eval=args.skip_eval)


if __name__ == "__main__":
    main()
