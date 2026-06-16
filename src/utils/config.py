"""
Central configuration: paths, teams, ratings, competition mappings, feature sets.

Everything that the rest of the pipeline needs to agree on lives here, so the
data-collection, preprocessing, feature, model and prediction layers all share a
single source of truth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths (all relative to the project root, resolved from this file's location)
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ELO_CACHE_DIR = RAW_DIR / "eloratings"
INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"

# Canonical input files (spec sections 2A-2D), produced by src/data_collection.
MATCHES_CSV = INPUT_DIR / "matches.csv"                 # 2A match results
FLASHSCORE_CSV = INPUT_DIR / "flashscore_stats.csv"     # 2B advanced stats
FIFA_CSV = INPUT_DIR / "fifa_rankings.csv"              # 2C FIFA rankings
ELO_CSV = INPUT_DIR / "elo_ratings.csv"                 # 2D Elo ratings

# Raw collected Flashscore extraction (one row per team per match).
FLASHSCORE_RAW_CSV = RAW_DIR / "flashscore_team_stats_raw.csv"

# Pipeline artefacts.
TEAM_PERSPECTIVE_CSV = PROCESSED_DIR / "team_perspective.csv"
MODEL_DATASET_CSV = PROCESSED_DIR / "model_dataset.csv"
PREDICTIONS_CSV = OUTPUTS_DIR / "norway_world_cup_predictions.csv"

for _d in (RAW_DIR, ELO_CACHE_DIR, INPUT_DIR, PROCESSED_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Time windows
# --------------------------------------------------------------------------- #

# Collect matches from this date on (user decision: "since Dec 2020").
HISTORY_START = pd.Timestamp(2020, 12, 1)

# Time-based train/test boundary for evaluation (spec section 9).
TRAIN_TEST_SPLIT_DATE = pd.Timestamp(2025, 1, 1)

# Read "latest available" features as of this date when predicting fixtures.
PREDICTION_DATE = pd.Timestamp(2026, 6, 14)

# Rolling-form windows (spec section 6).
ROLLING_WINDOWS = (5, 10)
ROLLING_MIN_PERIODS = 3   # need at least this many prior matches for a window value

# Time-decay sample weighting during model training (tidsavskrivning).
# Each training row is weighted 0.5 ** (age_in_days / half_life), so a match this
# many days old influences the fit half as much as the most recent one. Older
# matches (2021) count less than recent ones (2025/2026). Set to None to disable.
# Weights are normalized to mean 1 so the data-loss vs L2 (C) balance is unchanged.
TIME_DECAY_HALF_LIFE_DAYS = 1095   # ~3 years

# --------------------------------------------------------------------------- #
# Teams
# --------------------------------------------------------------------------- #

# Teams we build a labelled dataset / predictions for. Written generally so more
# teams can be added: add the eloratings 2-letter code below and (optionally) a
# FIFA history, and the rest of the pipeline picks it up.
FOCUS_TEAMS = {"Norway": "NO", "Iraq": "IQ", "Senegal": "SN", "France": "FR"}

# eloratings 2-letter code -> file/display name (spaces -> underscores in URL).
# Covers every focus-team opponent since 2020-12 plus the three WC opponents.
CODE2NAME = {
    "NO": "Norway", "FR": "France", "SN": "Senegal", "IQ": "Iraq",
    "AM": "Armenia", "AT": "Austria", "CH": "Switzerland", "CY": "Cyprus",
    "CZ": "Czechia", "DK": "Denmark", "EE": "Estonia", "ES": "Spain",
    "FI": "Finland", "FO": "Faroe_Islands", "GE": "Georgia", "GI": "Gibraltar",
    "GR": "Greece", "IE": "Ireland", "IL": "Israel", "IT": "Italy",
    "JO": "Jordan", "KO": "Kosovo", "KZ": "Kazakhstan", "LU": "Luxembourg",
    "LV": "Latvia", "MA": "Morocco", "MD": "Moldova", "ME": "Montenegro",
    "NL": "Netherlands", "NZ": "New_Zealand", "RS": "Serbia", "SE": "Sweden",
    "SI": "Slovenia", "SK": "Slovakia", "SQ": "Scotland", "TR": "Turkey",
    "AE": "United_Arab_Emirates", "BA": "Bosnia", "BR": "Brazil", "CL": "Chile",
    "CO": "Colombia", "CR": "Costa_Rica", "EG": "Egypt", "GH": "Ghana",
    "GN": "Guinea", "GW": "Guinea-Bissau", "HR": "Croatia", "KW": "Kuwait",
    "MR": "Mauritania", "MW": "Malawi", "MZ": "Mozambique", "NA": "Namibia",
    "NG": "Nigeria", "OM": "Oman", "PK": "Pakistan", "PY": "Paraguay",
    "QA": "Qatar", "SA": "Saudi_Arabia", "SD": "Sudan", "TG": "Togo",
    "TN": "Tunisia", "ZA": "South_Africa", "BJ": "Benin", "BF": "Burkina_Faso",
    "CD": "Congo_DR", "CG": "Congo", "CI": "Cote_d_Ivoire", "CM": "Cameroon",
    "GA": "Gabon", "GM": "Gambia", "RW": "Rwanda", "SS": "South_Sudan",
    "TZ": "Tanzania", "UG": "Uganda", "ZM": "Zambia", "BO": "Bolivia",
    "BI": "Burundi", "DZ": "Algeria", "AO": "Angola", "CV": "Cape_Verde",
}

NAME2CODE = {name: code for code, name in CODE2NAME.items()}

# --------------------------------------------------------------------------- #
# FIFA rankings (official year-end ranks, supplied for the four focus teams)
# --------------------------------------------------------------------------- #

# {team: {year: rank}}. FIFA points are not supplied; left blank (loader fills
# NaN). Used descriptively + as a sparse feature; Elo is the dense cross-team
# strength signal because opponents lack FIFA history here.
FIFA_RANK = {
    "France":  {2020: 2,  2021: 3,  2022: 3,  2023: 2,  2024: 2,  2025: 3,  2026: 3},
    "Senegal": {2020: 20, 2021: 20, 2022: 19, 2023: 20, 2024: 17, 2025: 19, 2026: 15},
    "Norway":  {2020: 44, 2021: 41, 2022: 43, 2023: 44, 2024: 43, 2025: 29, 2026: 31},
    "Iraq":    {2020: 70, 2021: 75, 2022: 68, 2023: 63, 2024: 56, 2025: 58, 2026: 57},
}

# --------------------------------------------------------------------------- #
# Competition codes (eloratings) -> (readable name, type bucket)
# --------------------------------------------------------------------------- #

COMP_INFO = {
    "F":   ("Friendly", "friendly"),
    "FT":  ("Friendly tournament", "friendly"),
    "WC":  ("FIFA World Cup", "world_cup"),
    "WQ":  ("World Cup qualification", "qualifier"),
    "WQS": ("World Cup qualification play-off", "qualifier"),
    "EC":  ("UEFA Euro", "continental_cup"),
    "EQ":  ("UEFA Euro qualification", "qualifier"),
    "ENA": ("UEFA Nations League", "nations_league"),
    "ENB": ("UEFA Nations League", "nations_league"),
    "ENC": ("UEFA Nations League", "nations_league"),
    "END": ("UEFA Nations League", "nations_league"),
    "ENL": ("UEFA Nations League finals", "nations_league"),
    "AC":  ("AFC Asian Cup", "continental_cup"),
    "AQ":  ("AFC Asian Cup qualification", "qualifier"),
    "AR":  ("Africa Cup of Nations", "continental_cup"),
    "FQ":  ("Africa Cup of Nations qualification", "qualifier"),
    "ARC": ("FIFA Arab Cup", "regional_cup"),
    "GLF": ("Arabian Gulf Cup", "regional_cup"),
    "KNG": ("King's Cup", "regional_cup"),
    "CSF": ("COSAFA Cup", "regional_cup"),
}

# Stable integer encoding for the competition_type categorical (used by models).
COMP_TYPE_CODES = {
    "friendly": 0, "qualifier": 1, "nations_league": 2, "continental_cup": 3,
    "world_cup": 4, "regional_cup": 5, "other": 6,
}

# Weights for the weighted rolling-form windows.
# Friendlies = 0 (not a tournament). Higher-stakes matches count proportionally more.
# Weight-0 rows are excluded from the rolling mean entirely; if all rows in a window
# are friendlies the feature is NaN (SimpleImputer fills with median in the pipeline).
COMP_WEIGHTS = {
    "friendly":     1.0,  # international friendly + friendly tournament
    "regional_cup":   1,  # King's Cup, COSAFA Cup, Arab Cup, Gulf Cup
    "qualifier":      2,  # WCQ, Euro qual, AFCON qual, AFC qual
    "nations_league": 2,  # UEFA Nations League (all divisions + finals)
    "continental_cup": 3, # UEFA Euro, AFCON, AFC Asian Cup
    "world_cup":      4,  # FIFA World Cup
    "other":          1,  # unknown / catch-all, conservative
}


def competition_info(code: str) -> tuple[str, str]:
    """eloratings competition code -> (readable name, type bucket)."""
    return COMP_INFO.get(code, (code, "other"))


# --------------------------------------------------------------------------- #
# eloratings.net source
# --------------------------------------------------------------------------- #

ELO_BASE = "https://www.eloratings.net"
USER_AGENT = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
REQUEST_TIMEOUT = 25
REQUEST_PAUSE = 0.3  # polite delay between downloads

# --------------------------------------------------------------------------- #
# Fixtures to predict (spec section 10)
# --------------------------------------------------------------------------- #

FIXTURES = [
    {"date": "2026-06-16", "team": "Norway", "opponent": "Iraq",    "is_home": 0, "is_neutral": 1},
    {"date": "2026-06-22", "team": "Norway", "opponent": "Senegal", "is_home": 1, "is_neutral": 1},
    {"date": "2026-06-26", "team": "Norway", "opponent": "France",  "is_home": 1, "is_neutral": 1},
]
# All three are World Cup matches at a neutral venue.
FIXTURE_COMPETITION_TYPE = "world_cup"

# --------------------------------------------------------------------------- #
# Feature sets used by the trained models
# --------------------------------------------------------------------------- #

# Rolling-form base metrics computed per team-perspective row (before any diff).
FORM_METRICS = [
    "win_rate", "draw_rate", "loss_rate", "points_per_game",
    "avg_goals_for", "avg_goals_against", "goal_diff_avg",
    "clean_sheet_rate", "failed_to_score_rate",
]

# Advanced (Flashscore) rolling metrics — optional; handled gracefully if absent.
ADVANCED_METRICS = [
    "xg_for", "xg_against", "xgot_for", "xgot_against",
    "shots_for", "shots_against", "shots_on_target_for", "shots_on_target_against",
    "big_chances_for", "big_chances_against", "possession_avg",
    "pass_accuracy_avg", "corners_for", "corners_against", "goals_prevented",
]

# Logistic model: cross-team-comparable signals only — Elo (dense for every
# team) + rolling-form *_diff + context. FIFA is sparse for opponents, so it
# is carried descriptively but NOT trained on.
MODEL_FEATURES = [
    "elo_diff",
    "win_rate_last_5_diff", "points_per_game_last_5_diff",
    "goal_diff_avg_last_5_diff", "avg_goals_against_last_5_diff",
    "win_rate_last_10_diff", "points_per_game_last_10_diff",
    "goal_diff_avg_last_10_diff",
    "rest_days_diff",
    "is_home", "is_neutral", "comp_type_code",
]

# 3-class label order (kept fixed so probability columns line up everywhere).
RESULT_CLASSES = ["win", "draw", "loss"]
