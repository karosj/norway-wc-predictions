"""
Poisson goals model (spec section 8D).

Classic independent-Poisson approach:

    attack_strength(team)  = team goals scored per game / league average
    defence_strength(team) = team goals conceded per game / league average
    expected goals (team)  = attack(team) * defence(opp) * league_avg * home_adv

The two expected-goal rates feed an independent-Poisson scoreline grid, which we
collapse to P(team win) / P(draw) / P(opponent win) and the single most likely
scoreline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..utils.config import HISTORY_START, RESULT_CLASSES
from ..utils.logging_utils import get_logger

log = get_logger(__name__)

MAX_GOALS = 10          # truncation for the scoreline grid
HOME_ADV = 1.10         # expected-goals multiplier for a true (non-neutral) home side


def compute_strengths(persp: pd.DataFrame, since: pd.Timestamp = HISTORY_START) -> tuple:
    """League-average goals + per-team attack/defence strengths since ``since``."""
    d = persp[persp["date"] >= since]
    mu = float(d["goals_for"].mean())
    rows = []
    for team, g in d.groupby("team"):
        gf, ga = g["goals_for"].mean(), g["goals_against"].mean()
        rows.append({"team": team, "games": len(g), "gf_per_game": gf,
                     "ga_per_game": ga, "attack": gf / mu, "defence": ga / mu})
    strengths = pd.DataFrame(rows).set_index("team")
    log.info("Poisson strengths: %d teams, league avg = %.2f goals/team/match",
             len(strengths), mu)
    return mu, strengths


def poisson_match(lam_team: float, lam_opp: float, max_goals: int = MAX_GOALS) -> dict:
    """Independent-Poisson scoreline grid -> W/D/L probs + most likely score."""
    goals = np.arange(max_goals + 1)
    a = poisson.pmf(goals, lam_team)
    b = poisson.pmf(goals, lam_opp)
    grid = np.outer(a, b)
    grid /= grid.sum()  # renormalise the truncated tail
    p_win = float(np.tril(grid, -1).sum())   # team scores more (row > col)
    p_draw = float(np.trace(grid))
    p_loss = float(np.triu(grid, 1).sum())   # opponent scores more
    i, j = np.unravel_index(grid.argmax(), grid.shape)
    return {"win": p_win, "draw": p_draw, "loss": p_loss,
            "lam_team": lam_team, "lam_opp": lam_opp, "score": (int(i), int(j))}


def predict(strengths: pd.DataFrame, mu: float, team: str, opponent: str,
            is_home: int = 0, is_neutral: int = 1) -> dict:
    """Expected goals for both sides, then the Poisson outcome grid.

    Falls back to league-average strength (1.0) for an unknown team.
    """
    def att(t): return strengths.at[t, "attack"] if t in strengths.index else 1.0
    def dfn(t): return strengths.at[t, "defence"] if t in strengths.index else 1.0

    home_mult = HOME_ADV if (is_home and not is_neutral) else 1.0
    lam_team = att(team) * dfn(opponent) * mu * home_mult
    lam_opp = att(opponent) * dfn(team) * mu
    return poisson_match(lam_team, lam_opp)


def proba_vector(result: dict) -> list[float]:
    """W/D/L dict -> probability list in fixed RESULT_CLASSES order."""
    return [result[c] for c in RESULT_CLASSES]
