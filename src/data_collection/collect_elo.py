"""
Collect match results + Elo ratings from eloratings.net (no API key).

eloratings publishes one TSV per national team with its *full* match history and
post-match Elo. We fetch the four focus teams and every opponent they have faced
since the history window, cache each file to ``data/raw/eloratings/``, and emit:

* ``data/input/matches.csv``      — spec 2A, one row per match (home/away view)
* ``data/input/elo_ratings.csv``  — spec 2D, one row per team per match (Elo timeline)

Re-running is cheap and offline-friendly: cached TSVs are reused, so the network
is only touched the first time (or for teams not yet cached).
"""

from __future__ import annotations

import csv
import io
import time

import numpy as np
import pandas as pd
import requests

from ..utils.config import (
    CODE2NAME, ELO_BASE, ELO_CACHE_DIR, ELO_CSV, FOCUS_TEAMS, HISTORY_START,
    MATCHES_CSV, REQUEST_PAUSE, REQUEST_TIMEOUT, USER_AGENT, competition_info,
)
from ..utils.logging_utils import get_logger
from ..utils.team_names import standardize_team_name

log = get_logger(__name__)

# Full-history match log per team, cached in-process so we parse each TSV once.
_TEAM_CACHE: dict[str, pd.DataFrame | None] = {}
# eloratings authoritative code -> raw country name (filled by load_master_names).
_MASTER_NAMES: dict[str, str] = {}


def load_master_names(session: requests.Session) -> dict[str, str]:
    """eloratings code -> country name from en.teams.tsv (331 codes, cached).

    This is the authoritative map used both to build the download filename and to
    resolve opponents-of-opponents that our hand-curated table misses.
    """
    if _MASTER_NAMES:
        return _MASTER_NAMES
    cache = ELO_CACHE_DIR / "en.teams.tsv"
    if cache.exists():
        text = cache.read_text(encoding="utf-8")
    else:
        try:
            r = session.get(f"{ELO_BASE}/en.teams.tsv", timeout=REQUEST_TIMEOUT)
            r.encoding = "utf-8"
            text = r.text
            cache.write_text(text, encoding="utf-8")
        except requests.RequestException as e:
            log.warning("could not load master team table: %s", e)
            return _MASTER_NAMES
    for row in csv.reader(io.StringIO(text), delimiter="\t"):
        if len(row) >= 2:
            _MASTER_NAMES[row[0]] = row[1]
    return _MASTER_NAMES


def resolve_name(code: str) -> str:
    """eloratings code -> canonical display name (master table -> curated -> code)."""
    raw = _MASTER_NAMES.get(code) or CODE2NAME.get(code, code)
    return standardize_team_name(raw)


def _to_int(s: str) -> float:
    s = (s or "").strip().replace("−", "-").replace("—", "-")
    if s in ("", "-"):
        return np.nan
    try:
        return int(s)
    except ValueError:
        return np.nan


def fetch_tsv(team_name: str, session: requests.Session) -> str | None:
    """Download (or read from cache) one team's eloratings TSV."""
    fname = team_name.replace(" ", "_")
    cache = ELO_CACHE_DIR / f"{fname}.tsv"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    try:
        r = session.get(f"{ELO_BASE}/{fname}.tsv", timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        log.warning("download failed for %s: %s", fname, e)
        return None
    if r.status_code != 200:
        log.warning("%s.tsv -> HTTP %s", fname, r.status_code)
        return None
    r.encoding = "utf-8"
    cache.write_text(r.text, encoding="utf-8")
    time.sleep(REQUEST_PAUSE)
    return r.text


def parse_team_tsv(text: str, self_code: str) -> pd.DataFrame:
    """One team's TSV -> tidy, date-sorted match log from that team's view.

    Columns: date, opp_code, listed_home, is_neutral, neutral_code, comp,
    goals_for, goals_against, result, points, elo_after. ``elo_after`` is the
    team's post-match Elo (the pre-match value for the next fixture).
    """
    rows = []
    for r in csv.reader(io.StringIO(text), delimiter="\t"):
        if len(r) < 16:
            continue
        try:
            date = pd.Timestamp(int(r[0]), int(r[1]), int(r[2]))
        except ValueError:
            continue
        home, away = r[3], r[4]
        hs, as_ = _to_int(r[5]), _to_int(r[6])
        if np.isnan(hs) or np.isnan(as_):
            continue
        comp, neutral = r[7], r[8].strip()
        home_elo_after, away_elo_after = _to_int(r[10]), _to_int(r[11])
        if self_code == home:
            gf, ga, opp, elo_after, listed_home = hs, as_, away, home_elo_after, True
        elif self_code == away:
            gf, ga, opp, elo_after, listed_home = as_, hs, home, away_elo_after, False
        else:
            continue
        result = "win" if gf > ga else ("draw" if gf == ga else "loss")
        rows.append({
            "date": date, "opp_code": opp, "listed_home": listed_home,
            "is_neutral": bool(neutral), "neutral_code": neutral, "comp": comp,
            "goals_for": int(gf), "goals_against": int(ga), "result": result,
            "points": 3 if result == "win" else (1 if result == "draw" else 0),
            "elo_after": elo_after,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def get_team_log(code: str, session: requests.Session) -> pd.DataFrame | None:
    """Full-history match log for an eloratings code (cached)."""
    if code in _TEAM_CACHE:
        return _TEAM_CACHE[code]
    name = _MASTER_NAMES.get(code) or CODE2NAME.get(code)
    if name is None:
        log.debug("no name mapping for code '%s' - skipping", code)
        _TEAM_CACHE[code] = None
        return None
    text = fetch_tsv(name, session)
    df = parse_team_tsv(text, code) if text else pd.DataFrame()
    result = df if not df.empty else None
    _TEAM_CACHE[code] = result
    return result


def collect(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch focus teams + their opponents and write matches.csv + elo_ratings.csv.

    Returns the two DataFrames. ``refresh`` is reserved for forcing re-download;
    cached TSVs are otherwise reused.
    """
    session = requests.Session()
    session.headers.update(USER_AGENT)
    load_master_names(session)

    # 1) Focus teams, then discover every opponent code they have faced.
    focus_logs: dict[str, pd.DataFrame] = {}
    opponent_codes: set[str] = set()
    for name, code in FOCUS_TEAMS.items():
        df = get_team_log(code, session)
        if df is None:
            log.warning("no data for focus team %s (%s)", name, code)
            continue
        focus_logs[code] = df
        recent = df[df["date"] >= HISTORY_START]
        opponent_codes.update(recent["opp_code"].unique())
    opponent_codes -= set(FOCUS_TEAMS.values())

    log.info("focus teams loaded: %d | distinct opponents to fetch: %d",
             len(focus_logs), len(opponent_codes))

    # 2) Fetch each opponent's history (needed for their pre-match form/Elo).
    all_logs: dict[str, pd.DataFrame] = dict(focus_logs)
    for code in sorted(opponent_codes):
        df = get_team_log(code, session)
        if df is not None:
            all_logs[code] = df

    # 3) Build the Elo timeline (spec 2D): one row per team per match.
    elo_rows = []
    for code, df in all_logs.items():
        team = resolve_name(code)
        for _, r in df.iterrows():
            if pd.notna(r["elo_after"]):
                elo_rows.append({"date": r["date"].date(), "team": team,
                                 "elo": float(r["elo_after"])})
    elo_df = (pd.DataFrame(elo_rows)
              .drop_duplicates(subset=["team", "date"])
              .sort_values(["team", "date"]).reset_index(drop=True))

    # 4) Build the match-results table (spec 2A): de-duplicated home/away rows.
    seen: set[tuple] = set()
    match_rows = []
    for code, df in all_logs.items():
        recent = df[df["date"] >= HISTORY_START]
        for _, r in recent.iterrows():
            if r["listed_home"]:
                home_code, away_code = code, r["opp_code"]
                home_score, away_score = r["goals_for"], r["goals_against"]
            else:
                home_code, away_code = r["opp_code"], code
                home_score, away_score = r["goals_against"], r["goals_for"]
            key = (r["date"].date(), home_code, away_code, home_score, away_score)
            if key in seen:
                continue
            seen.add(key)
            comp_name, comp_type = competition_info(r["comp"])
            match_rows.append({
                "match_id": f"{r['date'].date()}_{home_code}_{away_code}",
                "date": r["date"].date(),
                "home_team": resolve_name(home_code),
                "away_team": resolve_name(away_code),
                "home_score": int(home_score), "away_score": int(away_score),
                "competition": comp_name, "competition_type": comp_type,
                "neutral": bool(r["is_neutral"]), "source": "eloratings.net",
            })
    matches_df = (pd.DataFrame(match_rows)
                  .sort_values("date").reset_index(drop=True))

    matches_df.to_csv(MATCHES_CSV, index=False)
    elo_df.to_csv(ELO_CSV, index=False)
    log.info("wrote %s (%d matches) and %s (%d team-date Elo rows)",
             MATCHES_CSV.name, len(matches_df), ELO_CSV.name, len(elo_df))
    return matches_df, elo_df


if __name__ == "__main__":
    collect()
