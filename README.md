# Norway 2026 World Cup — match prediction pipeline

A reproducible football ML project that builds its own dataset from real match data,
engineers leakage-safe pre-match features, and outputs win/draw/loss probabilities
and a most-likely scoreline for Norway's three group-stage fixtures.

| Fixture | Date |
|---|---|
| Iraq vs Norway | 2026-06-16 |
| Norway vs Senegal | 2026-06-22 |
| Norway vs France | 2026-06-26 |

All three fixtures are treated as **neutral-venue** World Cup matches.

---

## Model core

Two complementary models, combined into an ensemble:

| Model | What it does |
|---|---|
| **Logistic regression** | Multinomial W/D/L classifier (L2, `C=1.0`). Features: Elo gap, rolling goal-difference form (last 5 & 10), venue, rest days. Trained on the broad all-team dataset (~9 k rows) for better-calibrated probabilities. |
| **Poisson** | Attack/defence strengths → expected goals → independent-Poisson scoreline grid. Gives scoreline probabilities and the most likely score. |
| **Ensemble** | Simple mean of logistic + Poisson probabilities. The headline output. |

---

## Quick start

```bash
pip install -r requirements.txt
py main.py              # Windows — plain `python` is a Microsoft Store stub
python main.py          # macOS / Linux
```

That single command runs the entire pipeline:

1. **Collect** — downloads eloratings.net data once and caches it to `data/raw/`
2. **Load + repair** — reshapes to team-perspective rows, fixes Flashscore mis-attribution
3. **Engineer** — builds leakage-safe rolling features with `shift(1)` before every window
4. **Evaluate** — time-based train/test split (train < 2025, test 2025+); prints accuracy + log-loss
5. **Predict** — scores the three fixtures, writes `data/outputs/norway_world_cup_predictions.csv`

```bash
py main.py --force-collect   # re-download eloratings and rebuild every input
py main.py --skip-eval       # skip evaluation, jump straight to predictions
```

For an interactive walkthrough of **Iraq vs Norway** (probabilities + scoreline grid),
open [`notebooks/iraq_vs_norway.ipynb`](notebooks/iraq_vs_norway.ipynb).

---

## Data

### Sources

| File | Content | Source |
|---|---|---|
| `data/input/matches.csv` | One row per match — date, teams, score, competition type, neutral flag | eloratings.net TSV |
| `data/input/elo_ratings.csv` | Elo rating timeline, one row per team per match | eloratings.net TSV |
| `data/input/fifa_rankings.csv` | Year-end FIFA rank for the four focus teams | Hardcoded in `config.py` |
| `data/input/flashscore_stats.csv` | Advanced per-team match stats (xG, xGOT, shots, possession, …) | Flashscore extraction (see `scripts/`) |

The pipeline downloads `matches.csv` and `elo_ratings.csv` automatically on first run
(eloratings.net TSV endpoints, no API key). The Flashscore file is pre-extracted and
ships in `data/raw/flashscore_team_stats_raw.csv` — see [`scripts/README.md`](scripts/README.md)
for how it was produced and how to refresh it.

### Why so many teams?

We fetch eloratings for every opponent Norway, Iraq, Senegal, and France have faced,
not just the four focus teams. This gives all-team context (rolling form + Elo) on both
sides of every training match and powers the ~9 k-row all-team training set.

### Flashscore data repair

The raw Flashscore extraction attributes stats to the wrong side in ~half of matches
(its `is_home` flag agrees with eloratings only ~49% of the time; 43 focus rows had
*goals > own shots on target*, which is impossible). `preprocessing.repair_flashscore`
fixes this using eloratings scorelines as ground truth — 45 matches corrected. After
repair, goals ↔ own shots-on-target correlation rises from +0.28 to +0.58.

---

## Feature engineering

Matches are reshaped into **team-perspective rows**: `Norway 2–1 Senegal` becomes two rows —
`Norway (gf=2, ga=1, win)` and `Senegal (gf=1, ga=2, loss)`.

Rolling features are computed per team, sorted by date, with `shift(1)` before the window
so a match never contributes to its own features:

```python
df.groupby("team")[metric].shift(1).rolling(window).mean()
```

Pre-match Elo and FIFA rank are attached with `merge_asof(..., allow_exact_matches=False)`,
taking only values strictly before the match date. For each feature we also create a
**difference vs the opponent** (`elo_diff`, `win_rate_last_5_diff`, etc.) — the actual
model inputs are these cross-team-comparable diff columns.

Future fixtures run through the same code path as training rows, so training and
prediction features are constructed identically.

---

## Evaluation

Time-based split: train on matches before 2025-01-01, test on 2025+ (67 matches).
The logistic model trains on the all-team dataset; Elo-gap alone is the dominant predictor.

```
REFERENCE BASELINES
  always-win    accuracy 0.657
  class-prior   accuracy 0.657   log-loss 0.910
  uniform 1/3                    log-loss 1.099

MODEL        accuracy   log-loss
  logistic     0.672      0.833
```

Honest caveats: Elo dominates — rolling form adds little on a small sample. Draws are
the hardest class to call (always-win already scores 0.657 accuracy). Advanced Flashscore
stats are sparse and mostly imputed for the June 2026 fixtures. Treat probabilities as
directional, not precise.

---

## Output

`data/outputs/norway_world_cup_predictions.csv` — one row per fixture × team perspective × model:

| Column | Meaning |
|---|---|
| `fixture` | e.g. `Iraq vs Norway` (home side listed first) |
| `team` / `opponent` | which team's perspective this row represents |
| `model_name` | `logistic` / `poisson` / `ensemble` |
| `predicted_win_probability` | P(team wins) |
| `predicted_draw_probability` | P(draw) |
| `predicted_loss_probability` | P(team loses) |
| `predicted_result` | most likely outcome for `team` |
| `likely_scoreline` | most likely exact score (Poisson rows only) |

Latest ensemble consensus:

| Fixture | Result |
|---|---|
| Iraq vs Norway | Norway ~66% / draw ~21% / Iraq ~13% — most likely **Norway 1–0** |
| Norway vs Senegal | Norway ~42% / draw ~28% / Senegal ~30% — most likely **1–1** |
| Norway vs France | Norway ~36% / draw ~28% / France ~37% — near toss-up, France slight favourite |

---

## Project layout

```
norway-wc/
│
├── main.py                          End-to-end entry point (5 stages, see above)
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── eloratings/              Per-team TSV files downloaded from eloratings.net
│   │   └── flashscore_team_stats_raw.csv   Raw advanced stats (home/away; needs repair)
│   ├── input/                       Canonical inputs built by data_collection/
│   ├── processed/                   team_perspective.csv, model_dataset.csv
│   └── outputs/                     norway_world_cup_predictions.csv
│
├── src/
│   ├── data_collection/
│   │   ├── collect_elo.py           Downloads eloratings.net TSVs for all relevant teams
│   │   ├── prepare_flashscore.py    Projects raw Flashscore CSV onto the tidy schema
│   │   ├── prepare_fifa.py          Writes fifa_rankings.csv from hardcoded config values
│   │   └── build_inputs.py          Orchestrates the three collectors; idempotent
│   │
│   ├── preprocessing/
│   │   └── load_data.py             Loaders, team-perspective reshape, Flashscore repair
│   │
│   ├── features/
│   │   └── build_features.py        shift(1) rolling features, matchup diff columns,
│   │                                fixture isolation (each fixture built separately
│   │                                to prevent placeholder rows leaking into windows)
│   │
│   ├── models/
│   │   ├── logistic_model.py        sklearn Pipeline (impute → scale → LogisticRegression);
│   │   │                            time-split evaluation, calibration, fit_final()
│   │   └── poisson_model.py         Attack/defence strength estimation → expected goals
│   │                                → scipy Poisson scoreline grid → W/D/L
│   │
│   ├── prediction/
│   │   └── predict_fixtures.py      Scores all fixtures with both models; assembles
│   │                                the ensemble; writes + returns predictions DataFrame
│   │
│   └── utils/
│       ├── config.py                All paths, constants, feature lists, fixture definitions
│       ├── logging_utils.py         Structured logger factory
│       └── team_names.py            Name-standardization map (eloratings ↔ Flashscore)
│
├── scripts/
│   ├── flashscore_scraper.py        Playwright scraper — renders one match stats page,
│   │                                parses home/away rows, exposes scrape_flashscore_url()
│   ├── flashscore_collect_all.py    Bulk collector — discovers match URLs per team,
│   │                                calls the scraper, writes flashscore_team_stats_raw.csv
│   └── README.md                    Setup + usage for the Flashscore scraper
│
└── notebooks/
    └── iraq_vs_norway.ipynb         Interactive prediction for the 2026-06-16 fixture:
                                     W/D/L table, Poisson scoreline grid, feature importance
```
