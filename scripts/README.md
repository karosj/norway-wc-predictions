# scripts/ — Flashscore advanced-stats scraper

These two scripts are how the advanced match stats (xG, xGOT, shots, possession,
passing, defensive stats, …) in `data/raw/flashscore_team_stats_raw.csv` were
produced. The main pipeline (`py main.py`) **reuses that cached CSV** — you do
not need to run these to make predictions. They are kept for provenance and so
the extraction can be refreshed if needed.

> Re-scraping Flashscore is brittle: pages are JavaScript-rendered and the site
> may block automation or change structure. Run sparingly and respect their
> terms / robots / rate limits.

## Files

| File | Role |
|---|---|
| `flashscore_scraper.py` | Core scraper. Renders one match's `…/summary/stats/overall/` page with Playwright and parses the stat rows into two team rows (home + away). Exposes `scrape_flashscore_url(...)`. |
| `flashscore_collect_all.py` | Bulk collector. Opens each team's *Results* page, clicks "Show more", discovers match URLs, then calls the scraper for each. Writes `data/raw/flashscore_team_stats_raw.csv` (the pipeline's raw input). |

## Setup

```bash
pip install playwright tqdm        # in addition to requirements.txt
python -m playwright install chromium
```

## Usage (run from the project root)

```bash
# Bulk: discover + scrape every team in data/input/flashscore_team_urls.csv
py scripts/flashscore_collect_all.py --headed

# One match only
py scripts/flashscore_scraper.py \
    --url "https://www.flashscore.com/match/football/iraq-K8aAGt6r/spain-bLyo6mco/summary/stats/overall/" \
    --home-team Iraq --away-team Spain --date 2024-01-01
```

`--headed` runs a visible browser (recommended — Flashscore often blocks
headless). After collection, the pipeline's `src/data_collection/prepare_flashscore.py`
projects the raw CSV onto the tidy schema, and `src/preprocessing/load_data.repair_flashscore`
fixes mis-attributed rows using eloratings scorelines as ground truth.
