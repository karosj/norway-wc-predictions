"""
Collect Flashscore advanced stats for national teams since a chosen date.

This script:
1. Opens each team's Flashscore RESULTS page
2. Clicks "Show more matches" repeatedly
3. Collects match URLs
4. Converts them to /summary/stats/overall/ URLs
5. Uses flashscore_scraper.py to scrape xG, shots, possession, passing,
   defensive stats, etc.
6. Saves one CSV with one row per team per match (the pipeline's raw input,
   data/raw/flashscore_team_stats_raw.csv)

IMPORTANT:
    This script requires flashscore_scraper.py to be in the same folder.

Install:
    pip install pandas playwright tqdm
    python -m playwright install chromium

Run (from the project root):
    py scripts/flashscore_collect_all.py --headed

Input:
    data/input/flashscore_team_urls.csv

Format:
    team,results_url
    Iraq,https://www.flashscore.com/team/iraq/K8aAGt6r/results/
    Norway,https://www.flashscore.com/team/norway/8rP6JO0H/results/
    Senegal,https://www.flashscore.com/team/senegal/hOIsJLJr/results/
    France,https://www.flashscore.com/team/france/QkGeVG1n/results/

How to find a results URL:
    1. Go to flashscore.com
    2. Search for the national team, e.g. "Norway"
    3. Open the football team page
    4. Click Results
    5. Copy the URL
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm

# This must be in the same directory as this script.
from flashscore_scraper import scrape_flashscore_url


DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw" / "flashscore_discovery"

DEFAULT_TEAMS_CSV = INPUT_DIR / "flashscore_team_urls.csv"
DEFAULT_DISCOVERED_MATCHES = PROCESSED_DIR / "flashscore_discovered_matches.csv"
# Write straight to the path the pipeline consumes (config.FLASHSCORE_RAW_CSV).
DEFAULT_OUTPUT = DATA_DIR / "raw" / "flashscore_team_stats_raw.csv"


DEFAULT_TEAMS = [
    {
        "team": "Iraq",
        "results_url": "https://www.flashscore.com/team/iraq/K8aAGt6r/results/",
    },
    {
        "team": "Norway",
        "results_url": "PASTE_NORWAY_RESULTS_URL_HERE",
    },
    {
        "team": "Senegal",
        "results_url": "PASTE_SENEGAL_RESULTS_URL_HERE",
    },
    {
        "team": "France",
        "results_url": "PASTE_FRANCE_RESULTS_URL_HERE",
    },
]


def ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw" / "flashscore").mkdir(parents=True, exist_ok=True)


def create_team_template() -> None:
    if DEFAULT_TEAMS_CSV.exists():
        return

    pd.DataFrame(DEFAULT_TEAMS).to_csv(DEFAULT_TEAMS_CSV, index=False)


def is_placeholder_url(url: str) -> bool:
    if pd.isna(url):
        return True

    text = str(url).strip()

    return (
        text == ""
        or text.startswith("PASTE_")
        or text.lower() in {"nan", "none", "null"}
    )


def load_team_urls(path: Path) -> pd.DataFrame:
    if not path.exists():
        create_team_template()

    df = pd.read_csv(path)

    required = {"team", "results_url"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    missing_urls = df[df["results_url"].apply(is_placeholder_url)]

    if not missing_urls.empty:
        teams = ", ".join(missing_urls["team"].astype(str).tolist())

        raise ValueError(
            f"Missing Flashscore results URLs for: {teams}\n\n"
            f"Fill this file first:\n{path}\n\n"
            "Example format:\n"
            "team,results_url\n"
            "Iraq,https://www.flashscore.com/team/iraq/K8aAGt6r/results/\n"
            "Norway,https://www.flashscore.com/team/norway/TEAM_ID/results/\n"
        )

    return df


def accept_cookies_if_present(page) -> None:
    possible_buttons = [
        "Accept all",
        "I accept",
        "Accept",
        "Agree",
        "OK",
    ]

    for text in possible_buttons:
        try:
            page.get_by_role("button", name=re.compile(text, re.I)).click(timeout=2000)
            time.sleep(1)
            return
        except Exception:
            pass


def normalize_match_stats_url(href: str) -> str:
    """
    Converts any Flashscore match URL to the stats overall URL.
    Preserves ?mid=<id> because Flashscore needs it to identify the specific match.
    """
    if not href.startswith("http"):
        href = urljoin("https://www.flashscore.com", href)

    # Capture ?mid= before the regex strips query params.
    mid_match = re.search(r"[?&]mid=([A-Za-z0-9]+)", href)
    mid_param = f"?mid={mid_match.group(1)}" if mid_match else ""

    # Keep only the match slug part.
    match = re.search(r"(https://www\.flashscore\.com/match/football/[^/]+/[^/]+/)", href)

    if match:
        base = match.group(1)
    else:
        # Sometimes links may be /match/football/<slug>/#/match-summary
        match = re.search(r"(https://www\.flashscore\.com/match/football/[^#?]+)", href)
        if not match:
            return href
        base = match.group(1).rstrip("/") + "/"

    return base + "summary/stats/overall/" + mid_param


def parse_teams_from_match_url(url: str) -> Dict[str, str]:
    """
    Example:
    /match/football/iraq-K8aAGt6r/spain-bLyo6mco/summary/stats/overall/
    -> Iraq, Spain
    """
    match = re.search(r"/match/football/([^/]+)/([^/]+)/", url)

    if not match:
        return {"home_team": "", "away_team": ""}

    home_slug = match.group(1)
    away_slug = match.group(2)

    def clean(slug: str) -> str:
        # Remove final Flashscore ID segment after the last hyphen
        parts = slug.split("-")
        if len(parts) > 1:
            parts = parts[:-1]

        return " ".join(word.capitalize() for word in parts)

    return {
        "home_team": clean(home_slug),
        "away_team": clean(away_slug),
    }


def parse_date_from_text(text: str) -> Optional[str]:
    """
    Tries to parse dates from Flashscore match row text.

    Common forms:
        15.10.2024
        15.10. 20:45
        15.10.
    """
    text = str(text)

    full = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", text)

    if full:
        day, month, year = full.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    return None


def discover_match_links_for_team(
    team: str,
    results_url: str,
    from_date: str,
    max_show_more_clicks: int,
    headless: bool,
    slow_mo: int,
) -> pd.DataFrame:
    """
    Opens a team results page and collects match URLs.

    Date filtering is best-effort because Flashscore's results DOM changes.
    We still save all discovered links; date can later be corrected from the
    match page or manually.
    """
    from_date_ts = pd.to_datetime(from_date)

    discovered: Dict[str, Dict[str, str]] = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, slow_mo=slow_mo)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            viewport={"width": 1400, "height": 1000},
        )
        page = context.new_page()

        print(f"\nOpening results page for {team}")
        page.goto(results_url, wait_until="domcontentloaded", timeout=60000)
        accept_cookies_if_present(page)

        try:
            page.wait_for_load_state("networkidle", timeout=20000)
        except PlaywrightTimeoutError:
            pass

        time.sleep(3)

        for click_no in range(max_show_more_clicks + 1):
            raw_html = page.content()
            (RAW_DIR / f"{team.lower()}_{click_no}.html").write_text(raw_html, encoding="utf-8")

            links = page.evaluate(
                """
                () => {
                    const anchors = Array.from(document.querySelectorAll('a[href*="/match/football/"]'));
                    return anchors.map(a => {
                        const row = a.closest('[class*="event__match"]') || a.parentElement;
                        return {
                            href: a.href,
                            text: row ? row.innerText : a.innerText
                        };
                    });
                }
                """
            )

            for link in links:
                href = link.get("href", "")
                text = link.get("text", "")

                if "/match/football/" not in href:
                    continue

                stats_url = normalize_match_stats_url(href)
                date = parse_date_from_text(text) or ""

                parsed = parse_teams_from_match_url(stats_url)

                # If we managed to parse a date older than from_date, keep it out.
                if date:
                    try:
                        if pd.to_datetime(date) < from_date_ts:
                            continue
                    except Exception:
                        pass

                discovered[stats_url] = {
                    "source_team": team,
                    "url": stats_url,
                    "date": date,
                    "home_team": parsed["home_team"],
                    "away_team": parsed["away_team"],
                    "competition": "",
                    "row_text": text.replace("\n", " | "),
                }

            print(f"{team}: discovered {len(discovered)} unique match URLs after {click_no} clicks")

            # Try clicking "Show more matches"
            clicked = False

            selectors = [
                "text=Show more matches",
                "text=Show more",
                "button:has-text('Show more')",
                "a:has-text('Show more')",
            ]

            for selector in selectors:
                try:
                    button = page.locator(selector).last
                    if button.is_visible(timeout=2000):
                        button.click(timeout=5000)
                        clicked = True
                        time.sleep(3)
                        break
                except Exception:
                    pass

            if not clicked:
                break

        browser.close()

    df = pd.DataFrame(discovered.values())
    return df


def discover_all_matches(
    team_urls_df: pd.DataFrame,
    from_date: str,
    max_show_more_clicks: int,
    headless: bool,
    slow_mo: int,
) -> pd.DataFrame:
    all_matches = []

    for _, row in team_urls_df.iterrows():
        team = row["team"]
        url = row["results_url"]

        df = discover_match_links_for_team(
            team=team,
            results_url=url,
            from_date=from_date,
            max_show_more_clicks=max_show_more_clicks,
            headless=headless,
            slow_mo=slow_mo,
        )

        if not df.empty:
            all_matches.append(df)

        time.sleep(3)

    if not all_matches:
        return pd.DataFrame()

    matches = pd.concat(all_matches, ignore_index=True)

    # One match may appear on both teams' pages. Deduplicate by URL.
    matches = matches.drop_duplicates(subset=["url"]).reset_index(drop=True)
    matches.to_csv(DEFAULT_DISCOVERED_MATCHES, index=False)

    print(f"\nSaved discovered matches: {DEFAULT_DISCOVERED_MATCHES}")
    print(f"Unique discovered matches: {len(matches)}")

    return matches


def scrape_discovered_matches(
    discovered_df: pd.DataFrame,
    output_path: Path,
    headless: bool,
    slow_mo: int,
    limit: Optional[int],
) -> pd.DataFrame:
    rows = []

    if limit is not None:
        discovered_df = discovered_df.head(limit).copy()

    for _, row in tqdm(discovered_df.iterrows(), total=len(discovered_df), desc="Scraping match stats"):
        url = row["url"]
        home_team = row.get("home_team", "")
        away_team = row.get("away_team", "")

        # If URL parsing failed, use source team and unknown opponent.
        if not home_team or not away_team:
            home_team = row.get("source_team", "")
            away_team = "Unknown"

        try:
            df = scrape_flashscore_url(
                url=url,
                home_team=home_team,
                away_team=away_team,
                date=str(row.get("date", "")),
                competition=str(row.get("competition", "")),
                headless=headless,
                slow_mo=slow_mo,
            )

            # Keep track of which target team's page found the match
            df["source_team_page"] = row.get("source_team", "")
            rows.append(df)

        except Exception as exc:
            print(f"\nFailed scraping {url}")
            print(f"Reason: {exc}")

        # Be polite. Do not hammer Flashscore.
        time.sleep(3)

    if not rows:
        print("No match stats scraped.")
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True, sort=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    print(f"\nSaved final stats: {output_path}")
    print(f"Rows: {len(combined)}")
    print(f"Columns: {len(combined.columns)}")

    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--teams-csv",
        type=Path,
        default=DEFAULT_TEAMS_CSV,
        help="CSV with team,results_url",
    )

    parser.add_argument(
        "--from-date",
        default="2021-01-01",
        help="Only keep matches from this date onward where date can be parsed.",
    )

    parser.add_argument(
        "--max-show-more-clicks",
        type=int,
        default=20,
        help="How many times to click Show more matches on each team page.",
    )

    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Only collect match URLs, do not scrape stats.",
    )

    parser.add_argument(
        "--use-discovered",
        type=Path,
        default=None,
        help="Skip discovery and scrape this discovered matches CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Final output CSV.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of matches to scrape, useful for testing.",
    )

    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser visibly. Recommended for Flashscore.",
    )

    parser.add_argument(
        "--slow-mo",
        type=int,
        default=0,
        help="Slow browser actions in ms, useful for debugging.",
    )

    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    create_team_template()

    args = parse_args()
    headless = not args.headed

    if args.use_discovered is not None:
        discovered = pd.read_csv(args.use_discovered)
    else:
        team_urls = load_team_urls(args.teams_csv)

        discovered = discover_all_matches(
            team_urls_df=team_urls,
            from_date=args.from_date,
            max_show_more_clicks=args.max_show_more_clicks,
            headless=headless,
            slow_mo=args.slow_mo,
        )

    if discovered.empty:
        print("No matches discovered.")
        return

    if args.discover_only:
        return

    scrape_discovered_matches(
        discovered_df=discovered,
        output_path=args.output,
        headless=headless,
        slow_mo=args.slow_mo,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
