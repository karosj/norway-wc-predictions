"""
Extract advanced match statistics from Flashscore match-stat pages.

Example page:
https://www.flashscore.com/match/football/iraq-K8aAGt6r/spain-bLyo6mco/summary/stats/overall/

Why Playwright?
Flashscore pages are rendered with JavaScript. Normal requests/BeautifulSoup often return
an empty shell, so this script opens the page in a real browser and reads the rendered DOM.

Install:
    pip install pandas playwright
    python -m playwright install chromium

Run one URL:
    python extract_flashscore_stats.py \
        --url "https://www.flashscore.com/match/football/iraq-K8aAGt6r/spain-bLyo6mco/summary/stats/overall/" \
        --home-team Iraq \
        --away-team Spain \
        --date 2024-01-01 \
        --competition "Friendly"

Run many URLs:
    python extract_flashscore_stats.py --input data/input/flashscore_urls.csv

Input CSV format:
    url,date,home_team,away_team,competition

Output:
    data/processed/flashscore_match_stats.csv

Notes:
    - This uses an unofficial scraping approach.
    - Flashscore may block automation or change their page structure.
    - Use carefully and respect Flashscore's terms/robots/rate limits.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "input"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw" / "flashscore"

DEFAULT_OUTPUT = PROCESSED_DIR / "flashscore_match_stats.csv"

KNOWN_STAT_LABELS = [
    # Top stats / shots
    "Expected goals (xG)",
    "xG on target (xGOT)",
    "Expected assists (xA)",
    "Ball possession",
    "Total shots",
    "Shots on target",
    "Shots off target",
    "Blocked shots",
    "Shots inside the box",
    "Shots outside the box",
    "Hit the woodwork",
    "Big chances",
    "Corner kicks",

    # Attack
    "Touches in opposition box",
    "Accurate through passes",
    "Offsides",
    "Free kicks",
    "Throw ins",

    # Passing
    "Passes",
    "Long passes",
    "Passes in final third",
    "Crosses",

    # Defence
    "Fouls",
    "Tackles",
    "Duels won",
    "Clearances",
    "Interceptions",
    "Errors leading to shot",
    "Errors leading to goal",
    "Yellow cards",
    "Red cards",

    # Goalkeeping
    "Goalkeeper saves",
    "xGOT faced",
    "Goals prevented",
]

STAT_COLUMN_MAP = {
    "Expected goals (xG)": "xg",
    "xG on target (xGOT)": "xgot",
    "Expected assists (xA)": "xa",
    "Ball possession": "possession_pct",
    "Total shots": "total_shots",
    "Shots on target": "shots_on_target",
    "Shots off target": "shots_off_target",
    "Blocked shots": "blocked_shots",
    "Shots inside the box": "shots_inside_box",
    "Shots outside the box": "shots_outside_box",
    "Hit the woodwork": "woodwork",
    "Big chances": "big_chances",
    "Corner kicks": "corners",
    "Touches in opposition box": "touches_opposition_box",
    "Accurate through passes": "accurate_through_passes",
    "Offsides": "offsides",
    "Free kicks": "free_kicks",
    "Throw ins": "throw_ins",
    "Passes": "passes",
    "Long passes": "long_passes",
    "Passes in final third": "final_third_passes",
    "Crosses": "crosses",
    "Fouls": "fouls",
    "Tackles": "tackles",
    "Duels won": "duels_won",
    "Clearances": "clearances",
    "Interceptions": "interceptions",
    "Errors leading to shot": "errors_leading_to_shot",
    "Errors leading to goal": "errors_leading_to_goal",
    "Yellow cards": "yellow_cards",
    "Red cards": "red_cards",
    "Goalkeeper saves": "goalkeeper_saves",
    "xGOT faced": "xgot_faced",
    "Goals prevented": "goals_prevented",
}


def ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def create_example_input() -> None:
    path = INPUT_DIR / "flashscore_urls.csv"

    if path.exists():
        return

    pd.DataFrame(
        [
            {
                "url": "https://www.flashscore.com/match/football/iraq-K8aAGt6r/spain-bLyo6mco/summary/stats/overall/",
                "date": "",
                "home_team": "Iraq",
                "away_team": "Spain",
                "competition": "",
            }
        ]
    ).to_csv(path, index=False)


def slug_from_url(url: str) -> str:
    # ?mid= is Flashscore's unique match ID — prefer it as the cache key.
    mid_match = re.search(r"[?&]mid=([A-Za-z0-9]+)", url)
    if mid_match:
        return mid_match.group(1)

    # Fall back to both path segments so home-vs-away collisions don't overwrite each other.
    match = re.search(r"/match/football/([^/?#]+)/([^/?#]+)/", url)
    if match:
        return f"{match.group(1)}__{match.group(2)}"

    safe = re.sub(r"[^A-Za-z0-9]+", "_", url).strip("_")
    return safe[:120]


def normalize_number(value: str) -> Optional[float]:
    """
    Converts Flashscore values to floats.

    Examples:
        "66%" -> 66.0
        "90% (616/681)" -> 90.0
        "616/681" -> 616.0
        "1.42" -> 1.42
        "13" -> 13.0
    """
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    if "%" in text:
        before_percent = text.split("%")[0].strip()
        try:
            return float(before_percent)
        except ValueError:
            pass

    ratio_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)

    if ratio_match:
        try:
            return float(ratio_match.group(1))
        except ValueError:
            pass

    number_match = re.search(r"-?\d+(?:\.\d+)?", text)

    if number_match:
        return float(number_match.group(0))

    return None


def parse_ratio(value: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parses values like:
        "90% (616/681)"
        "(616/681)"
        "616/681"

    Returns completed, attempted.
    """
    if value is None:
        return None, None

    text = str(value)
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)

    if not match:
        return None, None

    return float(match.group(1)), float(match.group(2))


def clean_label(label: str) -> str:
    return re.sub(r"\s+", " ", str(label)).strip()


def extract_rows_from_rendered_text(page_text: str) -> List[Dict[str, str]]:
    """
    Fallback parser using visible page text.

    Flashscore stat rows normally appear as:
        home_value
        stat_label
        away_value

    This parser searches for known labels and takes nearest values around them.
    """
    lines = [
        clean_label(line)
        for line in page_text.splitlines()
        if clean_label(line)
    ]

    rows = []

    for i, line in enumerate(lines):
        if line not in KNOWN_STAT_LABELS:
            continue

        # Find previous non-heading value and next non-heading value.
        # Headings like "Top stats", "Shots", "Attack", etc. are skipped naturally
        # because they are not numeric/stat values.
        if i - 1 < 0 or i + 1 >= len(lines):
            continue

        # Some stats (Passes, Tackles, Crosses, Long passes) split their value
        # across two lines: "89%" on line i-1 then "(336/378)" on line i-1.
        # Detect this: if lines[i-1] starts with "(" it's the ratio fragment, and
        # lines[i-2] holds the percentage prefix — combine them.
        home_value = lines[i - 1]
        if home_value.startswith("(") and i >= 2 and re.match(r"^[\d.]+", lines[i - 2]):
            home_value = f"{lines[i - 2]} {home_value}"

        away_value = lines[i + 1]
        if (
            i + 2 < len(lines)
            and lines[i + 2].startswith("(")
            and re.match(r"^[\d.]+", away_value)
        ):
            away_value = f"{away_value} {lines[i + 2]}"

        # Very light sanity check: at least one side should contain a number
        if normalize_number(home_value) is None and normalize_number(away_value) is None:
            continue

        rows.append(
            {
                "label": line,
                "home_value_raw": home_value,
                "away_value_raw": away_value,
            }
        )

    # Deduplicate repeated "Top stats" vs detailed sections by keeping first occurrence
    seen = set()
    deduped = []

    for row in rows:
        label = row["label"]

        if label in seen:
            continue

        seen.add(label)
        deduped.append(row)

    return deduped


def extract_rows_from_dom(page) -> List[Dict[str, str]]:
    """
    Primary parser.

    It searches the DOM for elements that look like stat rows by checking if the
    element's innerText contains a known stat label and at least three text lines.

    Some stats (Passes, Tackles, Crosses, Long passes) render their value across
    two lines, e.g. "89%" then "(336/378)". We join all lines before/after the
    label so both fragments are captured in a single value string.
    """
    rows = page.evaluate(
        """
        (knownLabels) => {
            const labels = new Set(knownLabels);
            const elements = Array.from(document.querySelectorAll('div'));
            const results = [];

            for (const el of elements) {
                const text = (el.innerText || '').trim();
                if (!text) continue;

                const lines = text.split('\\n')
                    .map(x => x.trim())
                    .filter(Boolean);

                if (lines.length < 3 || lines.length > 8) continue;

                for (const label of labels) {
                    if (!lines.includes(label)) continue;

                    const labelIndex = lines.indexOf(label);

                    if (labelIndex <= 0 || labelIndex >= lines.length - 1) continue;

                    // Join all lines before/after the label so two-line values
                    // like "89% (336/378)" are captured as a single string.
                    const homeValue = lines.slice(0, labelIndex).join(' ');
                    const awayValue = lines.slice(labelIndex + 1).join(' ');

                    results.push({
                        label: label,
                        home_value_raw: homeValue,
                        away_value_raw: awayValue
                    });
                }
            }

            return results;
        }
        """,
        KNOWN_STAT_LABELS,
    )

    seen = set()
    deduped = []

    for row in rows:
        key = row["label"]

        if key in seen:
            continue

        seen.add(key)
        deduped.append(row)

    return deduped


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


def scrape_flashscore_url(
    url: str,
    home_team: str,
    away_team: str,
    date: str = "",
    competition: str = "",
    headless: bool = True,
    slow_mo: int = 0,
) -> pd.DataFrame:
    """
    Scrapes one Flashscore stats URL into two team rows: home and away.
    """
    match_slug = slug_from_url(url)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

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

        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        accept_cookies_if_present(page)

        try:
            page.wait_for_load_state("networkidle", timeout=20000)
        except PlaywrightTimeoutError:
            pass

        # Give Flashscore a little extra time to render stats.
        time.sleep(3)

        raw_html_path = RAW_DIR / f"{match_slug}.html"
        raw_txt_path = RAW_DIR / f"{match_slug}.txt"

        # Retry page.content() — can fail if a redirect is still in flight.
        html_content = ""
        for _attempt in range(3):
            try:
                html_content = page.content()
                break
            except Exception:
                time.sleep(2)
        raw_html_path.write_text(html_content, encoding="utf-8")
        page_text = page.locator("body").inner_text(timeout=10000)
        raw_txt_path.write_text(page_text, encoding="utf-8")

        rows = extract_rows_from_dom(page)

        if not rows:
            rows = extract_rows_from_rendered_text(page_text)

        browser.close()

    if not rows:
        raise RuntimeError(
            "No stat rows extracted. Try running with --headed, check for captcha/consent, "
            "or inspect the saved raw text/html in data/raw/flashscore/."
        )

    home = {
        "source": "flashscore",
        "provider_match_id": match_slug,
        "source_url": url,
        "date": date,
        "competition": competition,
        "team": home_team,
        "opponent": away_team,
        "is_home": 1,
    }

    away = {
        "source": "flashscore",
        "provider_match_id": match_slug,
        "source_url": url,
        "date": date,
        "competition": competition,
        "team": away_team,
        "opponent": home_team,
        "is_home": 0,
    }

    for row in rows:
        label = row["label"]
        col = STAT_COLUMN_MAP.get(label, label.lower().replace(" ", "_"))

        home_raw = row["home_value_raw"]
        away_raw = row["away_value_raw"]

        home[col] = normalize_number(home_raw)
        away[col] = normalize_number(away_raw)

        home[f"{col}_raw"] = home_raw
        away[f"{col}_raw"] = away_raw

        # For passing-style ratio fields, also store completed and attempted
        home_completed, home_attempted = parse_ratio(home_raw)
        away_completed, away_attempted = parse_ratio(away_raw)

        if home_completed is not None or away_completed is not None:
            home[f"{col}_completed"] = home_completed
            home[f"{col}_attempted"] = home_attempted
            away[f"{col}_completed"] = away_completed
            away[f"{col}_attempted"] = away_attempted

    df = pd.DataFrame([home, away])
    df = add_opponent_and_diff_columns(df)

    return df


def add_opponent_and_diff_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    id_cols = {
        "source",
        "provider_match_id",
        "source_url",
        "date",
        "competition",
        "team",
        "opponent",
    }

    numeric_cols = [
        col for col in df.columns
        if col not in id_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    opponent_df = df[["provider_match_id", "team"] + numeric_cols].copy()
    opponent_df = opponent_df.rename(
        columns={
            "team": "opponent",
            **{col: f"opponent_{col}" for col in numeric_cols},
        }
    )

    out = df.merge(
        opponent_df,
        on=["provider_match_id", "opponent"],
        how="left",
    )

    for col in numeric_cols:
        if col == "is_home":
            continue

        opp_col = f"opponent_{col}"

        if opp_col in out.columns:
            out[f"{col}_diff"] = out[col] - out[opp_col]

    return out


def reparse_from_txt(discovered_csv: Path) -> pd.DataFrame:
    """
    Re-parse all saved .txt files in RAW_DIR using the current (fixed) text parser.

    Uses the discovered matches CSV to map slugs back to team names / metadata.
    No network access — fully offline.
    """
    txt_files = sorted(RAW_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {RAW_DIR}")

    # Build slug → metadata mapping from the discovered CSV.
    meta: dict = {}
    if discovered_csv.exists():
        disc = pd.read_csv(discovered_csv)
        for _, row in disc.iterrows():
            url = str(row.get("url", ""))
            # ?mid= style (preferred)
            m = re.search(r"[?&]mid=([A-Za-z0-9]+)", url)
            if m:
                meta[m.group(1)] = row
            # team1__team2 style fallback
            mm = re.search(r"/match/football/([^/?#]+)/([^/?#]+)/", url)
            if mm:
                meta[f"{mm.group(1)}__{mm.group(2)}"] = row

    all_rows = []
    for txt_path in txt_files:
        slug = txt_path.stem
        row_meta = meta.get(slug, {})
        home_team = str(row_meta.get("home_team", "")) if row_meta != {} else ""
        away_team = str(row_meta.get("away_team", "")) if row_meta != {} else ""
        date      = str(row_meta.get("date", ""))      if row_meta != {} else ""
        comp      = str(row_meta.get("competition", "")) if row_meta != {} else ""
        url       = str(row_meta.get("url", ""))       if row_meta != {} else ""

        if not home_team or not away_team:
            print(f"  skip {slug}: no team metadata found")
            continue

        page_text = txt_path.read_text(encoding="utf-8", errors="replace")
        rows = extract_rows_from_rendered_text(page_text)

        if not rows:
            print(f"  skip {slug}: no stat rows extracted from txt")
            continue

        home = {
            "source": "flashscore",
            "provider_match_id": slug,
            "source_url": url,
            "date": date,
            "competition": comp,
            "team": home_team,
            "opponent": away_team,
            "is_home": 1,
        }
        away = {
            "source": "flashscore",
            "provider_match_id": slug,
            "source_url": url,
            "date": date,
            "competition": comp,
            "team": away_team,
            "opponent": home_team,
            "is_home": 0,
        }

        for row in rows:
            label = row["label"]
            col = STAT_COLUMN_MAP.get(label, label.lower().replace(" ", "_"))
            home_raw = row["home_value_raw"]
            away_raw = row["away_value_raw"]
            home[col] = normalize_number(home_raw)
            away[col] = normalize_number(away_raw)
            home[f"{col}_raw"] = home_raw
            away[f"{col}_raw"] = away_raw
            h_c, h_a = parse_ratio(home_raw)
            a_c, a_a = parse_ratio(away_raw)
            if h_c is not None or a_c is not None:
                home[f"{col}_completed"] = h_c
                home[f"{col}_attempted"] = h_a
                away[f"{col}_completed"] = a_c
                away[f"{col}_attempted"] = a_a

        df = pd.DataFrame([home, away])
        df = add_opponent_and_diff_columns(df)
        all_rows.append(df)
        print(f"  parsed {slug}: {home_team} vs {away_team}")

    if not all_rows:
        raise RuntimeError("No rows reparsed. Check that txt files and discovered CSV are present.")

    return pd.concat(all_rows, ignore_index=True, sort=False)


def scrape_from_csv(input_path: Path, headless: bool, slow_mo: int) -> pd.DataFrame:
    matches = pd.read_csv(input_path)

    required = {"url", "home_team", "away_team"}
    missing = required - set(matches.columns)

    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    all_rows = []

    for _, row in matches.iterrows():
        print(f"Scraping: {row['home_team']} vs {row['away_team']}")

        df = scrape_flashscore_url(
            url=row["url"],
            home_team=row["home_team"],
            away_team=row["away_team"],
            date=str(row.get("date", "")),
            competition=str(row.get("competition", "")),
            headless=headless,
            slow_mo=slow_mo,
        )

        all_rows.append(df)

        # Be polite. Do not hammer Flashscore.
        time.sleep(3)

    return pd.concat(all_rows, ignore_index=True)


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print()
    print(f"Saved: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print()
    print(df.head())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--url", help="One Flashscore match stats URL.")
    parser.add_argument("--home-team", help="Home team name for one URL.")
    parser.add_argument("--away-team", help="Away team name for one URL.")
    parser.add_argument("--date", default="", help="Match date.")
    parser.add_argument("--competition", default="", help="Competition name.")

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="CSV with url,date,home_team,away_team,competition.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run browser visibly. Useful if Flashscore blocks headless mode.",
    )

    parser.add_argument(
        "--slow-mo",
        type=int,
        default=0,
        help="Slow browser actions in milliseconds. Useful for debugging.",
    )

    parser.add_argument(
        "--reparse-raw",
        action="store_true",
        help=(
            "Re-parse all saved .txt files in data/raw/flashscore/ with the current "
            "parser (no network). Requires data/processed/flashscore_discovered_matches.csv "
            "for team-name metadata."
        ),
    )

    parser.add_argument(
        "--discovered-csv",
        type=Path,
        default=Path("data/processed/flashscore_discovered_matches.csv"),
        help="Discovered matches CSV used for team-name lookup when --reparse-raw is set.",
    )

    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    create_example_input()

    args = parse_args()

    headless = not args.headed

    if args.reparse_raw:
        print(f"Re-parsing txt files from {RAW_DIR} ...")
        df = reparse_from_txt(args.discovered_csv)
    elif args.input is not None:
        df = scrape_from_csv(args.input, headless=headless, slow_mo=args.slow_mo)
    elif args.url:
        if not args.home_team or not args.away_team:
            raise ValueError("When using --url, you must also pass --home-team and --away-team.")

        df = scrape_flashscore_url(
            url=args.url,
            home_team=args.home_team,
            away_team=args.away_team,
            date=args.date,
            competition=args.competition,
            headless=headless,
            slow_mo=args.slow_mo,
        )
    else:
        raise ValueError("Pass either --url, --input, or --reparse-raw.")

    save_output(df, args.output)


if __name__ == "__main__":
    main()
