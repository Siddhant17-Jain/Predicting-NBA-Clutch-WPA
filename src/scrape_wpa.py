"""
Scrape NBA Player Win Probability Added (WPA) from inpredictable.com.

Strategy:
  - For each (season, variant): run up to 5 full attempts.
  - Each attempt scrapes groups 1..N until two consecutive empty groups are returned
    (max ceiling: grp=25, enough for 700+ players).
  - Final dataset per (season, variant) = UNION of all players found across all 5 attempts,
    keyed on player_id. This is the most comprehensive possible result.
  - Reports per-season row counts + how many attempts agreed.

Saves: data/raw/wpa_all_seasons.csv
"""

import time, sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

BASE_URL = "https://stats.inpredictable.com/nba/ssnPlayer.php"

PLAYOFF_VARIANTS = {"regular": 0, "playoffs": 1, "combined": 2}
SEASONS = list(range(2000, 2026))   # 2000-01 through 2025-26

MAX_GRP    = 25    # ceiling; stop early when two consecutive empty groups
N_ATTEMPTS = 5     # scrape attempts per (season, variant)
DELAY      = 0.75  # seconds between requests


def season_dates(start_year: int) -> tuple[str, str]:
    # 2019-20 bubble playoffs ran Aug–Oct 2020, past the standard Aug-01 cutoff
    end = "2020-11-01" if start_year == 2019 else f"{start_year + 1}-08-01"
    return f"{start_year}-10-01", end


def extract_player_id(href: str) -> str:
    if href and "pid=" in href:
        return href.split("pid=")[-1]
    return ""


def scrape_group(session: requests.Session, season: int,
                 grp: int, po: int) -> list[dict]:
    frdt, todt = season_dates(season)
    params = dict(season=season, team="ALL", pos="ALL", po=po,
                  frdt=frdt, todt=todt, rate="tot",
                  sort="sWPA", order="DESC", grp=grp)
    try:
        r = session.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"      Request error (season={season} grp={grp} po={po}): {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td", recursive=False)
        if not cells or len(cells) < 14:
            continue
        player_cell = cells[1]
        player_link = player_cell.find("a")
        if not player_link:
            continue
        try:
            row = {
                "rnk":       cells[0].get_text(strip=True),
                "player":    player_link.get_text(strip=True),
                "player_id": extract_player_id(player_link.get("href", "")),
                "pos":       cells[2].get_text(strip=True),
                "gms":       cells[3].get_text(strip=True),
                "wpa":       cells[4].get_text(strip=True),
                "ewpa":      cells[5].get_text(strip=True),
                "clwpa":     cells[6].get_text(strip=True),
                "gbwpa":     cells[7].get_text(strip=True),
                "sh":        cells[8].get_text(strip=True),
                "to":        cells[9].get_text(strip=True),
                "ft":        cells[10].get_text(strip=True),
                "reb":       cells[11].get_text(strip=True),
                "ast":       cells[12].get_text(strip=True),
                "stl":       cells[13].get_text(strip=True),
                "blk":       cells[14].get_text(strip=True) if len(cells) > 14 else "",
                "kwpa":      cells[15].get_text(strip=True) if len(cells) > 15 else "",
            }
            rows.append(row)
        except IndexError:
            continue
    return rows


def scrape_one_attempt(session: requests.Session, season: int, po: int) -> dict[str, dict]:
    """
    Scrape all groups for one (season, po) until two consecutive empty groups.
    Returns dict keyed by player_id -> row dict.
    """
    players: dict[str, dict] = {}
    consecutive_empty = 0

    for grp in range(1, MAX_GRP + 1):
        rows = scrape_group(session, season, grp, po)
        time.sleep(DELAY)

        if not rows:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            continue

        consecutive_empty = 0
        for row in rows:
            pid = row["player_id"]
            if pid and pid not in players:
                players[pid] = row

    return players


def scrape_season_variant(session: requests.Session, season: int,
                          variant_name: str, po_val: int) -> list[dict]:
    """
    Run N_ATTEMPTS full scrapes for (season, variant).
    Return union of all players found across all attempts.
    """
    all_players: dict[str, dict] = {}
    attempt_counts: list[int] = []

    for attempt in range(1, N_ATTEMPTS + 1):
        attempt_players = scrape_one_attempt(session, season, po_val)
        attempt_counts.append(len(attempt_players))

        for pid, row in attempt_players.items():
            if pid not in all_players:
                all_players[pid] = row

        sys.stdout.write(
            f"      attempt {attempt}/{N_ATTEMPTS}: {len(attempt_players)} players "
            f"(union so far: {len(all_players)})\n"
        )
        sys.stdout.flush()

    most_common = max(set(attempt_counts), key=attempt_counts.count)
    agreement   = attempt_counts.count(most_common)
    print(f"      counts: {attempt_counts}  "
          f"→ {agreement}/{N_ATTEMPTS} agree on {most_common}  "
          f"| final union: {len(all_players)}")

    return list(all_players.values())


def main():
    all_records = []
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    total_combos = len(SEASONS) * len(PLAYOFF_VARIANTS)
    done = 0

    for season in SEASONS:
        season_label = f"{season}-{season + 1}"
        for variant_name, po_val in PLAYOFF_VARIANTS.items():
            done += 1
            print(f"\n[{done}/{total_combos}] Season {season_label} | {variant_name} (po={po_val})")

            rows = scrape_season_variant(session, season, variant_name, po_val)
            for row in rows:
                row["season"]          = season_label
                row["playoff_variant"] = variant_name
            all_records.extend(rows)

    df = pd.DataFrame(all_records)

    col_order = ["season", "playoff_variant", "rnk", "player", "player_id", "pos", "gms",
                 "wpa", "ewpa", "clwpa", "gbwpa",
                 "sh", "to", "ft", "reb", "ast", "stl", "blk", "kwpa"]
    df = df[[c for c in col_order if c in df.columns]]

    for col in ["rnk", "gms", "wpa", "ewpa", "clwpa", "gbwpa",
                "sh", "to", "ft", "reb", "ast", "stl", "blk", "kwpa"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out_path = ROOT / "data" / "raw" / "wpa_all_seasons.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDone. {len(df)} rows written to {out_path}")

    print("\nPlayer counts per season × variant:")
    summary = df.groupby(["season", "playoff_variant"])["player_id"].count().unstack()
    print(summary.to_string())


if __name__ == "__main__":
    main()
