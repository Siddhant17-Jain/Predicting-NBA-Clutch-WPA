"""
Scrape Basketball-Reference regular season and playoff stats.
Writes into data/processed/seasons_stats_clean.csv and
data/processed/playoffStats_clean.csv.

URLs:
  Regular totals:  https://www.basketball-reference.com/leagues/NBA_{year}_totals.html
  Regular advanced:https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html
  Playoff totals:  https://www.basketball-reference.com/playoffs/NBA_{year}_totals.html
  Playoff advanced:https://www.basketball-reference.com/playoffs/NBA_{year}_advanced.html

YEAR CONVENTION — read before editing.
    Basketball-Reference URLs are keyed on the season's END year:
    NBA_2023_totals.html is the 2022-23 season. The processed files store the
    season's START year, matching the WPA `yr` field. So every scraped row gets

        Year = url_year - 1

    Earlier versions of this script wrote `Year = url_year`, which is the end
    year. Re-running that version today would append the 2025-26 season as
    Year=2026 alongside the existing (correct) Year=2025 row for the same
    season. The offset below is the fix; do not remove it.

REPLACE_YEARS — repairing a corrupted season.
    Years listed here are re-scraped and OVERWRITE any existing rows for that
    season, instead of being skipped as already-present.

    2022 (i.e. the 2021-22 season, stored as Year=2021) is listed because the
    original data/raw/seasons_stats.csv shipped with that season duplicated
    from 2020-21: raw Year 2021 and raw Year 2022 both contain 2020-21 totals
    (Joel Embiid 51 G / 1451 PTS in both, when his real 2021-22 was 68 G /
    2079 PTS). Verified affected: regular season only, that one year only.
    The playoff file was checked season by season and is correct throughout.
"""

import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from ftfy import fix_text
from unidecode import unidecode

ROOT = Path(__file__).resolve().parent.parent

# Basketball-Reference URL years (season END year). Stored as year - 1.
YEARS = [2023, 2024, 2025, 2026]

# URL years to force-refresh, overwriting existing rows. See docstring.
REPLACE_YEARS = {2022}

DELAY = 3  # seconds between requests

SESSION = requests.Session()
SESSION.headers["User-Agent"] = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# ── Column mapping: data-stat → target column name ────────────────────────────
# Regular season totals (new format uses name_display / team_name_abbr / games)
REG_TOTALS_MAP = {
    "name_display": "Player", "player": "Player",
    "team_name_abbr": "Tm",   "team_id": "Tm",
    "games": "G",             "g": "G",
    "games_started": "GS",    "gs": "GS",
    "pos": "Pos", "age": "Age", "mp": "MP",
    "fg": "FG", "fga": "FGA", "fg_pct": "FG%",
    "fg3": "3P", "fg3a": "3PA", "fg3_pct": "3P%",
    "fg2": "2P", "fg2a": "2PA", "fg2_pct": "2P%",
    "efg_pct": "eFG%",
    "ft": "FT", "fta": "FTA", "ft_pct": "FT%",
    "orb": "ORB", "drb": "DRB", "trb": "TRB",
    "ast": "AST", "stl": "STL", "blk": "BLK",
    "tov": "TOV", "pf": "PF", "pts": "PTS",
}

# Advanced (same for both regular and playoff)
ADV_MAP = {
    "name_display": "Player", "player": "Player",
    "team_name_abbr": "Tm",   "team_id": "Tm",
    "games": "G",             "g": "G",
    "games_started": "GS",    "gs": "GS",
    "pos": "Pos", "age": "Age", "mp": "MP",
    "per": "PER", "ts_pct": "TS%",
    "fg3a_per_fga_pct": "3PAr", "fta_per_fga_pct": "FTr",
    "orb_pct": "ORB%", "drb_pct": "DRB%", "trb_pct": "TRB%",
    "ast_pct": "AST%", "stl_pct": "STL%", "blk_pct": "BLK%",
    "tov_pct": "TOV%", "usg_pct": "USG%",
    "ows": "OWS", "dws": "DWS", "ws": "WS", "ws_per_48": "WS/48",
    "obpm": "OBPM", "dbpm": "DBPM", "bpm": "BPM", "vorp": "VORP",
}

SKIP_STATS = {"ranker", "DUMMY", "tpl_dbl", "awards"}

FINAL_COLS = [
    "Year", "Player", "Pos", "Age", "Tm", "G", "GS", "MP",
    "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%",
    "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%",
    "OWS", "DWS", "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP",
]

INT_COLS = ["G", "GS", "MP", "FG", "FGA", "3P", "3PA", "2P", "2PA",
            "FT", "FTA", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]


def fetch_table(url: str, table_id: str, col_map: dict) -> pd.DataFrame | None:
    print(f"  GET {url}")
    time.sleep(DELAY)
    try:
        r = SESSION.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"    ERROR: {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": table_id})
    if not table:
        # try first table if only one exists
        table = soup.find("table")
        if not table:
            print(f"    No table found (id={table_id})")
            return None

    # Parse headers from data-stat attributes
    header_row = table.find("thead").find_all("tr")[-1]
    raw_headers = [th.get("data-stat", "") for th in header_row.find_all("th")]
    headers = [col_map.get(h, h) for h in raw_headers if h not in SKIP_STATS]

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if "thead" in tr.get("class", []):
            continue  # skip repeat header rows
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        values = {}
        for th, cell in zip(raw_headers, [c.get_text(strip=True) for c in cells]):
            if th in SKIP_STATS:
                continue
            col = col_map.get(th, th)
            values[col] = cell if cell not in ("", "—") else np.nan
        if values.get("Player") and values["Player"] not in ("Player", ""):
            rows.append(values)

    df = pd.DataFrame(rows)
    # keep only mapped columns that exist
    keep = [c for c in headers if c in df.columns]
    return df[keep] if keep else df


COMBINED_FLAGS = {"TOT", "2TM", "3TM", "4TM"}

# Names where Basketball-Reference and inpredictable.com disagree, applied to
# the BBRef side so it matches WPA's spelling.
#
# ONLY put a rule here once it has been verified against the WPA data for the
# exact years it covers. WPA changes its own spelling partway through several
# careers, so a rule that is right for one season is wrong for another. An
# over-broad rule silently BREAKS rows that previously joined: applying
# "JJ Barea -> Jose Juan Barea" and "Otto Porter -> Otto Porter Jr. (2016+)"
# across the whole file once cost 8 correct matches, because WPA actually uses
# "JJ Barea" for 2008-2016 and "Otto Porter" in 2016.
#
# After scraping a NEW season, run the join check in build_xgboost.join_datasets
# and resolve whatever comes back unmatched rather than guessing a rule here.
#
#   (bbref_name, year_scope, wpa_name)   year_scope None = all years
NAME_FIXES = [
    ("Nene Hilario",         None,                     "Nene"),
    ("Luc Mbah",             None,                     "Luc Mbah a Moute"),
    ("Metta World",          None,                     "Metta World Peace"),
    ("Chuck Hayes",          None,                     "Charles Hayes"),
    ("Stanislav Medvedenko", None,                     "Slava Medvedenko"),
    ("Mike Sweetney",        None,                     "Michael Sweetney"),
    ("Alex Sarr",            None,                     "Alexandre Sarr"),
    ("Enes Freedom",         None,                     "Enes Kanter"),
    # The two BBRef files disagree with each other: the regular-season file says
    # "JJ Barea", the playoff file says "Jose Juan Barea". norm() strips periods,
    # so "JJ Barea" and WPA's "J.J. Barea" both reduce to "jj barea" and match —
    # only the playoff spelling is the outlier. Normalising it here fixes the
    # 2008 and 2010 playoff rows.
    ("Jose Juan Barea",      None,                     "JJ Barea"),
    # Verified year-scoped: WPA uses the Jr. form for 2020-2022 only, and
    # reverts to "KJ Martin" from 2023 onward.
    ("KJ Martin",            {2020, 2021, 2022},       "Kenyon Martin Jr."),
    ("Lou Williams",         {2011, 2012, 2013},       "Louis Williams"),
    ("Jimmy Butler",         lambda y: y >= 2024,      "Jimmy Butler III"),
    ("Otto Porter",          lambda y: y >= 2017,      "Otto Porter Jr."),
    ("Marcus Morris",        lambda y: y >= 2019,      "Marcus Morris Sr."),
    ("Timothe Luwawu-Cabarrot", {2016},                "Timothe Luwawu"),
]


def clean_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Repair name encoding and align spellings with the WPA source.

    Basketball-Reference serves double-encoded UTF-8 for accented names
    ("Nikola JokiÄ\\x87"). ftfy repairs the mojibake, unidecode flattens the
    result to ASCII so it matches the WPA side. Trailing Hall-of-Fame asterisks
    are stripped. Without this, freshly scraped seasons silently fail to join.
    """
    df = df.copy()
    df["Player"] = (df["Player"].astype(str)
                    .map(lambda s: unidecode(fix_text(s)).replace("*", "").strip()))

    for bbref_name, when, wpa_name in NAME_FIXES:
        hit = df["Player"] == bbref_name
        if not hit.any():
            continue
        if when is None:
            sel = hit
        elif isinstance(when, set):
            sel = hit & df["Year"].isin(when)
        else:
            sel = hit & df["Year"].map(when)
        df.loc[sel, "Player"] = wpa_name
    return df


def dedup_tot(df: pd.DataFrame) -> pd.DataFrame:
    """Keep combined row (TOT/2TM/3TM/4TM) for traded players; drop per-team splits."""
    combined_keys = set(
        zip(df[df["Tm"].isin(COMBINED_FLAGS)]["Year"],
            df[df["Tm"].isin(COMBINED_FLAGS)]["Player"])
    )
    mask = df.apply(
        lambda r: (r["Year"], r["Player"]) in combined_keys and r["Tm"] not in COMBINED_FLAGS,
        axis=1,
    )
    return df[~mask].reset_index(drop=True)


def scrape_season(year: int, variant: str) -> pd.DataFrame | None:
    """Scrape totals + advanced for one year/variant, merge, return clean DataFrame."""
    if variant == "regular":
        tot_url = f"https://www.basketball-reference.com/leagues/NBA_{year}_totals.html"
        adv_url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
        tot_id  = "totals_stats"
        adv_id  = "advanced"
    else:
        tot_url = f"https://www.basketball-reference.com/playoffs/NBA_{year}_totals.html"
        adv_url = f"https://www.basketball-reference.com/playoffs/NBA_{year}_advanced.html"
        tot_id  = "totals_stats"
        adv_id  = "advanced_stats"

    tot = fetch_table(tot_url, tot_id, REG_TOTALS_MAP)
    adv = fetch_table(adv_url, adv_id, ADV_MAP)

    if tot is None or tot.empty:
        print(f"    No totals data for {year} {variant}")
        return None

    # BBRef URL year is the season END year; store the START year. See docstring.
    tot["Year"] = year - 1

    # Numeric conversion for totals
    for col in INT_COLS:
        if col in tot.columns:
            tot[col] = pd.to_numeric(tot[col], errors="coerce")

    if adv is not None and not adv.empty:
        adv_key_cols = [c for c in ["Player", "Tm"] if c in adv.columns]
        adv_extra = [c for c in adv.columns if c not in tot.columns and c not in adv_key_cols]
        merged = tot.merge(adv[adv_key_cols + adv_extra], on=adv_key_cols, how="left")
    else:
        merged = tot

    # Numeric conversion for advanced cols
    adv_num_cols = ["PER","TS%","3PAr","FTr","ORB%","DRB%","TRB%","AST%","STL%",
                    "BLK%","TOV%","USG%","OWS","DWS","WS","WS/48","OBPM","DBPM","BPM","VORP"]
    for col in adv_num_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Age
    if "Age" in merged.columns:
        merged["Age"] = pd.to_numeric(merged["Age"], errors="coerce")

    # Keep only final columns present
    out_cols = [c for c in FINAL_COLS if c in merged.columns]
    return clean_names(merged[out_cols])


def main():
    reg_path = ROOT / "data" / "processed" / "seasons_stats_clean.csv"
    po_path  = ROOT / "data" / "processed" / "playoffStats_clean.csv"

    reg_existing = pd.read_csv(reg_path)
    po_existing  = pd.read_csv(po_path)

    print(f"Existing regular rows: {len(reg_existing)}, playoffs: {len(po_existing)}")
    print(f"Existing regular years: {sorted(reg_existing['Year'].unique())[-3:]}")
    print(f"Existing playoff years: {sorted(po_existing['Year'].unique())[-3:]}")

    new_reg_frames = []
    new_po_frames  = []

    for year in sorted(set(YEARS) | REPLACE_YEARS):
        print(f"\n=== {year} REGULAR ===")
        df = scrape_season(year, "regular")
        if df is not None:
            print(f"  Scraped {len(df)} rows before dedup")
            df = dedup_tot(df)
            print(f"  {len(df)} rows after TOT dedup")
            new_reg_frames.append(df)

        print(f"\n=== {year} PLAYOFFS ===")
        df = scrape_season(year, "playoffs")
        if df is not None:
            print(f"  Scraped {len(df)} rows before dedup")
            df = dedup_tot(df)
            print(f"  {len(df)} rows after TOT dedup")
            new_po_frames.append(df)

    # Start years being force-refreshed (REPLACE_YEARS holds URL/end years).
    replace_start_years = {y - 1 for y in REPLACE_YEARS}

    def merge_and_write(existing, frames, path, tag):
        if not frames:
            return
        new = pd.concat(frames, ignore_index=True)
        existing_years = set(existing["Year"].unique()) - replace_start_years
        new = new[~new["Year"].isin(existing_years)]

        replaced = sorted(replace_start_years & set(new["Year"].unique()))
        if replaced:
            before = len(existing)
            existing = existing[~existing["Year"].isin(replaced)]
            print(f"  {tag}: dropped {before - len(existing)} stale rows for "
                  f"replaced year(s) {replaced}")

        combined = (pd.concat([existing, new], ignore_index=True)
                    .sort_values(["Year", "Player"])
                    .reset_index(drop=True))
        combined.to_csv(path, index=False)
        print(f"Written {path} — {len(combined)} rows (+{len(new)} scraped)")

    print()
    merge_and_write(reg_existing, new_reg_frames, reg_path, "regular")
    merge_and_write(po_existing,  new_po_frames,  po_path,  "playoffs")


if __name__ == "__main__":
    main()
