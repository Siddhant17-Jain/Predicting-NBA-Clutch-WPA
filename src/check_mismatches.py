"""
Cross-source name-matching audit: Basketball-Reference vs inpredictable.com.

Writes data/outputs/player_mismatches.xlsx with five sheets:
    Regular_BBRef_only   qualifying BBRef player-seasons with no WPA row
    Regular_WPA_only     WPA player-seasons with no qualifying BBRef row
    Playoff_BBRef_only
    Playoff_WPA_only
    Summary              counts and join rates

Run this after scraping any new season. A freshly scraped year will not have
its name spellings aligned with WPA yet, and this is how you find out which
ones need a rule in scrape_bbref.NAME_FIXES.

THRESHOLD ASYMMETRY (deliberate). The games threshold is applied to the BBRef
side ONLY; WPA is left completely unfiltered. BBRef is the authoritative
games-played source and the two sites occasionally disagree by a game or two,
so filtering both sides would report false mismatches for players sitting right
on the boundary (e.g. 62 games in BBRef, 61 in WPA).

The `*_WPA_only` sheets are therefore expected to be large and are NOT errors:
they list every WPA player-season below the BBRef threshold. Only the
`*_BBRef_only` sheets indicate genuine unmatched names needing attention.
"""

from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_xgboost import load_wpa, load_bbref, norm

ROOT  = Path(__file__).resolve().parent.parent
OUT_D = ROOT / "data" / "outputs"

VARIANTS = [
    # (variant, bbref stem, games threshold, sheet prefix)
    ("regular",  "seasons_stats_clean", 62, "Regular"),
    ("playoffs", "playoffStats_clean",  10, "Playoff"),
]


def audit(variant: str, stem: str, thr: int):
    wpa = load_wpa()
    w = wpa[wpa["playoff_variant"] == variant].copy()
    w["_n"] = w["player"].map(norm)

    b = load_bbref(stem)
    b["G"] = pd.to_numeric(b["G"], errors="coerce")
    b = b[b["G"] >= thr].copy()
    b["_n"] = b["Player"].map(norm)

    bkeys = set(zip(b["Year"], b["_n"]))
    wkeys = set(zip(w["yr"], w["_n"]))

    bb_only = (b[[not k in wkeys for k in zip(b["Year"], b["_n"])]]
               [["Year", "Player", "Tm", "G", "PTS"]]
               .sort_values(["Year", "Player"]))
    wpa_only = (w[[not k in bkeys for k in zip(w["yr"], w["_n"])]]
                [["yr", "player", "pos", "gms", "clwpa"]]
                .sort_values(["yr", "player"]))

    matched = len(bkeys & wkeys)
    stats = {
        "variant": variant,
        "games_threshold": thr,
        "bbref_qualifying": len(bkeys),
        "matched": matched,
        "join_rate": round(matched / len(bkeys), 6) if bkeys else 0.0,
        "bbref_only": len(bb_only),
        "wpa_only_below_threshold": len(wpa_only),
    }
    return bb_only, wpa_only, stats


def main():
    sheets, summary = {}, []
    for variant, stem, thr, prefix in VARIANTS:
        bb_only, wpa_only, stats = audit(variant, stem, thr)
        sheets[f"{prefix}_BBRef_only"] = bb_only
        sheets[f"{prefix}_WPA_only"]   = wpa_only
        summary.append(stats)
        print(f"{variant:9s} G>={thr:2d}: {stats['matched']:,}/"
              f"{stats['bbref_qualifying']:,} joined "
              f"({stats['join_rate']:.2%})  |  BBRef-only: {stats['bbref_only']}")
        if stats["bbref_only"]:
            print("    unmatched BBRef names — add a verified rule to "
                  "scrape_bbref.NAME_FIXES:")
            for r in bb_only.itertuples():
                print(f"      {r.Player} ({r.Year}, {r.G:.0f} G)")

    sheets["Summary"] = pd.DataFrame(summary)

    path = OUT_D / "player_mismatches.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
    print(f"\nWritten {path}")


if __name__ == "__main__":
    main()
