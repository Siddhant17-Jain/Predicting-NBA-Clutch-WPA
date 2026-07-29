"""
Step 1b: Build the canonical dataset from data/raw/wpa_all_seasons.csv.

  - Keeps season, playoff_variant, player, player_id, pos, gms, clwpa
  - Repairs corrupted player names (see NAME_REPAIRS below)
  - Deduplicates on (season, playoff_variant, player_id), summing traded-player splits
  - Adds `yr` (int) sort key
  - Writes data/processed/cwpa_long.xlsx   <- THE canonical source
  - Rebuilds data/raw/wpa_all_seasons.xlsx (with player_id)

NO cwpa_long.csv IS WRITTEN, DELIBERATELY.
    A CSV copy used to sit beside the workbook and was what eda.py and
    analysis.py read. Because the name repair below had only ever been applied
    by hand to the workbook, the CSV kept 52 corrupted rows covering 27 players
    and the two files disagreed. Every consumer now loads the workbook through
    stats_utils.load_cwpa(), so there is exactly one source of truth. Do not
    reintroduce a second serialisation of this table.

NAME ENCODING — why an explicit map is unavoidable.
    inpredictable.com serves accented names inconsistently across seasons, in
    two mutually incompatible ways, and BOTH are lossy:

      1. "?" substitution   Nikola Joki?, Luka Don?i?
         The original byte is gone and "?" is valid ASCII, so ftfy cannot see
         a problem and unidecode has nothing to transliterate.

      2. U+FFFD substitution   Chris Ma<fffd>on, Mont<fffd> Morris
         unidecode DROPS the replacement character rather than guessing, which
         silently yields "Chris Maon" and "Mont Morris".

    ftfy + unidecode alone therefore cannot repair either case; running them
    produces plausible-looking but wrong names. The map below is applied first,
    keyed on the exact corrupted string, and ftfy + unidecode then handle the
    ordinary well-formed accents (Nikola Jokic, Jonas Valanciunas) that some
    seasons return correctly.

    Six names in the previous hand-repaired workbook had the accent dropped
    rather than transliterated (Pacme Dadiet, Mont Morris, Tidjane Salan, Chris
    Maon, Egor Dmin, Yanic Konan Niederhuser). All six sit below every analysis
    threshold so no published result depended on them, but they are corrected
    here. Target spellings match Basketball-Reference so the cross-source join
    keeps working.
"""

from pathlib import Path

import pandas as pd
from ftfy import fix_text
from unidecode import unidecode

ROOT = Path(__file__).resolve().parent.parent

# Corrupted string -> correct ASCII spelling. Keyed on the exact raw value.
# Both "?" and U+FFFD ("�") variants are listed because different seasons
# return different corruptions for the same player.
NAME_REPAIRS = {
    "Bogdan Bogdanovi?":        "Bogdan Bogdanovic",
    "Chris Ma�on":         "Chris Macon",
    "Dario �ari?":         "Dario Saric",
    "Dennis Schr�der":     "Dennis Schroder",
    "Egor D�min":          "Egor Demin",
    "Hugo Gonz�lez":       "Hugo Gonzalez",
    "Jonas Valan?i?nas":        "Jonas Valanciunas",
    "Jusuf Nurki?":             "Jusuf Nurkic",
    "Karlo Matkovi?":           "Karlo Matkovic",
    "Kasparas Jaku?ionis":      "Kasparas Jakucionis",
    "Kristaps Porzi??is":       "Kristaps Porzingis",
    "Luka Don?i?":              "Luka Doncic",
    "Mont� Morris":        "Monte Morris",
    "Moussa Diabat�":      "Moussa Diabate",
    "Nikola Joki?":             "Nikola Jokic",
    "Nikola Jovi?":             "Nikola Jovic",
    "Nikola Topi?":             "Nikola Topic",
    "Nikola Vu?evi?":           "Nikola Vucevic",
    "Pac�me Dadiet":       "Pacome Dadiet",
    "Tidjane Sala�n":      "Tidjane Salaun",
    "V�t Krej?�":     "Vit Krejci",
    "Yanic Konan Niederh�user": "Yanic Konan Niederhauser",
}


# WPA-side spelling alignments. These are NOT encoding repairs — the raw value
# is well-formed, it simply uses a different form of the name than
# Basketball-Reference, so the cross-source join in build_xgboost.py misses.
#
# The BBRef-side equivalents live in scrape_bbref.NAME_FIXES. A name is fixed on
# whichever side is the outlier; this dict is for the cases where WPA is.
#
# Barea: BBRef says "JJ Barea", WPA says "Jose Juan Barea". Both normalise to
# "jj barea" once periods are stripped, so aligning here fixes 7 player-seasons
# (2008-2015) that otherwise fail to join. This was previously a hand edit to
# the workbook and was silently lost the first time this script was re-run.
ALIGNMENT_FIXES = {
    "Jose Juan Barea": "J.J. Barea",
}


def repair_name(name) -> str:
    """Explicit corruption map, then ftfy + unidecode, then spelling alignment."""
    if not isinstance(name, str):
        return name
    s = NAME_REPAIRS.get(name.strip(), name)
    s = unidecode(fix_text(s)).strip()
    return ALIGNMENT_FIXES.get(s, s)


def main():
    src = ROOT / "data" / "raw" / "wpa_all_seasons.csv"
    df = pd.read_csv(src, dtype={"player_id": str})
    print(f"Raw rows: {len(df)}")

    keep = ["season", "playoff_variant", "player", "player_id", "pos", "gms", "clwpa"]
    df = df[keep].copy()
    df["player_id"] = df["player_id"].fillna("").astype(str).str.strip()

    def dirty(s):
        s = str(s)
        return any(ord(c) > 127 for c in s) or "?" in s

    before = int(df["player"].map(dirty).sum())
    df["player"] = df["player"].map(repair_name)
    after = int(df["player"].map(dirty).sum())
    print(f"Name repair: {before} corrupted rows -> {after}")
    if after:
        left = sorted(df.loc[df["player"].map(dirty), "player"].unique())
        raise SystemExit(
            "Unrepaired names remain; add them to NAME_REPAIRS:\n  "
            + "\n  ".join(repr(x) for x in left)
        )

    has_id, no_id = df[df["player_id"] != ""], df[df["player_id"] == ""]
    agg = (has_id.groupby(["season", "playoff_variant", "player_id"], as_index=False)
                 .agg(player=("player", "first"), pos=("pos", "first"),
                      gms=("gms", "sum"), clwpa=("clwpa", "sum")))

    clean = pd.concat([agg, no_id[keep]], ignore_index=True)
    clean["yr"] = clean["season"].str[:4].astype(int)

    dups = clean.duplicated(subset=["season", "playoff_variant", "player_id"]).sum()
    assert dups == 0, f"Dedup failed — {dups} duplicates remain"
    print(f"Duplicate (season, variant, player_id) rows after dedup: {dups}")

    clean = clean.sort_values(["yr", "playoff_variant", "player"]).reset_index(drop=True)

    variants = [("regular", "Regular"), ("playoffs", "Playoffs"), ("combined", "Combined")]

    long_xlsx = ROOT / "data" / "processed" / "cwpa_long.xlsx"
    with pd.ExcelWriter(long_xlsx, engine="openpyxl") as w:
        for variant, sheet in variants:
            sub = (clean[clean["playoff_variant"] == variant]
                   .drop(columns="playoff_variant").reset_index(drop=True))
            sub.to_excel(w, sheet_name=sheet, index=False)
            print(f"  {sheet}: {len(sub)} rows")
    print(f"Written {long_xlsx}  ({len(clean)} rows total)")

    out_cols = ["season", "player", "player_id", "pos", "gms", "clwpa"]
    raw_xlsx = ROOT / "data" / "raw" / "wpa_all_seasons.xlsx"
    with pd.ExcelWriter(raw_xlsx, engine="openpyxl") as w:
        for variant, sheet in variants:
            sub = clean[clean["playoff_variant"] == variant][out_cols]
            sub.to_excel(w, sheet_name=f"cWPA_{sheet}", index=False)
    print(f"Rebuilt {raw_xlsx}")

    # Two players share each of these names; player_id keeps them distinct.
    for name in ["Chris Johnson", "Marcus Williams"]:
        sub = clean[(clean["player"] == name) & (clean["playoff_variant"] == "regular")]
        print(f"\n--- Spot-check: {name} ({sub['player_id'].nunique()} distinct ids) ---")
        print(sub[["season", "player_id", "pos", "gms", "clwpa"]].to_string(index=False))


if __name__ == "__main__":
    main()
