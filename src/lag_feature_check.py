"""
Quick check: does last season's cWPA rank in the top 20 features?

Adds clwpa_lag1 (previous year's cWPA for the same player) as a candidate
feature for both the regular-season and playoff models, then runs Step 1
(feature importance on training years 2000-2020) to see where it lands.

No hyperparameter search, no test-set evaluation — this is purely about rank.
Run the full build_xgboost.py rebuild if the feature makes top 20.
"""

from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd

from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import load_cwpa

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
TRAIN_YEARS = list(range(2000, 2021))
TOP_K       = 20
RANDOM_STATE = 42

MODELS = [
    # (bbref_stem, wpa_variant, gms_thr, label, lag_variant)
    ("seasons_stats_clean", "regular",  62, "Regular Season", "regular"),
    ("playoffStats_clean",  "playoffs", 10, "Playoffs",       "playoffs"),
]

EXCLUDE = {"Year", "Player", "Pos", "Age", "Tm", "GS", "player_id", "yr"}


def norm(name):
    if not isinstance(name, str):
        return ""
    return name.replace(".", "").replace("*", "").strip().lower()


def add_lag(merged: pd.DataFrame, wpa: pd.DataFrame,
            lag_variant: str) -> pd.DataFrame:
    """
    Attach clwpa_lag1 = player's cWPA in (lag_variant, Year-1).

    merged already carries player_id from the initial join, so we join
    directly on player_id + (Year-1) against the WPA prior-year rows.
    """
    prior = (wpa[wpa["playoff_variant"] == lag_variant]
             [["player_id", "yr", "clwpa"]]
             .copy()
             .rename(columns={"clwpa": "clwpa_lag1"}))
    # Tag each prior-year row with the *next* season's Year so the join aligns
    prior["Year"] = prior["yr"] + 1

    lag_map = prior[["player_id", "Year", "clwpa_lag1"]]
    merged  = merged.merge(lag_map, on=["player_id", "Year"], how="left")

    n_have  = merged["clwpa_lag1"].notna().sum()
    n_total = len(merged)
    print(f"  clwpa_lag1: {n_have}/{n_total} rows have a prior-year value "
          f"({n_have/n_total:.1%}), {n_total - n_have} NaN "
          f"(rookies / gap years — will be dropped)")
    return merged


def feature_columns(df):
    drop = EXCLUDE | {"clwpa", "gms", "wpa_gms", "player_id",
                      "season", "playoff_variant", "yr", "player"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def run(stem, variant, gms_thr, label, lag_variant):
    print(f"\n{'='*70}")
    print(f"  {label}  (G>={gms_thr}, lag from '{lag_variant}' variant)")
    print(f"{'='*70}")

    bbref  = pd.read_csv(ROOT / "data" / "processed" / f"{stem}.csv")
    wpa    = load_cwpa()

    # --- join ---
    wpa_v = wpa[wpa["playoff_variant"] == variant].copy()
    bb    = bbref.copy()
    bb["G"] = pd.to_numeric(bb["G"], errors="coerce")
    bb = bb[bb["G"] >= gms_thr].copy()
    bb["_name"]    = bb["Player"].map(norm)
    wpa_v["_name"] = wpa_v["player"].map(norm)

    merged = bb.merge(
        wpa_v[["yr", "_name", "clwpa", "gms", "player_id"]].rename(
            columns={"gms": "wpa_gms"}
        ),
        left_on=["Year", "_name"], right_on=["yr", "_name"], how="inner",
    ).drop(columns=["yr", "_name"])

    print(f"  Joined rows: {len(merged):,}")

    # --- add lag feature ---
    print("  Adding clwpa_lag1:")
    merged = add_lag(merged, wpa, lag_variant)

    # --- train split ---
    # Drop rows with no prior-year cWPA (rookies, returning-from-injury players).
    # We want to measure whether the lag feature is useful when it exists, not
    # dilute its signal by filling in a learned NaN default for players who have
    # no prior-year data.
    merged_lag = merged.dropna(subset=["clwpa_lag1"]).reset_index(drop=True)
    n_dropped = len(merged) - len(merged_lag)
    print(f"  Dropped {n_dropped} rows with no prior-year cWPA "
          f"(rookies / injury returns) — {len(merged_lag):,} rows remain")

    train = merged_lag[merged_lag["Year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    feats = feature_columns(train)
    print(f"  Candidate features (including lag): {len(feats)}")
    print(f"  Training rows: {len(train):,}")

    # --- Step 1: feature importance ---
    X, y = train[feats], train["clwpa"]
    model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    model.fit(X, y)

    imp = (pd.DataFrame({"feature": feats,
                         "gain_importance": model.feature_importances_})
           .sort_values("gain_importance", ascending=False)
           .reset_index(drop=True))
    imp.insert(0, "rank", imp.index + 1)

    print(f"\n  Top 25 features (including clwpa_lag1):")
    print(f"  {'Rank':>5}  {'Feature':<30}  {'Gain importance':>16}")
    print(f"  {'-'*5}  {'-'*30}  {'-'*16}")
    for _, row in imp.head(25).iterrows():
        marker = " <-- LAG FEATURE" if row["feature"] == "clwpa_lag1" else ""
        print(f"  {int(row['rank']):>5}  {row['feature']:<30}  "
              f"{row['gain_importance']:>16.6f}{marker}")

    lag_row = imp[imp["feature"] == "clwpa_lag1"]
    if not lag_row.empty:
        lag_rank = int(lag_row.iloc[0]["rank"])
        lag_imp  = float(lag_row.iloc[0]["gain_importance"])
        in_top   = lag_rank <= TOP_K
        print(f"\n  >> clwpa_lag1 rank: {lag_rank} / {len(feats)} "
              f"(gain = {lag_imp:.6f})")
        print(f"  >> Makes top {TOP_K}? {'YES' if in_top else 'NO'}")
    else:
        print("  >> clwpa_lag1 not found in importance table — check feature pipeline")


def main():
    for args in MODELS:
        run(*args)


if __name__ == "__main__":
    main()
