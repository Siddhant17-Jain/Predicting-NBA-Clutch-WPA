"""
Step 2: Exploratory Data Analysis
Reads data/processed/cwpa_long.xlsx (full unfiltered dataset) and emits:
  figures/eda/coverage/
  figures/eda/distributions/
  figures/eda/careers/
  data/outputs/eda_summary.json
"""

import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from stats_utils import set_style, load_cwpa, C_REG, C_PO, C_COMB

warnings.filterwarnings("ignore")
set_style()

ROOT  = Path(__file__).resolve().parent.parent
EDA   = ROOT / "figures" / "eda"
COV   = EDA / "coverage"
DIST  = EDA / "distributions"
CAR   = EDA / "careers"
OUT_D = ROOT / "data" / "outputs"

df = load_cwpa()
reg  = df[df["playoff_variant"] == "regular"].copy()
po   = df[df["playoff_variant"] == "playoffs"].copy()
comb = df[df["playoff_variant"] == "combined"].copy()

seasons_sorted = sorted(df["season"].unique(), key=lambda s: int(s[:4]))

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

summary = {}
for name, sub in [("regular", reg), ("playoffs", po), ("combined", comb)]:
    n_rows     = len(sub)
    n_players  = sub["player_id"].nunique()
    n_seasons  = sub["season"].nunique()
    n_zero_gms = (sub["gms"] == 0).sum()
    n_gt82     = (sub["gms"] > 82).sum()
    print(f"\n[{name}]  rows={n_rows}  unique players={n_players}  seasons={n_seasons}")
    print(f"  gms=0: {n_zero_gms}   gms>82: {n_gt82}")
    summary[name] = dict(rows=n_rows, unique_players=n_players, seasons=n_seasons,
                         zero_gms=int(n_zero_gms), gt82_gms=int(n_gt82))


# ── 1. Coverage grid ──────────────────────────────────────────────────────────
print("\n[EDA 1] Coverage grid")
fig, ax = plt.subplots(figsize=(12, 4))
for sub, label, color in [(reg, "Regular", C_REG), (po, "Playoffs", C_PO), (comb, "Combined", C_COMB)]:
    cnt = sub.groupby("season")["player_id"].nunique().reindex(seasons_sorted)
    ax.plot(seasons_sorted, cnt.values, marker="o", ms=4, label=label, color=color)
ax.set_xlabel("Season"); ax.set_ylabel("Unique players")
ax.set_title("Coverage: unique players per season × variant")
ax.set_xticks(range(len(seasons_sorted)))
ax.set_xticklabels(seasons_sorted, rotation=45, ha="right", fontsize=7)
ax.legend()
fig.tight_layout()
fig.savefig(COV / "01_coverage_grid.png"); plt.close(fig)
print(f"  Saved {COV}/01_coverage_grid.png")

# ── 2. gms distributions ──────────────────────────────────────────────────────
print("[EDA 2] gms distributions")
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, sub, label, color in zip(axes, [reg, po, comb],
                                  ["Regular", "Playoffs", "Combined"],
                                  [C_REG, C_PO, C_COMB]):
    ax.hist(sub["gms"], bins=40, color=color, edgecolor="white", linewidth=0.3)
    ax.set_title(label); ax.set_xlabel("Games played")
    ax.axvline(sub["gms"].median(), color="black", ls="--", lw=1,
               label=f"Median={sub['gms'].median():.0f}")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Count")
fig.suptitle("Distribution of games played (gms) per variant", y=1.02)
fig.tight_layout()
fig.savefig(DIST / "02_gms_distributions.png"); plt.close(fig)

# ── 3. Position breakdown ─────────────────────────────────────────────────────
print("[EDA 3] Position breakdown")
pos_cnt = reg["pos"].value_counts()
fig, ax = plt.subplots(figsize=(8, 4))
pos_cnt.plot.bar(ax=ax, color=C_REG, edgecolor="white")
ax.set_xlabel("Position"); ax.set_ylabel("Player-seasons")
ax.set_title("Position breakdown (regular season, all years)")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(DIST / "03_position_breakdown.png"); plt.close(fig)

# ── 4. Overall clwpa histogram (regular) ─────────────────────────────────────
print("[EDA 4] clwpa histogram")
r_clean = reg["clwpa"].dropna()
sk  = sp_stats.skew(r_clean)
ku  = sp_stats.kurtosis(r_clean)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(r_clean, bins=80, color=C_REG, edgecolor="white", linewidth=0.3)
ax.axvline(0, color="black", lw=1, ls="--")
txt = (f"n={len(r_clean):,}  mean={r_clean.mean():.3f}  sd={r_clean.std():.3f}\n"
       f"skew={sk:.2f}  excess kurt={ku:.2f}")
ax.text(0.97, 0.96, txt, transform=ax.transAxes, va="top", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax.set_xlabel("cWPA (clutch Win Probability Added)")
ax.set_ylabel("Player-seasons")
ax.set_title("Distribution of cWPA — regular season (all years, all players)")
fig.tight_layout()
fig.savefig(DIST / "04_clwpa_histogram.png"); plt.close(fig)

# ── 5. Per-game rate histogram ────────────────────────────────────────────────
print("[EDA 5] per-game rate histogram")
reg_pg        = reg[reg["gms"] > 0].copy()
reg_pg["rate"] = reg_pg["clwpa"] / reg_pg["gms"]
sk2 = sp_stats.skew(reg_pg["rate"])
ku2 = sp_stats.kurtosis(reg_pg["rate"])
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(reg_pg["rate"], bins=80, color=C_REG, edgecolor="white", linewidth=0.3)
ax.axvline(0, color="black", lw=1, ls="--")
txt2 = (f"n={len(reg_pg):,}  mean={reg_pg['rate'].mean():.4f}  sd={reg_pg['rate'].std():.4f}\n"
        f"skew={sk2:.2f}  excess kurt={ku2:.2f}")
ax.text(0.97, 0.96, txt2, transform=ax.transAxes, va="top", ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax.set_xlabel("cWPA per game")
ax.set_ylabel("Player-seasons")
ax.set_title("Distribution of cWPA per game — regular season")
fig.tight_layout()
fig.savefig(DIST / "05_clwpa_per_game_histogram.png"); plt.close(fig)

# ── 6. Per-season violin (regular) ───────────────────────────────────────────
print("[EDA 6] per-season violin")
fig, ax = plt.subplots(figsize=(16, 5))
data_by_season = [reg[reg["season"] == s]["clwpa"].dropna().values for s in seasons_sorted]
parts = ax.violinplot(data_by_season, positions=range(len(seasons_sorted)),
                      showmedians=True, showextrema=False)
for pc in parts["bodies"]:
    pc.set_facecolor(C_REG); pc.set_alpha(0.6)
ax.set_xticks(range(len(seasons_sorted)))
ax.set_xticklabels(seasons_sorted, rotation=45, ha="right", fontsize=7)
ax.set_xlabel("Season"); ax.set_ylabel("cWPA")
ax.set_title("cWPA distribution per season — regular (violin, full unfiltered)")
ax.axhline(0, color="black", lw=0.8, ls="--")
fig.tight_layout()
fig.savefig(DIST / "06_seasonal_violin_regular.png"); plt.close(fig)

# ── 7. League-level yearly aggregates ────────────────────────────────────────
print("[EDA 7] league yearly aggregates")
agg = reg.groupby("season")["clwpa"].agg(
    mean_clwpa="mean", median_clwpa="median", total_clwpa="sum",
    strong_clutch=lambda x: (x >= 1).sum()
).reindex(seasons_sorted)

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, col, label in zip(
    axes.flat,
    ["mean_clwpa", "median_clwpa", "total_clwpa", "strong_clutch"],
    ["Mean cWPA", "Median cWPA", "Total cWPA (all players)", "Player-seasons with cWPA ≥ 1"],
):
    ax.plot(seasons_sorted, agg[col].values, marker="o", ms=4, color=C_REG)
    ax.set_xticks(range(len(seasons_sorted)))
    ax.set_xticklabels(seasons_sorted, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(label); ax.set_title(label)
    ax.axhline(agg[col].mean(), color="gray", lw=0.8, ls="--")
fig.suptitle("League-level cWPA trends over time — regular season", fontsize=12)
fig.tight_layout()
fig.savefig(DIST / "07_league_yearly_aggregates.png"); plt.close(fig)

# ── 8. Career totals — top 25 ────────────────────────────────────────────────
print("[EDA 8] career totals")
for sub, label, color, tag in [
    (reg,  "Regular",  C_REG,  "regular"),
    (po,   "Playoffs", C_PO,   "playoffs"),
    (comb, "Combined", C_COMB, "combined"),
]:
    car = (sub.groupby(["player_id", "player"])["clwpa"]
             .sum().reset_index()
             .sort_values("clwpa", ascending=False)
             .head(25))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(car["player"][::-1], car["clwpa"][::-1], color=color)
    ax.set_xlabel("Career cWPA")
    ax.set_title(f"Top 25 career cWPA — {label}")
    ax.axvline(0, color="black", lw=0.6)
    fig.tight_layout()
    fig.savefig(CAR / f"08_career_top25_{tag}.png"); plt.close(fig)

# ── 9. Career longevity ───────────────────────────────────────────────────────
print("[EDA 9] career longevity")
car_reg = (reg.groupby("player_id")
             .agg(seasons=("season", "nunique"), total_clwpa=("clwpa", "sum"))
             .reset_index())
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(car_reg["seasons"], bins=range(1, 28), color=C_REG, edgecolor="white")
axes[0].set_xlabel("Seasons in dataset"); axes[0].set_ylabel("# Players")
axes[0].set_title("Career longevity — regular (seasons per player)")
axes[1].scatter(car_reg["seasons"], car_reg["total_clwpa"],
                alpha=0.35, s=14, color=C_REG, linewidths=0)
name_map = reg.drop_duplicates("player_id").set_index("player_id")["player"]
for _, row in car_reg.nlargest(8, "total_clwpa").iterrows():
    name = name_map.get(row["player_id"], row["player_id"])
    axes[1].annotate(name, (row["seasons"], row["total_clwpa"]), fontsize=6,
                     xytext=(3, 2), textcoords="offset points")
axes[1].set_xlabel("Seasons"); axes[1].set_ylabel("Career total cWPA")
axes[1].set_title("Career total cWPA vs seasons played")
fig.tight_layout()
fig.savefig(CAR / "09_career_longevity.png"); plt.close(fig)

# ── 10. Leaderboards ─────────────────────────────────────────────────────────
print("[EDA 10] leaderboards")
best  = reg.nlargest(20, "clwpa")[["season", "player", "pos", "gms", "clwpa"]]
worst = reg.nsmallest(20, "clwpa")[["season", "player", "pos", "gms", "clwpa"]]
print("\nTop 10 single-season regular cWPA:")
print(best.head(10).to_string(index=False))
print("\nBottom 10 single-season regular cWPA:")
print(worst.head(10).to_string(index=False))
summary["top10_seasons"]    = best.head(10).to_dict("records")
summary["bottom10_seasons"] = worst.head(10).to_dict("records")

# ── Save summary JSON ─────────────────────────────────────────────────────────
summary_path = OUT_D / "eda_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nEDA complete.")
print(f"  Figures → figures/eda/{{coverage,distributions,careers}}/")
print(f"  Summary → {summary_path}")
