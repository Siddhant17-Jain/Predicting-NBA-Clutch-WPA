"""
Step 3: Correlation Analysis — 3 prongs × {per-game, total} + Spearman + sensitivity sweep.
Reads data/processed/cwpa_long.xlsx and emits:
  figures/analysis/prong_1_regular_yoy/
  figures/analysis/prong_2_reg_to_playoff/
  figures/analysis/prong_3_combined_yoy/
  figures/analysis/sensitivity/
  data/outputs/analysis_results.csv
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

from stats_utils import set_style, load_cwpa, corr_summary, scatter_corr, pearson_with_ci, spearman_rho
from stats_utils import C_REG, C_PO, C_COMB

warnings.filterwarnings("ignore")
set_style()

ROOT  = Path(__file__).resolve().parent.parent
P1    = ROOT / "figures" / "analysis" / "prong_1_regular_yoy"
P2    = ROOT / "figures" / "analysis" / "prong_2_reg_to_playoff"
P3    = ROOT / "figures" / "analysis" / "prong_3_combined_yoy"
SENS  = ROOT / "figures" / "analysis" / "sensitivity"
OUT_D = ROOT / "data" / "outputs"

df   = load_cwpa()
reg  = df[df["playoff_variant"] == "regular"].copy()
po   = df[df["playoff_variant"] == "playoffs"].copy()
comb = df[df["playoff_variant"] == "combined"].copy()

THR_REG  = 62
THR_PO   = 10
THR_COMB = 72

results = []


def add_rate(df_in):
    d = df_in.copy()
    d["rate"] = np.where(d["gms"] > 0, d["clwpa"] / d["gms"], np.nan)
    return d

def build_yoy_pairs(df_in, thr, id_col="player_id"):
    d = df_in[df_in["gms"] >= thr].copy()
    d = d[d[id_col].notna() & (d[id_col] != "")]
    d = add_rate(d)
    pairs = []
    for yr in sorted(d["yr"].unique()):
        y0 = (d[d["yr"] == yr][[id_col, "clwpa", "rate", "gms", "player"]]
              .add_suffix("_n").rename(columns={f"{id_col}_n": id_col}))
        y1 = (d[d["yr"] == yr + 1][[id_col, "clwpa", "rate", "gms", "player"]]
              .add_suffix("_n1").rename(columns={f"{id_col}_n1": id_col}))
        merged = y0.merge(y1, on=id_col)
        merged["yr_n"] = yr
        pairs.append(merged)
    return pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()

def record(prong, norm, r, p, ci_lo, ci_hi, rho, rho_p, n):
    results.append(dict(prong=prong, normalization=norm,
                        pearson_r=round(r, 4), p_pearson=round(p, 8),
                        ci_lo=round(ci_lo, 4), ci_hi=round(ci_hi, 4),
                        spearman_rho=round(rho, 4), p_spearman=round(rho_p, 8),
                        n=n))


# ══════════════════════════════════════════════════════════════════════════════
# PRONG 1 — Regular-season YoY persistence
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"PRONG 1: Regular-season YoY persistence  (gms ≥ {THR_REG} both years)")
print("="*60)

p1 = build_yoy_pairs(reg, THR_REG)
print(f"  Total paired player-seasons: {len(p1)}")

for norm, xcol, ycol, xlabel, ylabel in [
    ("per_game", "rate_n",  "rate_n1",  "cWPA/game  Year N",  "cWPA/game  Year N+1"),
    ("total",    "clwpa_n", "clwpa_n1", "cWPA total  Year N", "cWPA total  Year N+1"),
]:
    x, y = p1[xcol].values, p1[ycol].values
    corr_summary(x, y, label=f"P1/{norm}")
    r, p_val, ci_lo, ci_hi, n = pearson_with_ci(x, y)
    rho, rho_p, _ = spearman_rho(x, y)
    record("P1_reg_yoy", norm, r, p_val, ci_lo, ci_hi, rho, rho_p, n)
    scatter_corr(x, y, xlabel, ylabel,
                 f"Prong 1 — Regular YoY Persistence ({norm.replace('_', ' ')})\n"
                 f"gms ≥ {THR_REG} both seasons  |  n={n:,}",
                 P1 / f"p1_regular_yoy_{norm}.png", color=C_REG)

# Per-transition r distribution
print("\n  Per-transition Pearson r (Prong 1, per-game):")
trans_rs = []
for yr in sorted(p1["yr_n"].unique()):
    sub = p1[p1["yr_n"] == yr]
    if len(sub) < 10:
        continue
    r_t, _, _, _, n_t = pearson_with_ci(sub["rate_n"].values, sub["rate_n1"].values)
    trans_rs.append(dict(yr_n=yr, r=r_t, n=n_t))
    print(f"    {yr}→{yr+1}  r={r_t:.3f}  n={n_t}")

trans_df = pd.DataFrame(trans_rs)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(trans_df["yr_n"].astype(str), trans_df["r"], color=C_REG, edgecolor="white")
ax.axhline(trans_df["r"].mean(), color="black", lw=1.5, ls="--",
           label=f"Mean r = {trans_df['r'].mean():.3f}")
ax.set_xlabel("Year N"); ax.set_ylabel("Pearson r (yr N → yr N+1)")
ax.set_title("Prong 1: Per-transition YoY r — Regular (per-game rate)")
ax.set_xticklabels(trans_df["yr_n"].astype(str), rotation=45, ha="right", fontsize=7)
ax.legend()
fig.tight_layout()
fig.savefig(P1 / "p1_per_transition_r.png"); plt.close(fig)
print(f"  Saved prong_1_regular_yoy/p1_per_transition_r.png")


# ══════════════════════════════════════════════════════════════════════════════
# PRONG 2 — Regular → Playoff transfer (same season)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"PRONG 2: Regular → Playoff transfer  (reg gms≥{THR_REG}, po gms≥{THR_PO})")
print("="*60)

r2 = add_rate(reg[reg["gms"] >= THR_REG].copy())
p2 = add_rate(po[po["gms"]   >= THR_PO].copy())
r2 = r2[r2["player_id"].notna() & (r2["player_id"] != "")]
p2 = p2[p2["player_id"].notna() & (p2["player_id"] != "")]

merged2 = r2.merge(p2[["season", "player_id", "clwpa", "rate", "gms"]],
                   on=["season", "player_id"], suffixes=("_reg", "_po"))
print(f"  Paired player-seasons: {len(merged2)}")

for norm, xcol, ycol, xlabel, ylabel in [
    ("per_game", "rate_reg",  "rate_po",  "Regular cWPA/game",  "Playoff cWPA/game"),
    ("total",    "clwpa_reg", "clwpa_po", "Regular cWPA total", "Playoff cWPA total"),
]:
    x, y = merged2[xcol].values, merged2[ycol].values
    corr_summary(x, y, label=f"P2/{norm}")
    r, p_val, ci_lo, ci_hi, n = pearson_with_ci(x, y)
    rho, rho_p, _ = spearman_rho(x, y)
    record("P2_reg_to_po", norm, r, p_val, ci_lo, ci_hi, rho, rho_p, n)
    scatter_corr(x, y, xlabel, ylabel,
                 f"Prong 2 — Regular → Playoff Transfer ({norm.replace('_', ' ')})\n"
                 f"reg gms≥{THR_REG}, po gms≥{THR_PO}  |  n={n:,}",
                 P2 / f"p2_reg_to_playoff_{norm}.png", color=C_PO)


# ══════════════════════════════════════════════════════════════════════════════
# PRONG 3 — Combined YoY persistence
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"PRONG 3: Combined YoY persistence  (gms ≥ {THR_COMB} both years)")
print("="*60)

p3 = build_yoy_pairs(comb, THR_COMB)
print(f"  Total paired player-seasons: {len(p3)}")

for norm, xcol, ycol, xlabel, ylabel in [
    ("per_game", "rate_n",  "rate_n1",  "Combined cWPA/game  Year N",  "Combined cWPA/game  Year N+1"),
    ("total",    "clwpa_n", "clwpa_n1", "Combined cWPA total  Year N", "Combined cWPA total  Year N+1"),
]:
    x, y = p3[xcol].values, p3[ycol].values
    corr_summary(x, y, label=f"P3/{norm}")
    r, p_val, ci_lo, ci_hi, n = pearson_with_ci(x, y)
    rho, rho_p, _ = spearman_rho(x, y)
    record("P3_comb_yoy", norm, r, p_val, ci_lo, ci_hi, rho, rho_p, n)
    scatter_corr(x, y, xlabel, ylabel,
                 f"Prong 3 — Combined YoY Persistence ({norm.replace('_', ' ')})\n"
                 f"gms ≥ {THR_COMB} both seasons  |  n={n:,}",
                 P3 / f"p3_combined_yoy_{norm}.png", color=C_COMB)

p1_pg_r = next(r["pearson_r"] for r in results if r["prong"]=="P1_reg_yoy"  and r["normalization"]=="per_game")
p3_pg_r = next(r["pearson_r"] for r in results if r["prong"]=="P3_comb_yoy" and r["normalization"]=="per_game")
print(f"\n  P1 vs P3 (per-game):  P1 r={p1_pg_r}  P3 r={p3_pg_r}  Δr={p3_pg_r-p1_pg_r:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY SWEEP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SENSITIVITY SWEEP")
print("="*60)

reg_thrs  = [41, 52, 62, 72, 82]
po_thrs   = [5, 8, 10, 13, 16]
comb_thrs = [41, 52, 62, 72, 82]

print("\n  Prong 1 sweep (regular YoY, per-game):")
sweep_p1 = []
for thr in reg_thrs:
    pairs = build_yoy_pairs(reg, thr)
    if len(pairs) < 30:
        sweep_p1.append((thr, np.nan, np.nan, 0)); continue
    r_pg,  _, _, _, n_pg = pearson_with_ci(pairs["rate_n"].values,  pairs["rate_n1"].values)
    r_tot, _, _, _, _    = pearson_with_ci(pairs["clwpa_n"].values, pairs["clwpa_n1"].values)
    sweep_p1.append((thr, r_pg, r_tot, n_pg))
    print(f"    gms≥{thr}: n={n_pg}  r(per-game)={r_pg:.4f}  r(total)={r_tot:.4f}")

print("\n  Prong 3 sweep (combined YoY, per-game):")
sweep_p3 = []
for thr in comb_thrs:
    pairs = build_yoy_pairs(comb, thr)
    if len(pairs) < 30:
        sweep_p3.append((thr, np.nan, np.nan, 0)); continue
    r_pg,  _, _, _, n_pg = pearson_with_ci(pairs["rate_n"].values,  pairs["rate_n1"].values)
    r_tot, _, _, _, _    = pearson_with_ci(pairs["clwpa_n"].values, pairs["clwpa_n1"].values)
    sweep_p3.append((thr, r_pg, r_tot, n_pg))
    print(f"    gms≥{thr}: n={n_pg}  r(per-game)={r_pg:.4f}  r(total)={r_tot:.4f}")

print("\n  Prong 2 sweep (reg × po thresholds, per-game):")
sweep_p2   = np.full((len(reg_thrs), len(po_thrs)), np.nan)
sweep_p2_n = np.zeros((len(reg_thrs), len(po_thrs)), dtype=int)
for i, rthr in enumerate(reg_thrs):
    for j, pthr in enumerate(po_thrs):
        rx = add_rate(reg[reg["gms"] >= rthr].copy())
        px = add_rate(po[po["gms"]   >= pthr].copy())
        rx = rx[rx["player_id"].notna() & (rx["player_id"] != "")]
        px = px[px["player_id"].notna() & (px["player_id"] != "")]
        m  = rx.merge(px[["season","player_id","rate"]], on=["season","player_id"],
                      suffixes=("", "_po"))
        if len(m) < 30:
            continue
        r_v, _, _, _, n_v = pearson_with_ci(m["rate"].values, m["rate_po"].values)
        sweep_p2[i, j]   = round(r_v, 4)
        sweep_p2_n[i, j] = n_v
        print(f"    reg≥{rthr}, po≥{pthr}: n={n_v}  r={r_v:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
thrs1   = [t for t, *_ in sweep_p1]
rs_pg1  = [r for _, r, *_ in sweep_p1]
rs_tot1 = [r for _, _, r, _ in sweep_p1]
ax.plot(thrs1, rs_pg1,  marker="o",          label="Per-game", color=C_REG)
ax.plot(thrs1, rs_tot1, marker="s", ls="--", label="Total",    color=C_REG, alpha=0.6)
ax.axvline(THR_REG, color="gray", ls=":", lw=1)
ax.set_xlabel("Min games"); ax.set_ylabel("Pearson r"); ax.set_ylim(0, 0.7)
ax.set_title("P1: Regular YoY — sensitivity"); ax.legend(fontsize=8)

ax = axes[1]
thrs3   = [t for t, *_ in sweep_p3]
rs_pg3  = [r for _, r, *_ in sweep_p3]
rs_tot3 = [r for _, _, r, _ in sweep_p3]
ax.plot(thrs3, rs_pg3,  marker="o",          label="Per-game", color=C_COMB)
ax.plot(thrs3, rs_tot3, marker="s", ls="--", label="Total",    color=C_COMB, alpha=0.6)
ax.axvline(THR_COMB, color="gray", ls=":", lw=1)
ax.set_xlabel("Min games"); ax.set_ylabel("Pearson r"); ax.set_ylim(0, 0.7)
ax.set_title("P3: Combined YoY — sensitivity"); ax.legend(fontsize=8)

ax = axes[2]
sns.heatmap(sweep_p2, ax=ax, annot=True, fmt=".3f",
            xticklabels=po_thrs, yticklabels=reg_thrs,
            cmap="YlOrRd", vmin=0, vmax=0.4, linewidths=0.5)
ax.set_xlabel("Min playoff games"); ax.set_ylabel("Min regular games")
ax.set_title("P2: Reg→Playoff r — sensitivity\n(per-game Pearson r)")

fig.suptitle("Sensitivity sweep across game-count thresholds", fontsize=12)
fig.tight_layout()
fig.savefig(SENS / "sensitivity_sweep.png"); plt.close(fig)
print(f"\n  Saved sensitivity/sensitivity_sweep.png")

res_path = OUT_D / "analysis_results.csv"
pd.DataFrame(results).to_csv(res_path, index=False)
print(f"\nSaved {res_path}  ({len(results)} rows)")
print(pd.DataFrame(results).to_string())
print("\nAnalysis complete.")
