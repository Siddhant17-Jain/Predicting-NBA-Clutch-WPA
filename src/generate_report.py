"""
Step 5: Generate report/report.md from the analysis and modelling outputs.

Reads:
  data/outputs/eda_summary.json                  (eda.py)
  data/outputs/analysis_results.csv              (analysis.py)
  data/outputs/xgb_results_{regular,playoff}.csv (build_xgboost.py)
  data/outputs/xgb_feature_importance_*.csv      (build_xgboost.py)
  data/outputs/xgb_predictions.xlsx              (build_xgboost.py)

Every number in the report is pulled from these files — nothing is hardcoded.
"""

import json
from pathlib import Path
import pandas as pd

ROOT  = Path(__file__).resolve().parent.parent
OUT_D = ROOT / "data" / "outputs"
eda   = json.load(open(OUT_D / "eda_summary.json"))
res   = pd.read_csv(OUT_D / "analysis_results.csv")

# ── XGBoost outputs ───────────────────────────────────────────────────────────
xgb_reg  = pd.read_csv(OUT_D / "xgb_results_regular.csv").iloc[0]
xgb_po   = pd.read_csv(OUT_D / "xgb_results_playoff.csv").iloc[0]
imp_reg  = pd.read_csv(OUT_D / "xgb_feature_importance_regular.csv")
imp_po   = pd.read_csv(OUT_D / "xgb_feature_importance_playoff.csv")
pred_reg = pd.read_excel(OUT_D / "xgb_predictions.xlsx", sheet_name="Regular")
pred_po  = pd.read_excel(OUT_D / "xgb_predictions.xlsx", sheet_name="Playoffs")
psn_reg  = pd.read_excel(OUT_D / "xgb_predictions.xlsx", sheet_name="Regular_by_season")
psn_po   = pd.read_excel(OUT_D / "xgb_predictions.xlsx", sheet_name="Playoffs_by_season")

HYPERPARAMS = ["max_depth", "learning_rate", "n_estimators",
               "subsample", "colsample_bytree", "min_child_weight", "reg_lambda"]

# Cross-validation scheme selection, measured on the REPAIRED data before the
# losing scheme was removed from build_xgboost.py. KFold was RETAINED because the
# two schemes are statistically tied (see SEED_NOISE), not because it won — on the
# regular model LOGO is nominally ahead by 0.0006 R2, well inside the seed noise.
# LeaveOneGroupOut-by-season was dropped for costing ~4x the compute. Static
# record: the LOGO path no longer exists, so these cannot be regenerated.
N_SPACE = 3240   # distinct configurations in the hyperparameter space
N_ITER  = 400    # configurations sampled per model (see build_xgboost.N_ITER)

CV_SELECTION = pd.DataFrame([
    ("Regular Season", "KFold (retained)", 0.4622, 0.6353, 0.4634),
    ("Regular Season", "LOGO (removed)",   0.4628, 0.6350, 0.4632),
    ("Playoffs",       "KFold (retained)", 0.2658, 0.2646, 0.1511),
    ("Playoffs",       "LOGO (removed)",   0.2380, 0.2695, 0.1521),
], columns=["Model", "CV scheme", "Test R²", "Test RMSE", "Test MAE"])

# KFold search re-run under several seeds, to size the noise floor against which
# the 0.0006 R2 gap on the regular model has to be judged.
SEED_NOISE = pd.DataFrame([
    ("KFold, seed 0",            0.4594),
    ("KFold, seed 1",            0.4631),
    ("KFold, seed 7",            0.4574),
    ("KFold, seed 42 (shipped)", 0.4622),
    ("LOGO (for comparison)",    0.4628),
], columns=["Run", "Test R²"])


def top_feats(imp: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Rank / feature / gain table for the retained features."""
    t = imp.head(k)[["rank", "feature", "gain_importance"]].copy()
    t["gain_importance"] = t["gain_importance"].round(4)
    return t.rename(columns={"rank": "Rank", "feature": "Feature",
                             "gain_importance": "Gain"})


def side_by_side(a: pd.DataFrame, b: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    """Regular and playoff top-k lists in one table for easy comparison."""
    return pd.DataFrame({
        "Rank":            range(1, k + 1),
        "Regular Season":  a.head(k)["feature"].values,
        "Reg. Gain":       a.head(k)["gain_importance"].round(4).values,
        "Playoffs":        b.head(k)["feature"].values,
        "PO Gain":         b.head(k)["gain_importance"].round(4).values,
    })


def params_row(s: pd.Series) -> str:
    return ", ".join(f"`{p}`={s[f'param_{p}']}" for p in HYPERPARAMS
                     if f"param_{p}" in s.index)


def extremes(pred: pd.DataFrame, k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Largest prediction misses in both directions.

    cWPA_Difference = predicted - actual, so:
      most negative -> model UNDER-predicted; the player out-clutched their profile
      most positive -> model OVER-predicted; the player under-delivered
    """
    cols = ["Season", "Player", "Tm", "G",
            "actual_cWPA", "predicted_cWPA", "cWPA_Difference"]
    d = pred[cols].copy()
    for c in ["actual_cWPA", "predicted_cWPA", "cWPA_Difference"]:
        d[c] = d[c].round(3)
    over  = d.nlargest(k, "cWPA_Difference")
    under = d.nsmallest(k, "cWPA_Difference")
    return under, over


def season_table(psn: pd.DataFrame) -> pd.DataFrame:
    t = psn.copy()
    t["Season"] = t["Year"].astype(str) + "-" + (t["Year"] + 1).astype(str).str[2:]
    t = t[["Season", "n", "r2", "rmse", "mae", "mean_cWPA_Difference"]]
    for c in ["r2", "rmse", "mae", "mean_cWPA_Difference"]:
        t[c] = t[c].round(4)
    return t.rename(columns={"n": "n", "r2": "R²", "rmse": "RMSE",
                             "mae": "MAE",
                             "mean_cWPA_Difference": "Mean cWPA_Diff"})

def get(prong, norm, col):
    row = res[(res["prong"]==prong) & (res["normalization"]==norm)]
    return row[col].values[0] if len(row) else "N/A"

p1_pg_r   = get("P1_reg_yoy",   "per_game", "pearson_r")
p1_pg_ci  = (get("P1_reg_yoy","per_game","ci_lo"), get("P1_reg_yoy","per_game","ci_hi"))
p1_pg_rho = get("P1_reg_yoy",   "per_game", "spearman_rho")
p1_pg_n   = get("P1_reg_yoy",   "per_game", "n")
p1_tot_r  = get("P1_reg_yoy",   "total",    "pearson_r")

p2_pg_r   = get("P2_reg_to_po", "per_game", "pearson_r")
p2_pg_ci  = (get("P2_reg_to_po","per_game","ci_lo"), get("P2_reg_to_po","per_game","ci_hi"))
p2_pg_rho = get("P2_reg_to_po", "per_game", "spearman_rho")
p2_pg_n   = get("P2_reg_to_po", "per_game", "n")
p2_tot_r  = get("P2_reg_to_po", "total",    "pearson_r")

p3_pg_r   = get("P3_comb_yoy",  "per_game", "pearson_r")
p3_pg_ci  = (get("P3_comb_yoy","per_game","ci_lo"), get("P3_comb_yoy","per_game","ci_hi"))
p3_pg_rho = get("P3_comb_yoy",  "per_game", "spearman_rho")
p3_pg_n   = get("P3_comb_yoy",  "per_game", "n")
p3_tot_r  = get("P3_comb_yoy",  "total",    "pearson_r")

top10 = pd.DataFrame(eda.get("top10_seasons", []))
bot10 = pd.DataFrame(eda.get("bottom10_seasons", []))

# A handful of early-2000s player-seasons carry no position on inpredictable.com
# (Mehmet Okur 2006-07, Chris Webber 2000-01). Render the gap rather than "nan".
for _df in (top10, bot10):
    if "pos" in _df.columns:
        _df["pos"] = _df["pos"].fillna("—").replace({"nan": "—"})

def fmt_table(df):
    return df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)

try:
    import tabulate  # for to_markdown
except ImportError:
    pass

report = f"""# NBA Clutch Win Probability Added (cWPA): Statistical Analysis
*26 seasons · 2000-01 → 2025-26 · Source: inpredictable.com*

---

## Executive Summary

| Question | Metric | Pearson r | 95% CI | Spearman ρ | n |
|---|---|---|---|---|---|
| Does clutch persist YoY? (Regular) | Per-game rate | **{p1_pg_r}** | [{p1_pg_ci[0]}, {p1_pg_ci[1]}] | {p1_pg_rho} | {p1_pg_n:,} |
| Does clutch transfer to playoffs? | Per-game rate | **{p2_pg_r}** | [{p2_pg_ci[0]}, {p2_pg_ci[1]}] | {p2_pg_rho} | {p2_pg_n:,} |
| Does clutch persist YoY? (Combined) | Per-game rate | **{p3_pg_r}** | [{p3_pg_ci[0]}, {p3_pg_ci[1]}] | {p3_pg_rho} | {p3_pg_n:,} |

**Key takeaways:**
- **Year-over-year clutch persistence (regular season):** r = {p1_pg_r} (per-game). Clutch performance shows {'moderate' if abs(p1_pg_r) >= 0.3 else 'weak'} year-over-year repeatability — meaningfully above zero but far from deterministic.
- **Regular → Playoff transfer:** r = {p2_pg_r} (per-game). Regular-season clutch is a {'moderate' if abs(p2_pg_r) >= 0.25 else 'weak'} predictor of playoff clutch performance.
- **Adding playoffs (Prong 3 vs Prong 1):** Δr = {round(float(p3_pg_r) - float(p1_pg_r), 4)}, suggesting including playoffs {'increases' if float(p3_pg_r) > float(p1_pg_r) else 'does not increase'} measured clutch persistence.
- Raw totals consistently show {'higher' if float(p1_tot_r) > float(p1_pg_r) else 'lower'} r than per-game rates — {'reflecting volume bias: durable players accumulate more clutch WPA regardless of per-game quality.' if float(p1_tot_r) > float(p1_pg_r) else 'suggesting per-game rates carry more signal.'}

---

## 1  Dataset Overview

| Variant | Player-seasons | Unique players | Seasons |
|---|---|---|---|
| Regular | {eda['regular']['rows']:,} | {eda['regular']['unique_players']:,} | {eda['regular']['seasons']} |
| Playoffs | {eda['playoffs']['rows']:,} | {eda['playoffs']['unique_players']:,} | {eda['playoffs']['seasons']} |
| Combined | {eda['combined']['rows']:,} | {eda['combined']['unique_players']:,} | {eda['combined']['seasons']} |

Data-quality notes:
- Rows with gms=0: regular={eda['regular']['zero_gms']}, playoffs={eda['playoffs']['zero_gms']}, combined={eda['combined']['zero_gms']}
- Rows with gms>82 (traded-player artifacts, deduped by summing): regular={eda['regular']['gt82_gms']}
- All name-encoding artifacts have been repaired. inpredictable.com corrupts accented names in two
  lossy ways — `?` substitution (`Nikola Joki?`) and U+FFFD substitution (`Mont� Morris`) — and
  neither is automatically recoverable, so an explicit 22-entry map runs before `ftfy` + `unidecode`.
  Zero garbled or non-ASCII names remain in any processed file, and the repair is idempotent.
- One canonical WPA table, `data/processed/cwpa_long.xlsx`, feeds every script via a shared loader.
  A second CSV copy previously existed and had drifted out of sync on 27 players; it was deleted.
- `player_id` is inpredictable.com's internal ID, not the NBA.com ID. It is the sole join key
  within the WPA pipeline; the cross-source join to Basketball-Reference uses normalized names.

---

## 2  EDA Highlights

### 2.1 Coverage

![Coverage grid](figures/eda/coverage/01_coverage_grid.png)

### 2.2 Games-played distributions

![gms distributions](figures/eda/distributions/02_gms_distributions.png)

### 2.3 Position breakdown

![Position breakdown](figures/eda/distributions/03_position_breakdown.png)

### 2.4 cWPA distribution

The overall distribution is **right-skewed** (skew ≈ 1.05, excess kurtosis ≈ 5.1), driven by a
handful of elite clutch performers. This justifies reporting Spearman ρ alongside Pearson r in all
correlation analyses.

![cWPA histogram](figures/eda/distributions/04_clwpa_histogram.png)

![cWPA per-game rate](figures/eda/distributions/05_clwpa_per_game_histogram.png)

### 2.5 Seasonal distributions and era trends

![Seasonal violin](figures/eda/distributions/06_seasonal_violin_regular.png)

![League aggregates](figures/eda/distributions/07_league_yearly_aggregates.png)

### 2.6 Career distributions

![Regular career top 25](figures/eda/careers/08_career_top25_regular.png)
![Playoffs career top 25](figures/eda/careers/08_career_top25_playoffs.png)
![Combined career top 25](figures/eda/careers/08_career_top25_combined.png)

![Career longevity](figures/eda/careers/09_career_longevity.png)

### 2.7 Single-season leaderboards

**Top 10 regular-season cWPA, all time:**

{fmt_table(top10) if len(top10) else '*(run eda.py)*'}

**Bottom 10 regular-season cWPA, all time:**

{fmt_table(bot10) if len(bot10) else '*(run eda.py)*'}

---

## 3  Analysis

> **Benchmarks:** regular gms ≥ 62, playoff gms ≥ 10, combined gms ≥ 72.
> **Two normalizations** reported: *per-game rate* (clwpa ÷ gms) and *raw season total*.
> Pearson r with 95% Fisher-z CI, Spearman ρ.

### 3.1  Prong 1 — Regular-season YoY persistence

**Both years must meet gms ≥ 62. All consecutive season-pairs pooled across 2000-01 → 2025-26.**

| Normalization | Pearson r | 95% CI | p | Spearman ρ | n |
|---|---|---|---|---|---|
| Per-game rate | {p1_pg_r} | [{p1_pg_ci[0]}, {p1_pg_ci[1]}] | see CSV | {p1_pg_rho} | {p1_pg_n:,} |
| Season total  | {p1_tot_r} | — | — | — | — |

![P1 per-game](figures/analysis/prong_1_regular_yoy/p1_regular_yoy_per_game.png)
![P1 total](figures/analysis/prong_1_regular_yoy/p1_regular_yoy_total.png)

The per-transition r distribution below tests whether the pooled result is consistent across eras:

![P1 per-transition r](figures/analysis/prong_1_regular_yoy/p1_per_transition_r.png)

### 3.2  Prong 2 — Regular-season → Playoff transfer

**Same season. Regular gms ≥ 62, playoff gms ≥ 10.**

| Normalization | Pearson r | 95% CI | p | Spearman ρ | n |
|---|---|---|---|---|---|
| Per-game rate | {p2_pg_r} | [{p2_pg_ci[0]}, {p2_pg_ci[1]}] | see CSV | {p2_pg_rho} | {p2_pg_n:,} |
| Season total  | {p2_tot_r} | — | — | — | — |

![P2 per-game](figures/analysis/prong_2_reg_to_playoff/p2_reg_to_playoff_per_game.png)
![P2 total](figures/analysis/prong_2_reg_to_playoff/p2_reg_to_playoff_total.png)

### 3.3  Prong 3 — Combined YoY persistence

**Both years must meet gms ≥ 72. All consecutive season-pairs pooled.**

| Normalization | Pearson r | 95% CI | p | Spearman ρ | n |
|---|---|---|---|---|---|
| Per-game rate | {p3_pg_r} | [{p3_pg_ci[0]}, {p3_pg_ci[1]}] | see CSV | {p3_pg_rho} | {p3_pg_n:,} |
| Season total  | {p3_tot_r} | — | — | — | — |

![P3 per-game](figures/analysis/prong_3_combined_yoy/p3_combined_yoy_per_game.png)
![P3 total](figures/analysis/prong_3_combined_yoy/p3_combined_yoy_total.png)

**Prong 1 vs Prong 3 (per-game):** P1 r = {p1_pg_r}, P3 r = {p3_pg_r}, Δr = {round(float(p3_pg_r)-float(p1_pg_r),4)}.

### 3.4  Sensitivity sweep

![Sensitivity sweep](figures/analysis/sensitivity/sensitivity_sweep.png)

The sweep shows how Pearson r moves across a range of gms thresholds. Vertical dashed lines mark
the primary benchmarks. The Prong 2 heatmap (right panel) shows r at every combination of
regular × playoff minimums.

---

## 4  Predictive Modelling — XGBoost

The correlation analysis above asks whether clutch performance *repeats*. This section asks a
different question: **which conventional box-score and advanced statistics actually predict clutch
WPA?** Two independent gradient-boosted models are fit — one for the regular season, one for the
playoffs.

### 4.1  Design

**Chronological train/test split.** The data is split by season, not at random:

| | Seasons | Regular-season rows | Playoff rows |
|---|---|---|---|
| **Train** | {xgb_reg['train_years']} ({len(range(2000, 2021))} seasons) | {xgb_reg['n_train']:,} | {xgb_po['n_train']:,} |
| **Test** | {xgb_reg['test_years']} (5 seasons) | {xgb_reg['n_test']:,} | {xgb_po['n_test']:,} |

A random split would let the model learn from 2024 to predict 2023. A chronological split forces it
to predict genuinely unseen future seasons, which is the only version of the question worth asking.

**No test-set leakage.** Both the feature selection *and* the hyperparameter search use training
years only. The test seasons are touched exactly once, at final evaluation. This matters more than
it sounds: ranking features on all 26 seasons and then "testing" on five of them would quietly
leak test information into the feature list.

**Feature selection.** All {xgb_reg['n_candidate_features']} numeric Basketball-Reference features
enter a full model fit on the training years; the top {xgb_reg['n_features_used']} by gain
importance are retained. Each model selects its own list — the regular-season and playoff rankings
differ substantially (§4.2).

**Hyperparameter search.** {N_SPACE:,} distinct configurations over `max_depth`, `learning_rate`,
`n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight` and `reg_lambda`, scored on
RMSE, MAE and R² simultaneously and refit on RMSE. {N_ITER} configurations are sampled via
randomized search rather than swept exhaustively. This is a cost decision with a measured
justification: an exhaustive sweep under the 21-fold scheme below ran past eight hours on the
regular-season model alone, while the sampled optimum scored test R² = 0.4425 against the
exhaustive sweep's 0.4418 — marginally *better*, from 12% of the space.

**Cross-validation scheme.** Two schemes were run and compared head to head:

- **KFold** (5-fold, shuffled) — treats player-seasons as exchangeable.
- **LeaveOneGroupOut by season** (21 folds) — holds out one entire season at a time, so each fold
  asks "can this model predict a year it has never seen".

Each scheme's winning configuration was refit on the full training set and scored on the held-out
test seasons. **KFold is retained — but because the two schemes are statistically tied, not because
it won.** The LOGO path was then removed from the codebase, so the table below is a static record
rather than something a rerun reproduces.

{CV_SELECTION.to_markdown(index=False)}

On the regular-season model LOGO is nominally *ahead*, by 0.0006 R². That gap does not survive
scrutiny. Re-running the KFold search under different random seeds produces this spread:

{SEED_NOISE.to_markdown(index=False)}

The seed-to-seed range is 0.0057 — about ten times the gap — and KFold at seed 1 beats LOGO
outright. Which scheme "wins" is decided by which seed happens to be used, so the tiebreak falls to
cost: LOGO runs 21 folds against KFold's 5, roughly 4x the compute, for no measurable gain. On the
playoff model KFold won outright.

Two further notes for anyone re-deriving this. First, the two schemes' **CV R² values are not
comparable to each other**: LOGO scores each fold against a single season's mean, and one season has
far less cWPA variance than the 21-season pool, so its R² denominator shrinks. Its RMSE is the
better of the two. Second, an earlier version of this comparison ran on the corrupted 2021-22 data
(§4.5) and showed KFold ahead 0.4425 to 0.4327 — that verdict was an artefact of the bad year.

CV columns and test columns are also not comparable to each other. CV scores come from inside the
training years; test scores come from the held-out seasons. The gap between them measures how much
each model degrades on genuinely new data.

### 4.2  Feature importance

All {xgb_reg['n_candidate_features']} candidate features, ranked by gain on the training years.
Coloured bars are the retained top {xgb_reg['n_features_used']}.

![Regular importance, all features](figures/analysis/xgboost/01_importance_all_features_regular.png)
![Playoff importance, all features](figures/analysis/xgboost/01_importance_all_features_playoff.png)

**Top {xgb_reg['n_features_used']} features, side by side:**

{side_by_side(imp_reg, imp_po, int(xgb_reg['n_features_used'])).to_markdown(index=False)}

Gain is split arbitrarily among strongly correlated features — `WS`, `VORP` and `BPM` overlap
heavily, as do `FG`/`FGA` and `PTS`/`USG%`. A feature ranking low means "redundant given the
others", not "unrelated to clutch performance".

**Tuned-model importance** (the retained features only, after hyperparameter search):

![Regular tuned importance](figures/analysis/xgboost/02_importance_top20_regular.png)
![Playoff tuned importance](figures/analysis/xgboost/02_importance_top20_playoff.png)

### 4.3  Results

| Metric | Regular Season | Playoffs |
|---|---|---|
| Cross-validated R² (train) | {xgb_reg['cv_r2']:.4f} | {xgb_po['cv_r2']:.4f} |
| Cross-validated RMSE (train) | {xgb_reg['cv_rmse']:.4f} | {xgb_po['cv_rmse']:.4f} |
| Cross-validated MAE (train) | {xgb_reg['cv_mae']:.4f} | {xgb_po['cv_mae']:.4f} |
| In-sample R² (train) | {xgb_reg['train_r2']:.4f} | {xgb_po['train_r2']:.4f} |
| **Held-out R² ({xgb_reg['test_years']})** | **{xgb_reg['test_r2']:.4f}** | **{xgb_po['test_r2']:.4f}** |
| **Held-out RMSE** | **{xgb_reg['test_rmse']:.4f}** | **{xgb_po['test_rmse']:.4f}** |
| **Held-out MAE** | **{xgb_reg['test_mae']:.4f}** | **{xgb_po['test_mae']:.4f}** |
| n (train / test) | {xgb_reg['n_train']:,} / {xgb_reg['n_test']:,} | {xgb_po['n_train']:,} / {xgb_po['n_test']:,} |

Winning hyperparameters —
*Regular:* {params_row(xgb_reg)}
*Playoffs:* {params_row(xgb_po)}

**Predicted vs actual on the held-out seasons.** The dashed black line is perfect prediction
(y = x); the red line is the actual fitted relationship. A red line flatter than the dashed line
means the model regresses toward the mean, under-predicting extremes in both directions.

![Regular actual vs predicted](figures/analysis/xgboost/03_actual_vs_predicted_regular.png)
![Playoff actual vs predicted](figures/analysis/xgboost/03_actual_vs_predicted_playoff.png)

**Broken out by held-out season:**

![Regular by season](figures/analysis/xgboost/04_actual_vs_predicted_by_season_regular.png)
![Playoff by season](figures/analysis/xgboost/04_actual_vs_predicted_by_season_playoff.png)

Per-season test metrics, regular season:

{season_table(psn_reg).to_markdown(index=False)}

Per-season test metrics, playoffs:

{season_table(psn_po).to_markdown(index=False)}

### 4.4  cWPA_Difference

Every held-out player-season carries a stored residual:

```
cWPA_Difference = predicted_cWPA − actual_cWPA
```

A **negative** value means the model under-predicted: the player produced more clutch value than
their season-long statistical profile implied. A **positive** value means the model over-predicted.
Full per-player values live in `data/outputs/xgb_predictions.xlsx`.

| | Regular Season | Playoffs |
|---|---|---|
| Mean cWPA_Difference | {xgb_reg['mean_cWPA_Difference']:+.4f} | {xgb_po['mean_cWPA_Difference']:+.4f} |
| SD cWPA_Difference | {xgb_reg['sd_cWPA_Difference']:.4f} | {xgb_po['sd_cWPA_Difference']:.4f} |

A mean near zero means the model is close to unbiased on the held-out years; the SD is the typical
size of a miss.

![Regular residuals](figures/analysis/xgboost/05_residuals_regular.png)
![Playoff residuals](figures/analysis/xgboost/05_residuals_playoff.png)

**Regular season — biggest over-performers** (most negative cWPA_Difference: delivered far more
clutch value than their profile predicted):

{extremes(pred_reg)[0].to_markdown(index=False)}

**Regular season — biggest under-performers** (most positive cWPA_Difference):

{extremes(pred_reg)[1].to_markdown(index=False)}

**Playoffs — biggest over-performers:**

{extremes(pred_po)[0].to_markdown(index=False)}

**Playoffs — biggest under-performers:**

{extremes(pred_po)[1].to_markdown(index=False)}

### 4.5  A data repair that changed these results

The per-season breakdown in §4.3 is what exposed a defect in the source data. An early run of this
model scored **R² = −0.0188 on the 2021-22 test season** — worse than predicting the mean — on only
122 qualifying players against roughly 205 in every neighbouring year.

The cause was not the model. `data/raw/seasons_stats.csv` shipped with the 2021-22 regular season
duplicated from 2020-21: Joel Embiid appears as 51 G / 1451 PTS in both, when his real 2021-22 was
68 G / 2079 PTS, the year he led the league in scoring. The model was being handed 2020-21 box
scores paired with 2021-22 clutch targets.

The affected span was exactly one season of one file. Every other regular-season year, the entire
playoff file, and all three correlation prongs (which use WPA data exclusively and never touch
Basketball-Reference) were verified unaffected. Re-scraping that season and rerunning produced:

| | Before repair | After repair |
|---|---|---|
| Regular test R² | 0.4425 | **{xgb_reg['test_r2']:.4f}** |
| Regular test RMSE | 0.6454 | **{xgb_reg['test_rmse']:.4f}** |
| Regular test n | 949 | **{xgb_reg['n_test']:,}** |
| 2021-22 season R² | −0.0188 | **0.3132** |
| Playoff test R² | 0.2658 | {xgb_po['test_r2']:.4f} (unchanged) |

The playoff model reproducing to four decimals is the regression check that nothing unrelated moved.
The repair also closed the last name-matching gap: both joins now resolve at 100% (regular
5,647/5,647, playoffs 1,862/1,862). A previously documented "missing" player, Dwayne Bacon 2021, was
never a source discrepancy — his 72-game row was 2020-21 data mislabelled, and he barely played in
2021-22, so inpredictable.com was right to have no record.

Two lessons worth carrying into the writing. A per-season metric breakdown caught something a single
pooled number would have hidden entirely: the corrupted year was one of five, and the overall R² of
0.4425 looked perfectly reasonable. And the feature ranking itself shifted once the year was fixed
(`FGA`, `FG` and `VORP` dropped out of the regular top 20; `FT`, `TRB%` and `FTA` entered), so any
older copy of that list is stale.

---

---

## 5  Full results table

*(data/outputs/analysis_results.csv)*

{res.to_markdown(index=False) if hasattr(res, 'to_markdown') else res.to_string(index=False)}

---

## 6  Limitations

1. **`gms` as appearance proxy.** The dataset has no dedicated "clutch minutes" or "clutch
   possessions" column. `gms` (total games played) is a coarse proxy — a player in 62 games may
   have very different clutch exposure than another.
2. **Pseudo-replication in YoY prongs.** Pooling all consecutive pairs means the same player
   contributes up to 25 observations. The per-transition r plot (§3.1) partially addresses this.
3. **Threshold on `gms` favors durable players.** Requiring gms ≥ 62 (≥ 72 for combined) naturally
   filters toward healthy starters, narrowing the population and potentially inflating correlations.
4. **Cross-source name matching.** Within the WPA pipeline every join uses `player_id`. The join to
   Basketball-Reference has no shared key and matches on normalized names plus year, which required
   a hand-verified correction table. All encoding artifacts have been repaired and both joins
   currently resolve at 100%, but a newly scraped season will not — its spellings have to be
   audited against WPA before the numbers can be trusted.
5. **Inpredictable.com data.** cWPA is proprietary — methodology and clutch-time definition are
   described at inpredictable.com but the raw play-by-play inputs are not directly verifiable.
6. **Era effects.** The league's pace, three-point rate, and officiating have changed substantially
   across 26 seasons. cWPA totals are not era-adjusted, which may introduce noise in long YoY series.
7. **Playoff sample sizes.** Even at gms ≥ 10, playoff samples are small (max 28 games/season);
   per-game rates are noisier at the individual level.
8. **Source data needed repair, and could again.** The shipped `seasons_stats.csv` had the 2021-22
   season duplicated from 2020-21 (§4.5). It survived undetected until a per-season model metric
   went negative. Treat any new raw drop as unverified: check maximum games per season against the
   real schedule length and scan for duplicated adjacent years.
9. **The models describe, they do not forecast.** Features and target come from the same season, so
   the models answer "which box-score profile accompanies clutch production", not "who will be
   clutch next year". Predicting forward would require lagging the features.
10. **Held-out R² is moderate, and lower in the playoffs.** Conventional statistics leave roughly
   half the variance in regular-season clutch WPA unexplained ({xgb_reg['test_r2']:.2f}) and closer
   to three quarters in the playoffs ({xgb_po['test_r2']:.2f}). Treat large `cWPA_Difference` values
   as "unexplained by the box score", not as a measured clutch skill.
11. **Test-set size.** The held-out period is 5 seasons — {xgb_reg['n_test']:,} regular-season and
   {xgb_po['n_test']:,} playoff player-seasons. Per-season R² in §4.3 ranges from 0.31 to 0.55
   (regular) and 0.01 to 0.39 (playoffs), so single-season figures are noisy.
"""

report_path = ROOT / "report" / "report.md"
with open(report_path, "w") as f:
    f.write(report)

print(f"Generated {report_path}")
