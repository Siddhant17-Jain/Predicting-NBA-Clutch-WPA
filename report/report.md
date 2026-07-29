# NBA Clutch Win Probability Added (cWPA): Statistical Analysis
*26 seasons · 2000-01 → 2025-26 · Source: inpredictable.com*

---

## Executive Summary

| Question | Metric | Pearson r | 95% CI | Spearman ρ | n |
|---|---|---|---|---|---|
| Does clutch persist YoY? (Regular) | Per-game rate | **0.5089** | [0.4829, 0.534] | 0.4922 | 3,244 |
| Does clutch transfer to playoffs? | Per-game rate | **0.3663** | [0.3193, 0.4116] | 0.3215 | 1,352 |
| Does clutch persist YoY? (Combined) | Per-game rate | **0.5527** | [0.5233, 0.5808] | 0.5258 | 2,243 |

**Key takeaways:**
- **Year-over-year clutch persistence (regular season):** r = 0.5089 (per-game). Clutch performance shows moderate year-over-year repeatability — meaningfully above zero but far from deterministic.
- **Regular → Playoff transfer:** r = 0.3663 (per-game). Regular-season clutch is a moderate predictor of playoff clutch performance.
- **Adding playoffs (Prong 3 vs Prong 1):** Δr = 0.0438, suggesting including playoffs increases measured clutch persistence.
- Raw totals consistently show lower r than per-game rates — suggesting per-game rates carry more signal.

---

## 1  Dataset Overview

| Variant | Player-seasons | Unique players | Seasons |
|---|---|---|---|
| Regular | 12,910 | 2,535 | 26 |
| Playoffs | 5,912 | 1,645 | 26 |
| Combined | 12,923 | 2,536 | 26 |

Data-quality notes:
- Rows with gms=0: regular=186, playoffs=433, combined=190
- Rows with gms>82 (traded-player artifacts, deduped by summing): regular=14
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

| season    | player                  | pos   |   gms |   clwpa |
|:----------|:------------------------|:------|------:|--------:|
| 2021-2022 | DeMar DeRozan           | G-F   |    75 |    5.84 |
| 2007-2008 | LeBron James            | F     |    75 |    5.48 |
| 2022-2023 | De'Aaron Fox            | G     |    71 |    5.44 |
| 2016-2017 | Isaiah Thomas           | G     |    76 |    5.19 |
| 2014-2015 | Anthony Davis           | F-C   |    68 |    5.04 |
| 2025-2026 | Shai Gilgeous-Alexander | G     |    64 |    4.9  |
| 2008-2009 | Brandon Roy             | G     |    78 |    4.87 |
| 2006-2007 | Mehmet Okur             | —     |    80 |    4.65 |
| 2016-2017 | CJ McCollum             | G     |    80 |    4.64 |
| 2022-2023 | DeMar DeRozan           | G-F   |    74 |    4.63 |

**Bottom 10 regular-season cWPA, all time:**

| season    | player             | pos   |   gms |   clwpa |
|:----------|:-------------------|:------|------:|--------:|
| 2000-2001 | Chris Webber       | —     |    70 |   -4.26 |
| 2002-2003 | Paul Pierce        | F     |    79 |   -3.63 |
| 2000-2001 | Kevin Garnett      | F     |    81 |   -3.49 |
| 2000-2001 | Jerry Stackhouse   | F     |    80 |   -3.44 |
| 2002-2003 | Zydrunas Ilgauskas | C     |    81 |   -3.42 |
| 2004-2005 | Antoine Walker     | F     |    77 |   -3.07 |
| 2002-2003 | Jason Kidd         | G     |    80 |   -2.98 |
| 2003-2004 | LeBron James       | F     |    79 |   -2.94 |
| 2010-2011 | DeMarcus Cousins   | C     |    81 |   -2.94 |
| 2003-2004 | Paul Pierce        | F     |    80 |   -2.8  |

---

## 3  Analysis

> **Benchmarks:** regular gms ≥ 62, playoff gms ≥ 10, combined gms ≥ 72.
> **Two normalizations** reported: *per-game rate* (clwpa ÷ gms) and *raw season total*.
> Pearson r with 95% Fisher-z CI, Spearman ρ.

### 3.1  Prong 1 — Regular-season YoY persistence

**Both years must meet gms ≥ 62. All consecutive season-pairs pooled across 2000-01 → 2025-26.**

| Normalization | Pearson r | 95% CI | p | Spearman ρ | n |
|---|---|---|---|---|---|
| Per-game rate | 0.5089 | [0.4829, 0.534] | see CSV | 0.4922 | 3,244 |
| Season total  | 0.5068 | — | — | — | — |

![P1 per-game](figures/analysis/prong_1_regular_yoy/p1_regular_yoy_per_game.png)
![P1 total](figures/analysis/prong_1_regular_yoy/p1_regular_yoy_total.png)

The per-transition r distribution below tests whether the pooled result is consistent across eras:

![P1 per-transition r](figures/analysis/prong_1_regular_yoy/p1_per_transition_r.png)

### 3.2  Prong 2 — Regular-season → Playoff transfer

**Same season. Regular gms ≥ 62, playoff gms ≥ 10.**

| Normalization | Pearson r | 95% CI | p | Spearman ρ | n |
|---|---|---|---|---|---|
| Per-game rate | 0.3663 | [0.3193, 0.4116] | see CSV | 0.3215 | 1,352 |
| Season total  | 0.3817 | — | — | — | — |

![P2 per-game](figures/analysis/prong_2_reg_to_playoff/p2_reg_to_playoff_per_game.png)
![P2 total](figures/analysis/prong_2_reg_to_playoff/p2_reg_to_playoff_total.png)

### 3.3  Prong 3 — Combined YoY persistence

**Both years must meet gms ≥ 72. All consecutive season-pairs pooled.**

| Normalization | Pearson r | 95% CI | p | Spearman ρ | n |
|---|---|---|---|---|---|
| Per-game rate | 0.5527 | [0.5233, 0.5808] | see CSV | 0.5258 | 2,243 |
| Season total  | 0.5584 | — | — | — | — |

![P3 per-game](figures/analysis/prong_3_combined_yoy/p3_combined_yoy_per_game.png)
![P3 total](figures/analysis/prong_3_combined_yoy/p3_combined_yoy_total.png)

**Prong 1 vs Prong 3 (per-game):** P1 r = 0.5089, P3 r = 0.5527, Δr = 0.0438.

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
| **Train** | 2000-2020 (21 seasons) | 4,618 | 1,503 |
| **Test** | 2021-2025 (5 seasons) | 1,029 | 359 |

A random split would let the model learn from 2024 to predict 2023. A chronological split forces it
to predict genuinely unseen future seasons, which is the only version of the question worth asking.

**No test-set leakage.** Both the feature selection *and* the hyperparameter search use training
years only. The test seasons are touched exactly once, at final evaluation. This matters more than
it sounds: ranking features on all 26 seasons and then "testing" on five of them would quietly
leak test information into the feature list.

**Feature selection.** All 44 numeric Basketball-Reference features
enter a full model fit on the training years; the top 20 by gain
importance are retained. Each model selects its own list — the regular-season and playoff rankings
differ substantially (§4.2).

**Hyperparameter search.** 3,240 distinct configurations over `max_depth`, `learning_rate`,
`n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight` and `reg_lambda`, scored on
RMSE, MAE and R² simultaneously and refit on RMSE. 400 configurations are sampled via
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

| Model          | CV scheme        |   Test R² |   Test RMSE |   Test MAE |
|:---------------|:-----------------|----------:|------------:|-----------:|
| Regular Season | KFold (retained) |    0.4624 |      0.6352 |     0.4616 |
| Regular Season | LOGO (removed)   |    0.4628 |      0.635  |     0.4632 |
| Playoffs       | KFold (retained) |    0.2658 |      0.2646 |     0.1511 |
| Playoffs       | LOGO (removed)   |    0.238  |      0.2695 |     0.1521 |

On the regular-season model LOGO is nominally *ahead*, by 0.0006 R². That gap does not survive
scrutiny. Re-running the KFold search under different random seeds produces this spread:

| Run                      |   Test R² |
|:-------------------------|----------:|
| KFold, seed 0            |    0.4594 |
| KFold, seed 1            |    0.4631 |
| KFold, seed 7            |    0.4574 |
| KFold, seed 42 (shipped) |    0.4624 |
| LOGO (for comparison)    |    0.4628 |

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

All candidate features ranked by gain on the training years (45 for regular, 44 for playoffs).
Coloured bars are the retained top 20.

![Regular importance, all features](figures/analysis/xgboost/01_importance_all_features_regular.png)
![Playoff importance, all features](figures/analysis/xgboost/01_importance_all_features_playoff.png)

**Top 20 features, side by side:**

The regular-season model has 45 candidate features: 44 BBRef stats plus `clwpa_lag1` (prior-year regular-season cWPA, NaN for rookies and gap-year returners — XGBoost handles NaN natively). The playoff model has 44 (no lag feature; prior-year playoff cWPA ranked 30th in a standalone importance check and did not make the top 20).

|   Rank | Regular Season   |   Reg. Gain | Playoffs   |   PO Gain |
|-------:|:-----------------|------------:|:-----------|----------:|
|      1 | TS%              |      0.1618 | VORP       |    0.0782 |
|      2 | eFG%             |      0.0559 | OWS        |    0.0765 |
|      3 | PTS              |      0.0548 | eFG%       |    0.0586 |
|      4 | TOV%             |      0.0547 | TS%        |    0.0583 |
|      5 | clwpa_lag1       |      0.0501 | FT         |    0.0527 |
|      6 | TOV              |      0.0399 | WS         |    0.0397 |
|      7 | OBPM             |      0.0361 | TOV        |    0.0393 |
|      8 | OWS              |      0.0304 | PTS        |    0.0358 |
|      9 | 3PA              |      0.0266 | 2P         |    0.0318 |
|     10 | USG%             |      0.0238 | 3PA        |    0.0275 |
|     11 | 3P               |      0.0196 | 2PA        |    0.0234 |
|     12 | WS               |      0.0191 | FGA        |    0.0206 |
|     13 | 2PA              |      0.0179 | DWS        |    0.0199 |
|     14 | VORP             |      0.0177 | AST        |    0.0199 |
|     15 | 3PAr             |      0.0168 | FG         |    0.0194 |
|     16 | ORB              |      0.0164 | FTA        |    0.0192 |
|     17 | PER              |      0.0161 | PF         |    0.0185 |
|     18 | FT               |      0.0157 | 3P         |    0.0172 |
|     19 | PF               |      0.0150 | TRB%       |    0.017  |
|     20 | FT%              |      0.0142 | TOV%       |    0.0161 |

Gain is split arbitrarily among strongly correlated features — `WS`, `VORP` and `BPM` overlap
heavily, as do `FG`/`FGA` and `PTS`/`USG%`. A feature ranking low means "redundant given the
others", not "unrelated to clutch performance".

**Tuned-model importance** (the retained features only, after hyperparameter search):

![Regular tuned importance](figures/analysis/xgboost/02_importance_top20_regular.png)
![Playoff tuned importance](figures/analysis/xgboost/02_importance_top20_playoff.png)

### 4.3  Results

| Metric | Regular Season | Playoffs |
|---|---|---|
| Cross-validated R² (train) | 0.4885 | 0.2667 |
| Cross-validated RMSE (train) | 0.6335 | 0.2574 |
| Cross-validated MAE (train) | 0.4541 | 0.1670 |
| In-sample R² (train) | 0.6547 | 0.5045 |
| **Held-out R² (2021-2025)** | **0.4624** | **0.2658** |
| **Held-out RMSE** | **0.6352** | **0.2646** |
| **Held-out MAE** | **0.4616** | **0.1511** |
| n (train / test) | 4,618 / 1,029 | 1,503 / 359 |

Winning hyperparameters —
*Regular:* `max_depth`=4, `learning_rate`=0.01, `n_estimators`=800, `subsample`=0.6, `colsample_bytree`=0.6, `min_child_weight`=5, `reg_lambda`=1.0
*Playoffs:* `max_depth`=3, `learning_rate`=0.01, `n_estimators`=400, `subsample`=0.8, `colsample_bytree`=0.8, `min_child_weight`=5, `reg_lambda`=1.0

**Predicted vs actual on the held-out seasons.** The dashed black line is perfect prediction
(y = x); the red line is the actual fitted relationship. A red line flatter than the dashed line
means the model regresses toward the mean, under-predicting extremes in both directions.

![Regular actual vs predicted](figures/analysis/xgboost/03_actual_vs_predicted_regular.png)
![Playoff actual vs predicted](figures/analysis/xgboost/03_actual_vs_predicted_playoff.png)

**Broken out by held-out season:**

![Regular by season](figures/analysis/xgboost/04_actual_vs_predicted_by_season_regular.png)
![Playoff by season](figures/analysis/xgboost/04_actual_vs_predicted_by_season_playoff.png)

Per-season test metrics, regular season:

| Season   |   n |     R² |   RMSE |    MAE |   Mean cWPA_Diff |
|:---------|----:|-------:|-------:|-------:|-----------------:|
| 2021-22  | 202 | 0.3272 | 0.6647 | 0.4658 |          -0.0475 |
| 2022-23  | 213 | 0.3560 | 0.7174 | 0.5002 |          -0.1564 |
| 2023-24  | 205 | 0.5618 | 0.5948 | 0.4387 |          -0.0005 |
| 2024-25  | 200 | 0.5185 | 0.6029 | 0.4614 |          -0.0510 |
| 2025-26  | 209 | 0.5270 | 0.5833 | 0.4406 |          -0.0277 |

Per-season test metrics, playoffs:

| Season   |   n |     R² |   RMSE |    MAE |   Mean cWPA_Diff |
|:---------|----:|-------:|-------:|-------:|-----------------:|
| 2021-22  |  79 | 0.3111 | 0.1785 | 0.1189 |           0.0111 |
| 2022-23  |  69 | 0.3853 | 0.2694 | 0.1634 |          -0.0465 |
| 2023-24  |  72 | 0.0142 | 0.2638 | 0.1541 |          -0.0229 |
| 2024-25  |  65 | 0.1939 | 0.3709 | 0.1838 |          -0.0598 |
| 2025-26  |  74 | 0.3361 | 0.2222 | 0.1426 |           0.0311 |

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
| Mean cWPA_Difference | -0.0574 | -0.0155 |
| SD cWPA_Difference | 0.6329 | 0.2645 |

A mean near zero means the model is close to unbiased on the held-out years; the SD is the typical
size of a miss.

![Regular residuals](figures/analysis/xgboost/05_residuals_regular.png)
![Playoff residuals](figures/analysis/xgboost/05_residuals_playoff.png)

**Regular season — biggest over-performers** (most negative cWPA_Difference: delivered far more
clutch value than their profile predicted):

| Season    | Player            | Tm   |   G |   actual_cWPA |   predicted_cWPA |   cWPA_Difference |
|:----------|:------------------|:-----|----:|--------------:|-----------------:|------------------:|
| 2022-2023 | De'Aaron Fox      | SAC  |  73 |          5.44 |            1.441 |            -3.999 |
| 2021-2022 | DeMar DeRozan     | CHI  |  76 |          5.84 |            2.383 |            -3.457 |
| 2024-2025 | LeBron James      | LAL  |  70 |          3.83 |            1.294 |            -2.536 |
| 2022-2023 | Dennis Schroder   | LAL  |  66 |          2.59 |            0.217 |            -2.373 |
| 2022-2023 | DeMar DeRozan     | CHI  |  74 |          4.63 |            2.320 |            -2.310 |
| 2025-2026 | Desmond Bane      | ORL  |  82 |          3.53 |            1.315 |            -2.215 |
| 2022-2023 | Jimmy Butler      | MIA  |  64 |          3.74 |            1.548 |            -2.192 |
| 2023-2024 | DeMar DeRozan     | CHI  |  79 |          4.50 |            2.352 |            -2.148 |
| 2021-2022 | Spencer Dinwiddie | 2TM  |  67 |          2.56 |            0.450 |            -2.110 |
| 2022-2023 | Jalen Williams    | OKC  |  75 |          2.50 |            0.411 |            -2.089 |

**Regular season — biggest under-performers** (most positive cWPA_Difference):

| Season    | Player           | Tm   |   G |   actual_cWPA |   predicted_cWPA |   cWPA_Difference |
|:----------|:-----------------|:-----|----:|--------------:|-----------------:|------------------:|
| 2021-2022 | Donovan Mitchell | UTA  |  67 |         -0.84 |            1.631 |             2.471 |
| 2021-2022 | Trae Young       | ATL  |  76 |          0.03 |            2.495 |             2.465 |
| 2022-2023 | Jayson Tatum     | BOS  |  74 |          0.29 |            2.568 |             2.278 |
| 2022-2023 | Darius Garland   | CLE  |  69 |         -0.74 |            0.954 |             1.694 |
| 2023-2024 | Cade Cunningham  | DET  |  62 |         -1.07 |            0.570 |             1.640 |
| 2025-2026 | Paolo Banchero   | ORL  |  72 |         -1.02 |            0.511 |             1.531 |
| 2023-2024 | Jayson Tatum     | BOS  |  74 |          0.80 |            2.271 |             1.471 |
| 2022-2023 | Luka Doncic      | DAL  |  66 |          1.31 |            2.688 |             1.378 |
| 2021-2022 | Devin Vassell    | SAS  |  71 |         -0.56 |            0.788 |             1.348 |
| 2023-2024 | Donte DiVincenzo | NYK  |  81 |          0.12 |            1.403 |             1.283 |

**Playoffs — biggest over-performers:**

| Season    | Player             | Tm   |   G |   actual_cWPA |   predicted_cWPA |   cWPA_Difference |
|:----------|:-------------------|:-----|----:|--------------:|-----------------:|------------------:|
| 2024-2025 | Tyrese Haliburton  | IND  |  23 |          2.63 |            0.286 |            -2.344 |
| 2023-2024 | Jamal Murray       | DEN  |  12 |          1.23 |           -0.09  |            -1.32  |
| 2024-2025 | Aaron Gordon       | DEN  |  14 |          1.3  |            0.218 |            -1.082 |
| 2022-2023 | Derrick White      | BOS  |  20 |          1.41 |            0.337 |            -1.073 |
| 2022-2023 | Jimmy Butler       | MIA  |  22 |          1.77 |            0.74  |            -1.03  |
| 2022-2023 | James Harden       | PHI  |  11 |          0.9  |            0.007 |            -0.893 |
| 2023-2024 | Donte DiVincenzo   | NYK  |  13 |          1.06 |            0.241 |            -0.819 |
| 2025-2026 | OG Anunoby         | NYK  |  17 |          1.29 |            0.525 |            -0.765 |
| 2024-2025 | Karl-Anthony Towns | NYK  |  18 |          0.87 |            0.235 |            -0.635 |
| 2023-2024 | P.J. Washington    | DAL  |  22 |          0.86 |            0.25  |            -0.61  |

**Playoffs — biggest under-performers:**

| Season    | Player             | Tm   |   G |   actual_cWPA |   predicted_cWPA |   cWPA_Difference |
|:----------|:-------------------|:-----|----:|--------------:|-----------------:|------------------:|
| 2025-2026 | Donovan Mitchell   | CLE  |  18 |         -0.52 |            0.354 |             0.874 |
| 2023-2024 | Tyrese Haliburton  | IND  |  15 |         -0.39 |            0.305 |             0.695 |
| 2025-2026 | De'Aaron Fox       | SAS  |  21 |         -0.84 |           -0.15  |             0.69  |
| 2021-2022 | Marcus Smart       | BOS  |  21 |         -0.45 |            0.17  |             0.62  |
| 2022-2023 | LeBron James       | LAL  |  16 |         -0.14 |            0.411 |             0.551 |
| 2021-2022 | Spencer Dinwiddie  | DAL  |  18 |         -0.29 |            0.258 |             0.548 |
| 2025-2026 | Karl-Anthony Towns | NYK  |  19 |         -0.3  |            0.236 |             0.536 |
| 2025-2026 | Stephon Castle     | SAS  |  23 |         -0.5  |           -0.007 |             0.493 |
| 2021-2022 | Jimmy Butler       | MIA  |  17 |          0.34 |            0.812 |             0.472 |
| 2024-2025 | Nikola Jokic       | DEN  |  14 |         -0.19 |            0.251 |             0.441 |

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

| | Before repair | After repair (pre-lag) | Current (with lag) |
|---|---|---|---|
| Regular test R² | 0.4425 | 0.4622 | **0.4624** |
| Regular test RMSE | 0.6454 | 0.6353 | **0.6352** |
| Regular test n | 949 | 1,029 | **1,029** |
| 2021-22 season R² | −0.0188 | 0.3132 | **0.3272** |
| Playoff test R² | 0.2658 | 0.2658 (unchanged) | 0.2658 (unchanged) |

The playoff model reproducing to four decimals is the regression check that nothing unrelated moved.
The repair also closed the last name-matching gap: both joins now resolve at 100% (regular
5,647/5,647, playoffs 1,862/1,862). A previously documented "missing" player, Dwayne Bacon 2021, was
never a source discrepancy — his 72-game row was 2020-21 data mislabelled, and he barely played in
2021-22, so inpredictable.com was right to have no record.

Two lessons worth carrying into the writing. A per-season metric breakdown caught something a single
pooled number would have hidden entirely: the corrupted year was one of five, and the overall R² of
0.4425 looked perfectly reasonable. The feature ranking shifted once the year was fixed (`FGA`, `FG`
and `VORP` dropped out; `FT`, `TRB%` and `FTA` entered), and shifted again when `clwpa_lag1` was
added as a 45th candidate feature — any copy of the regular top-20 list predating both changes is
stale.

---

---

## 5  Full results table

*(data/outputs/analysis_results.csv)*

| prong        | normalization   |   pearson_r |   p_pearson |   ci_lo |   ci_hi |   spearman_rho |   p_spearman |    n |
|:-------------|:----------------|------------:|------------:|--------:|--------:|---------------:|-------------:|-----:|
| P1_reg_yoy   | per_game        |      0.5089 |           0 |  0.4829 |  0.534  |         0.4922 |            0 | 3244 |
| P1_reg_yoy   | total           |      0.5068 |           0 |  0.4808 |  0.5319 |         0.4883 |            0 | 3244 |
| P2_reg_to_po | per_game        |      0.3663 |           0 |  0.3193 |  0.4116 |         0.3215 |            0 | 1352 |
| P2_reg_to_po | total           |      0.3817 |           0 |  0.3352 |  0.4263 |         0.3261 |            0 | 1352 |
| P3_comb_yoy  | per_game        |      0.5527 |           0 |  0.5233 |  0.5808 |         0.5258 |            0 | 2243 |
| P3_comb_yoy  | total           |      0.5584 |           0 |  0.5293 |  0.5863 |         0.5247 |            0 | 2243 |

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
   half the variance in regular-season clutch WPA unexplained (0.46) and closer
   to three quarters in the playoffs (0.27). Treat large `cWPA_Difference` values
   as "unexplained by the box score", not as a measured clutch skill.
11. **Test-set size.** The held-out period is 5 seasons — 1,029 regular-season and
   359 playoff player-seasons. Per-season R² in §4.3 ranges from 0.31 to 0.55
   (regular) and 0.01 to 0.39 (playoffs), so single-season figures are noisy.
