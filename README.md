# Predicting NBA Clutch WPA

26 seasons of clutch Win Probability Added data, covering every qualifying NBA player from 2000-01 through 2025-26. Three research questions: does clutch performance repeat year over year? Does it transfer to the playoffs? Which box-score statistics can predict it?

---

## The Metric

**cWPA (clutch Win Probability Added)** comes from inpredictable.com. It measures a player's net impact on win probability during clutch situations: within five points, final five minutes. Shots, turnovers, free throws, rebounds, assists, steals, and blocks all count. Positive means the player helped; negative means they hurt. The window is narrower than what most people call "clutch," which is exactly what makes the data useful.

Three variants: **regular season**, **playoffs**, and **combined** (regular + playoffs folded together). The raw dataset covers 26 seasons, 31,745 player-season rows.

For the correlation analysis: minimum 62 games for regular season, 10 for playoffs. Those thresholds cut injury-shortened seasons, late call-ups, and first-round sweep victims with minimal clutch exposure. After filtering: roughly 5,500 qualifying regular-season player-seasons and 1,800 playoff entries.

---

## Findings

### 1. Regular Season, Year Over Year

Across 3,244 consecutive season pairs, 25 transitions (2000-01 through 2024-25):

**Pearson r = 0.509. R² = 0.26. Spearman ρ = 0.492.**

Prior-year cWPA explains 26% of the variance in next-year cWPA. The correlation holds across every era tested. No single decade drives it. Chris Paul finished with positive cWPA in all 13 of his qualifying regular seasons. Kevin Durant, same. 95% confidence interval: [0.483, 0.534].

The caveat: 74% unexplained. Prior-year cWPA predicts direction, not precise rank order.

### 2. Regular Season to Same-Season Playoffs

Across 1,352 player-seasons qualifying in both regular season and playoffs in the same year:

**Pearson r = 0.366. R² = 0.15. Spearman ρ = 0.322.**

The signal drops by 11 percentage points versus year-over-year. Smaller samples, better opponents, targeted defensive schemes, and physical attrition across a series all add noise. Regular-season clutch production does not transfer cleanly to the playoffs.

### 3. Combined Year Over Year

Across 2,243 season pairs using the combined variant:

**Pearson r = 0.553. R² = 0.31. Spearman ρ = 0.525.**

The strongest signal of the three. A player who produces in both settings in the same year provides stronger evidence of a real skill than either number alone. Combined cWPA is the best available predictor of future clutch performance in this dataset.

---

## XGBoost Models

Two independent models: regular season and playoffs. Training window: 2000-01 through 2020-21 (21 seasons). Test window: 2021-22 through 2025-26 (5 held-out seasons, never touched during training or feature selection). Features: 44 standard Basketball-Reference box-score stats, narrowed to the top 20 by importance gain. The regular-season model adds one engineered feature: prior-year regular-season cWPA (`clwpa_lag1`), which ranks 5th in importance and directly reflects the year-over-year persistence signal.

Hyperparameter search: 400 randomly sampled configurations, 5-fold cross-validation, scored on RMSE.

| | Regular Season | Playoffs |
|---|---|---|
| **Test R²** | **0.4624** | **0.2658** |
| **Test RMSE** | **0.6352** | **0.2646** |
| **Test MAE** | **0.4616** | **0.1511** |
| Train R² | 0.6547 | 0.5045 |
| CV R² | 0.4885 | 0.2667 |
| n (train / test) | 4,618 / 1,029 | 1,503 / 359 |

The regular-season model explains 46% of clutch variance on unseen data. The playoff model: 27%. Both beat the naive baseline. More than half the variance in both cases goes unexplained. That gap is where clutch separates from box-score basketball.

Winning hyperparameters — *Regular:* `max_depth=4, learning_rate=0.01, n_estimators=800, subsample=0.6, colsample_bytree=0.6, min_child_weight=5, reg_lambda=1.0`. *Playoffs:* `max_depth=3, learning_rate=0.01, n_estimators=400, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0`. Both models converged on slow learning rates and many shallow trees, consistent with a noisy target.

### Feature Importance (Top 10)

**Regular season:** `TS%, eFG%, PTS, TOV%, clwpa_lag1, TOV, OBPM, OWS, 3PA, USG%`

Efficiency leads. Players who convert at a high rate and protect the ball tend to produce positive clutch WPA. Prior-year cWPA ranks 5th.

**Playoffs:** `VORP, OWS, eFG%, TS%, FT, WS, TOV, PTS, 2P, 3PA`

Cumulative value leads instead of precision. Playoff clutch production tracks overall player quality more than any specific efficiency split. The best players get more volume in clutch playoff moments, and that volume converts.

---

## Leaderboards

All leaderboard numbers below use raw cWPA totals with no games threshold applied.

### All-Time Regular Season (26 seasons, 2000-01 through 2025-26)

| # | Player | Career cWPA | Seasons |
|---|--------|-------------|---------|
| 1 | LeBron James | +45.00 | 23 |
| 2 | Kevin Durant | +35.00 | 19 |
| 3 | DeMar DeRozan | +32.02 | 17 |
| 4 | Stephen Curry | +30.05 | 17 |
| 5 | Dirk Nowitzki | +29.78 | 19 |
| 6 | James Harden | +26.83 | 17 |
| 7 | Chris Paul | +26.05 | 21 |
| 8 | Damian Lillard | +24.98 | 13 |
| 9 | Anthony Davis | +24.90 | 14 |
| 10 | Jimmy Butler | +20.83 | 14 |

LeBron leads through longevity as much as peak: 23 qualifying seasons. Durant and DeRozan follow, both positive in nearly every season they appeared.

### Last 4 Years Regular Season (2022-23 through 2025-26)

| # | Player | 4yr cWPA | Seasons |
|---|--------|----------|---------|
| 1 | DeMar DeRozan | +14.71 | 4 |
| 2 | Shai Gilgeous-Alexander | +13.41 | 4 |
| 3 | Stephen Curry | +11.11 | 4 |
| 4 | Nikola Jokic | +10.89 | 4 |
| 5 | LeBron James | +9.81 | 4 |
| 6 | Kevin Durant | +9.74 | 4 |
| 7 | De'Aaron Fox | +9.06 | 4 |
| 8 | Jalen Brunson | +8.98 | 4 |
| 9 | Jamal Murray | +8.36 | 4 |
| 10 | Austin Reaves | +8.33 | 4 |

DeRozan's 4-year total of 14.71 is the widest lead at the top of any rolling window in this dataset. His four individual seasons: +4.63, +4.50, +2.81, +2.77. No year below 2.77.

### All-Time Playoffs (26 seasons, 2000-01 through 2025-26)

| # | Player | Career cWPA | Seasons |
|---|--------|-------------|---------|
| 1 | LeBron James | +10.80 | 19 |
| 2 | Jimmy Butler | +4.78 | 12 |
| 3 | Ray Allen | +4.41 | 9 |
| 4 | Dirk Nowitzki | +4.04 | 15 |
| 5 | Chris Paul | +4.04 | 16 |
| 6 | Kevin Durant | +3.88 | 14 |
| 7 | OG Anunoby | +3.59 | 7 |
| 8 | James Harden | +3.56 | 17 |
| 9 | Kawhi Leonard | +3.09 | 13 |
| 10 | Al Horford | +3.07 | 16 |

LeBron's playoff total is more than double the next closest player. OG Anunoby ranks 7th all-time in only 7 playoff seasons.

### Last 4 Years Playoffs (2022-23 through 2025-26)

| # | Player | 4yr cWPA | Seasons |
|---|--------|----------|---------|
| 1 | Tyrese Haliburton | +2.24 | 2 |
| 2 | Jalen Brunson | +2.11 | 4 |
| 3 | Shai Gilgeous-Alexander | +2.04 | 4 |
| 4 | OG Anunoby | +1.95 | 3 |
| 5 | Jimmy Butler | +1.80 | 2 |
| 6 | Aaron Gordon | +1.78 | 4 |
| 7 | Derrick White | +1.77 | 4 |
| 8 | James Harden | +1.57 | 4 |
| 9 | Tyrese Maxey | +1.41 | 3 |
| 10 | Donte DiVincenzo | +1.35 | 4 |

Haliburton leads despite qualifying in only 2 of 4 seasons. His 2024-25 postseason (2.63 cWPA) is the best single playoff clutch season in the 26-year dataset. Dirk Nowitzki held that record at 2.15 from the 2010-11 championship run.

---

## Single-Season Records

### Regular Season

| Rank | Player | Season | cWPA |
|------|--------|--------|------|
| 1 | DeMar DeRozan | 2021-22 | +5.84 |
| 2 | LeBron James | 2007-08 | +5.48 |
| 3 | De'Aaron Fox | 2022-23 | +5.44 |
| 4 | Isaiah Thomas | 2016-17 | +5.19 |
| 5 | Shai Gilgeous-Alexander | 2025-26 | +4.90 |

DeRozan's 2021-22 stands apart. The next four cluster between 4.90 and 5.48. The median qualifying player-season is 0.21. The distribution is right-skewed with a long thin tail.

### Playoffs

Haliburton 2024-25: +2.63. Nowitzki 2010-11: +2.15 (previous record). The gap between first and second is larger than the gap between second and tenth.

---

## Model Over and Under-Performers (Test Window: 2021-22 through 2025-26)

cWPA Difference = predicted minus actual. Negative means the player over-delivered relative to their box score. Positive means they fell short.

### Regular Season Over-Performers

| Season | Player | Predicted | Actual | Difference |
|--------|--------|-----------|--------|------------|
| 2022-23 | De'Aaron Fox | +1.44 | +5.44 | -4.00 |
| 2021-22 | DeMar DeRozan | +2.38 | +5.84 | -3.46 |
| 2024-25 | LeBron James | +1.29 | +3.83 | -2.54 |
| 2022-23 | Dennis Schroder | +0.22 | +2.59 | -2.37 |
| 2022-23 | DeMar DeRozan | +2.32 | +4.63 | -2.31 |

Fox's 2022-23 gap of 4.00 wins is the largest in the test set. DeRozan appears twice in the top five. Both are mid-range-heavy players whose clutch shot conversion doesn't surface in aggregate efficiency metrics. The model sees a mid-range shooter and projects modest output. They deliver something different.

### Regular Season Under-Performers

| Season | Player | Predicted | Actual | Difference |
|--------|--------|-----------|--------|------------|
| 2021-22 | Donovan Mitchell | +1.63 | -0.84 | +2.47 |
| 2021-22 | Trae Young | +2.49 | +0.03 | +2.46 |
| 2022-23 | Jayson Tatum | +2.57 | +0.29 | +2.28 |
| 2022-23 | Darius Garland | +0.95 | -0.74 | +1.69 |
| 2023-24 | Cade Cunningham | +0.57 | -1.07 | +1.64 |

Mitchell and Young are separated by 0.006 cWPA, which is effectively the same miss. Both had elite efficiency profiles that the model correctly translated into high clutch projections. Neither delivered in the five-minutes-within-five-points window that season.

### Playoff Over-Performers

| Season | Player | Predicted | Actual | Difference |
|--------|--------|-----------|--------|------------|
| 2024-25 | Tyrese Haliburton | +0.29 | +2.63 | -2.34 |
| 2022-23 | Jamal Murray | -0.09 | +1.23 | -1.32 |
| 2024-25 | Aaron Gordon | +0.22 | +1.30 | -1.08 |
| 2022-23 | Derrick White | +0.34 | +1.41 | -1.07 |
| 2022-23 | Jimmy Butler | +0.74 | +1.77 | -1.03 |

Haliburton's miss is the largest in the playoff dataset. The projection of 0.29 was reasonable given his prior playoff track record. He was an under-performer in 2023-24 (-0.39 actual vs +0.31 projected). In 2024-25, he posted 2.63.

### Playoff Under-Performers

| Season | Player | Predicted | Actual | Difference |
|--------|--------|-----------|--------|------------|
| 2025-26 | Donovan Mitchell | +0.35 | -0.52 | +0.87 |
| 2023-24 | Tyrese Haliburton | +0.31 | -0.39 | +0.70 |
| 2025-26 | De'Aaron Fox | -0.15 | -0.84 | +0.69 |
| 2021-22 | Marcus Smart | +0.17 | -0.45 | +0.62 |
| 2022-23 | LeBron James | +0.41 | -0.14 | +0.55 |

---

## Sensitivity Analysis

The primary thresholds (62 games for regular, 10 for playoffs) were tested across a grid of alternatives: [41, 52, 62, 72, 82] for regular/combined, [5, 8, 10, 13, 16] for playoffs. The correlation results are not threshold-sensitive. Pearson r for regular-season YoY ranges from 0.49 to 0.52 across the full grid.

---

## Blog Post

`report/blog_post.md` and `report/blog_post.docx` contain the full written analysis. The blog covers the Knicks' 2024-25 and 2025-26 comeback context (down 20 twice against the Celtics, down 29 in the third quarter against the Spurs in the greatest comeback in playoff history), detailed narrative for each finding, per-player career arcs, and what the model's failures tell us about what box scores can and cannot measure.

---

## Figures

**EDA** (`figures/eda/`):
- `coverage/01_coverage_grid.png` — unique players per season per variant
- `distributions/02_gms_distributions.png` — games played histograms, three panels
- `distributions/03_position_breakdown.png` — position breakdown, regular season
- `distributions/04_clwpa_histogram.png` — cWPA distribution with skew and kurtosis
- `distributions/05_clwpa_per_game_histogram.png` — per-game rate distribution
- `distributions/06_seasonal_violin_regular.png` — per-season violin, full unfiltered data
- `distributions/07_league_yearly_aggregates.png` — mean, median, total cWPA and strong clutch count by season
- `careers/08_career_top25_{regular,playoffs,combined}.png` — career cWPA top-25 bar charts
- `careers/09_career_longevity.png` — career longevity histogram and total vs. seasons scatter

**Correlation Analysis** (`figures/analysis/`):
- `prong_1_regular_yoy/` — year-over-year scatter (total and per-game), per-transition Pearson r bar chart across 25 season transitions
- `prong_2_reg_to_playoff/` — regular-to-playoff transfer scatter (total and per-game)
- `prong_3_combined_yoy/` — combined variant year-over-year scatter
- `sensitivity/sensitivity_sweep.png` — Pearson r across the full threshold grid

**XGBoost** (`figures/analysis/xgboost/`):
- `01_importance_all_features_{regular,playoff}.png` — all 44 features ranked by gain
- `02_importance_top20_{regular,playoff}.png` — top 20 features
- `03_actual_vs_predicted_{regular,playoff}.png` — predicted vs. actual cWPA scatter
- `04_actual_vs_predicted_by_season_{regular,playoff}.png` — per-season breakdown
- `05_residuals_{regular,playoff}.png` — residual plots

**Blog** (`figures/blog/`):
- `01_seasons_scatter.png` — every qualifying regular-season player-season as a dot (n=5,505), top 5 highlighted with annotations
- `02_career_leaders.png` — all-time career cWPA dot plot, top 30, G≥62
- `03_five_year_reg.png` — 5-year regular-season leaders dot plot, top 30, G≥62

---

## Data Sources

**inpredictable.com** — cWPA, 26 seasons (2000-01 through 2025-26), three variants (regular, playoffs, combined). 31,745 raw rows. Scraped with `src/scrape_wpa.py`.

**Basketball-Reference** — standard box-score stats, 2000-01 through 2025-26, regular season and playoffs. Joined to cWPA by normalized player name and season year. 100% join rate on both regular (5,647 rows) and playoff (1,862 rows) qualifying datasets.

---

*Data: inpredictable.com (cWPA), Basketball-Reference (box-score stats). 2000-01 through 2025-26.*
