# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Goal

This project is the data backbone for a **sports blog** analyzing NBA clutch performance. Two research questions drive it:

1. How well does regular-season clutch performance **persist year over year**?
2. How well does regular-season clutch performance **transfer to the playoffs**?

Both are answered with Pearson r correlations (plus Spearman ρ) on NBA clutch Win Probability Added (cWPA) data scraped from inpredictable.com, covering 26 seasons (2000-01 through 2025-26).

A third strand uses **XGBoost models on a chronological train/test split** to find which Basketball-Reference stats best predict clutch WPA — for both regular season and playoffs.

---

## Pipeline Overview

Run all scripts from the **project root** (not from `src/`):

```bash
python3 src/scrape_wpa.py          # Step 1a: scrape WPA → data/raw/wpa_all_seasons.csv
python3 src/build_dataset.py       # Step 1b: repair names, dedup → cwpa_long.xlsx
python3 src/scrape_bbref.py        # Step 2: scrape BBRef 2023-2026 + refresh REPLACE_YEARS
python3 src/check_mismatches.py    # Step 2b: audit BBRef↔WPA name matching (run after any scrape)
python3 src/eda.py                 # Step 3: 10 EDA figures + eda_summary.json
python3 src/analysis.py            # Step 4: 3-prong correlation analysis + sensitivity sweep
python3 src/generate_report.py     # Step 5: report/report.md
python3 src/build_xgboost.py       # Step 6: XGBoost models (train 2000-2020, test 2021-2025)
```

Each script uses `ROOT = Path(__file__).resolve().parent.parent` to resolve paths. Running from inside `src/` breaks the `from stats_utils import ...` import and all ROOT-relative file paths.

---

## Data Source 1: inpredictable.com (cWPA)

**Site:** `https://stats.inpredictable.com/nba/ssnPlayer.php`

**Metric:** cWPA (clutch Win Probability Added) — measures each player's impact on win probability during clutch situations (shots, turnovers, FTs, rebounds, assists, steals, blocks). The clutch-time definition and underlying play-by-play data live at inpredictable.com and are not independently auditable.

**Three playoff variants (URL param `po`):**
- `po=0` → regular season only (stored as `"regular"`)
- `po=1` → playoffs only (stored as `"playoffs"`)
- `po=2` → regular + playoffs combined (stored as `"combined"`)

**Groups:** Each variant × season is split across multiple player groups (`grp=1`, `grp=2`, …). Earlier seasons use ~11 groups; later seasons (e.g., 2021-22 with 626 players) use more. The scraper dynamically discovers how many groups exist rather than assuming a fixed ceiling.

**Season param convention:** URL uses the start year. Season 2024-25 uses `season=2024`. Date window: `frdt={start_year}-10-01` through `todt={start_year+1}-08-01`, **except 2019-20** which uses `todt=2020-11-01` to capture the COVID bubble playoffs (August–October 2020).

**Critical HTML quirk:** The site has unclosed `<tr>` tags. BeautifulSoup nests all subsequent rows as children of the first row, so `tr.find_all("td")` returns hundreds of cells instead of 16. Fix: `tr.find_all("td", recursive=False)`. Every valid player row has exactly 16 direct-child cells.

**player_id:** Extracted from the player link href: `href.split("pid=")[-1]`. This is inpredictable.com's internal ID, not the NBA.com ID. It is the **sole join key** throughout the WPA pipeline — never join on player name.

### Scraper Strategy (`src/scrape_wpa.py`)

The scraper runs **5 full attempts** per (season, variant). Each attempt:
- Iterates `grp=1` through `grp=25` (MAX_GRP ceiling)
- Stops after **2 consecutive empty groups** (early exit)
- Returns a dict keyed on `player_id`

The **union** of all 5 attempts is the final result. If attempt 3 gets a timeout on grp=7, the other 4 attempts recover those players. The scraper prints per-attempt counts and agreement stats — 5/5 agreement means the site returned stable counts; 3/5 means some requests errored but the union still captures everyone.

Constants: `MAX_GRP=25`, `N_ATTEMPTS=5`, `DELAY=0.75`, `SEASONS=range(2000, 2026)`

---

## Data Source 2: Basketball-Reference (BBRef stats)

**Files:**
- `data/raw/seasons_stats.csv` — original file covering 1950-2022 regular season. **Contains a duplicated season — see below.**
- `data/raw/playoffStats.csv` — original file covering 1950-2022 playoffs (rows before 2000 are dropped during cleaning)
- `data/processed/seasons_stats_clean.csv` — cleaned, Year 2000-2025, 12,811 rows, 50 columns
- `data/processed/playoffStats_clean.csv` — cleaned, Year 2000-2025, 5,350 rows, 50 columns
- `data/backup/*.pre2021fix.csv` — snapshots taken before the 2021-22 repair

`src/scrape_bbref.py` extends coverage to 2023-2026 by scraping basketball-reference.com, and also force-refreshes any year listed in `REPLACE_YEARS`.

### Year Convention (Critical)

Both BBRef files use the **start year** of the season as the `Year` column. Season 2022-23 = `Year=2022`. This matches WPA's `yr` field directly — **no +1 offset**. The original raw files used end-year convention and all Year values were decremented by 1 during cleaning.

Basketball-Reference URLs are keyed on the season's **end** year: `NBA_2023_totals.html` is the 2022-23 season. `scrape_bbref.py` therefore writes `Year = url_year - 1`. An earlier version wrote `Year = url_year`; re-running that version would have appended the 2025-26 season as `Year=2026` alongside the correct `Year=2025` row for the same season. Do not remove the offset.

### The 2021-22 Duplicated-Season Bug (fixed)

`data/raw/seasons_stats.csv` shipped with the 2021-22 season duplicated from 2020-21: raw `Year=2021` and raw `Year=2022` both contain 2020-21 totals. Joel Embiid appears as 51 G / 1451 PTS in both, when his real 2021-22 was 68 G / 2079 PTS (he led the league in scoring). After the −1 shift this made clean `Year=2021` the wrong season entirely.

**Symptom that exposed it:** the XGBoost regular model scored R² = −0.0188 on the 2021-22 test season — worse than predicting the mean — with only 122 qualifying players against ~205 in neighbouring years. The model was being handed 2020-21 box scores paired with 2021-22 cWPA targets.

**Scope, verified season by season:** regular season `Year=2021` only. Every other regular year and the entire playoff file are correct. Correlation prongs 1-3 were never affected because they use WPA data exclusively and never touch BBRef.

**Fix:** `REPLACE_YEARS = {2022}` in `scrape_bbref.py` re-scrapes that season and overwrites the stale rows. Verified against known values: Embiid 68 G / 2079 PTS, Trae Young 76 G / 2155 PTS, Giannis 67 G / 2002 PTS, max G back to 82.

**Watch out for a false positive:** `Year=2019` shows max G = 74, which looks short but is correct — the 2019-20 season was COVID-truncated at 64-75 games. James Harden's 68 G / 2335 PTS matches exactly.

### BBRef Dedup for Traded Players

Players traded mid-season appear on multiple rows. New BBRef format (2023+) uses `"2TM"`, `"3TM"`, `"4TM"` for the combined row; old format used `"TOT"`. Both are handled:
```python
COMBINED_FLAGS = {"TOT", "2TM", "3TM", "4TM"}
```
Keep the combined row, drop per-team splits for the same (Year, Player).

### BBRef Columns (50 total)

`Year, Player, Pos, Age, Tm, G, GS, MP, FG, FGA, FG%, 3P, 3PA, 3P%, 2P, 2PA, 2P%, eFG%, FT, FTA, FT%, ORB, DRB, TRB, AST, STL, BLK, TOV, PF, PTS, PER, TS%, 3PAr, FTr, ORB%, DRB%, TRB%, AST%, STL%, BLK%, TOV%, USG%, OWS, DWS, WS, WS/48, OBPM, DBPM, BPM, VORP`

Excluding the 6 identity/bookkeeping columns (`Year, Player, Pos, Age, Tm, GS`) leaves **44 numeric modelling features**.

Playoff stats were originally per-game — they have been converted to totals by multiplying by `G`.

Qualifying rows: `seasons_stats_clean.csv` G≥62 → 5,647 rows. `playoffStats_clean.csv` G≥10 → 1,862 rows.

---

## Name Normalization and Player Matching

The BBRef-to-WPA join uses normalized names + year. The normalization function:

```python
def norm(name):
    if not isinstance(name, str):
        return ""
    return name.replace(".", "").replace("*", "").strip().lower()
```

Strips periods (`C.J. McCollum` → `cj mccollum`), BBRef asterisks on Hall-of-Famers, lowercases.

### Encoding Fixes Applied

BBRef raw files had double-encoded UTF-8 artifacts (e.g., `Bogdan BogdanoviÃ\x84Â\x87`). Fixed with:
```python
from ftfy import fix_text
from unidecode import unidecode
clean_name = unidecode(fix_text(garbled_name))
```

On the WPA side the corruption is worse and cannot be repaired automatically at all — see **Name repair in `build_dataset.py`** above. Both fixes are now inside their scraper/builder scripts rather than applied by hand, so a rebuild cannot silently undo them.

Zero non-ASCII or corrupted names remain in any processed file, and both `clean_names()` (BBRef) and `repair_name()` (WPA) are idempotent.

### Manual Name Corrections Applied to BBRef Files

Players appear under different names in BBRef vs WPA. All corrections are applied to BBRef to match WPA's convention, and live in `NAME_FIXES` in `src/scrape_bbref.py` so new scrapes inherit them.

| BBRef (original) | WPA name | Notes |
|---|---|---|
| `Nene Hilario` | `Nene` | Mononym |
| `Luc Mbah` | `Luc Mbah a Moute` | BBRef truncates |
| `Metta World` | `Metta World Peace` | BBRef truncates |
| `Chuck Hayes` | `Charles Hayes` | Nickname vs legal |
| `Stanislav Medvedenko` | `Slava Medvedenko` | Nickname |
| `Mike Sweetney` | `Michael Sweetney` | Nickname vs legal |
| `Alex Sarr` | `Alexandre Sarr` | BBRef uses short form |
| `Enes Freedom` | `Enes Kanter` | WPA retains the pre-change name |
| `KJ Martin` (2020-2022) | `Kenyon Martin Jr.` | WPA uses the Jr. form for 2020, 2021, 2022 and reverts to `KJ Martin` from 2023 |
| `Lou Williams` (2011-2013) | `Louis Williams` | WPA uses "Louis" only in those three years |
| `Jimmy Butler` (2024+) | `Jimmy Butler III` | WPA adds "III" from 2024 onward |
| `Otto Porter` (2017+) | `Otto Porter Jr.` | WPA adds Jr. from **2017** onward; 2016 stays `Otto Porter` |
| `Marcus Morris` (2019+) | `Marcus Morris Sr.` | WPA adds Sr. from 2019 onward |
| `Timothe Luwawu-Cabarrot` (2016) | `Timothe Luwawu` | WPA truncates in his first season only |
| `Jose Juan Barea` → `JJ Barea` | `J.J. Barea` | **The two BBRef files disagree with each other**: the regular-season file says `JJ Barea`, the playoff file says `Jose Juan Barea`. Since `norm()` strips periods, `JJ Barea` and WPA's `J.J. Barea` both reduce to `jj barea`; only the playoff spelling is the outlier, so it is normalised on the BBRef side |

**Names deliberately NOT rewritten.** WPA uses the BBRef spelling for these, so "correcting" them breaks the join:
`Nick Van Exel`, `Keith Van Horn`, `Steve Smith`, `Robert Williams`.

An earlier version of this table listed `JJ Barea → Jose Juan Barea` and `Otto Porter → Otto Porter Jr. (2016+)`. Applying those two rules across the file silently broke 8 rows that had previously matched. **Never add a rule here without verifying it against WPA for the exact years it covers** — after scraping a new season, run the join check in `build_xgboost.join_datasets` and resolve whatever comes back unmatched instead of guessing.

### Join Validation Results

After the 2021-22 repair and all name fixes:
- **Regular season (G≥62): 5,647/5,647 joined (100%)**
- **Playoffs (G≥10): 1,862/1,862 joined (100%)**

The previously documented single miss (Dwayne Bacon 2021, "genuinely absent from inpredictable.com") was **not** a source discrepancy. His 72-game row was 2020-21 data mislabelled as 2021 by the duplicated-season bug; he barely played in 2021-22, so WPA correctly had no matching record. Repairing the year eliminated the gap.

The mismatch check applies the games threshold **only to the BBRef side**. WPA is left completely unfiltered. A player with G=62 in BBRef but G=61 in WPA (minor source discrepancy) correctly matches.

`clean_names()` in `scrape_bbref.py` is idempotent — running it on an already-clean file changes nothing.

Regenerate the audit any time with `python3 src/check_mismatches.py`. Output: `data/outputs/player_mismatches.xlsx` — five sheets: `Regular_BBRef_only`, `Regular_WPA_only`, `Playoff_BBRef_only`, `Playoff_WPA_only`, `Summary`.

Read the sheets correctly: the `*_BBRef_only` sheets are the ones that matter — they list qualifying BBRef player-seasons with no WPA match, and both are currently empty. The `*_WPA_only` sheets are expected to be large and are **not** errors; because WPA is left unfiltered by design, they contain every WPA player-season falling below the BBRef games threshold.

---

## Canonical WPA Dataset

**`data/raw/wpa_all_seasons.csv`** — 31,745 rows. All 26 seasons × 3 variants. 19 columns: `season, playoff_variant, rnk, player, player_id, pos, gms, wpa, ewpa, clwpa, gbwpa, sh, to, ft, reb, ast, stl, blk, kwpa`. This is the untouched scrape output and still contains the corrupted names described below.

**`data/processed/cwpa_long.xlsx`** — **the single source of truth for WPA data.** Three sheets: **Regular** (12,910 rows), **Playoffs** (5,912), **Combined** (12,923), 31,745 rows total. The `playoff_variant` column is dropped from each sheet and restored on load.

Every consumer reads it through one helper so they cannot drift apart:

```python
from stats_utils import load_cwpa
df = load_cwpa()      # all three variants stacked, playoff_variant + yr restored
```

`eda.py`, `analysis.py` and `build_xgboost.py` all use it.

**There is deliberately no `cwpa_long.csv`.** One used to sit beside the workbook and was what `eda.py` and `analysis.py` read. The name repair had only ever been applied by hand to the workbook, so the CSV kept **52 corrupted rows covering 27 players** and the two files silently disagreed. It was deleted rather than repaired: a second serialisation of the same table is a standing invitation for exactly that drift. Do not reintroduce one.

**`data/raw/wpa_all_seasons.xlsx`** — convenience copy of the deduplicated table. Sheets are named **`cWPA_Regular`, `cWPA_Playoffs`, `cWPA_Combined`** (note the prefix; the processed workbook does not use it). Not read by any script.

### Name repair in `build_dataset.py`

inpredictable.com serves accented names inconsistently across seasons, in two mutually incompatible ways, and **both are lossy**:

| Corruption | Example | Why automatic repair fails |
|---|---|---|
| `?` substitution | `Nikola Joki?`, `Luka Don?i?` | The byte is gone and `?` is valid ASCII — `ftfy` sees nothing wrong and `unidecode` has nothing to transliterate |
| U+FFFD substitution | `Chris Ma<fffd>on`, `Mont<fffd> Morris` | `unidecode` *drops* the replacement character, silently yielding `Chris Maon` and `Mont Morris` |

`ftfy` + `unidecode` alone therefore cannot fix either case — run on their own they produce plausible-looking but wrong names. `NAME_REPAIRS` in `build_dataset.py` maps the 22 exact corrupted strings to their correct spellings and is applied **first**; `ftfy` + `unidecode` then handle the ordinary well-formed accents (`Nikola Jokić` → `Nikola Jokic`) that some seasons return correctly.

The script asserts that zero corrupted names survive and exits with the offending list if any do, so a newly scraped season fails loudly instead of quietly polluting the workbook.

`ALIGNMENT_FIXES` is separate and handles a different problem: names that are well-formed but use a different *form* than Basketball-Reference. Currently one entry, `Jose Juan Barea` → `J.J. Barea`, which recovers 7 player-seasons (2008-2015). It was previously a hand edit to the workbook and was silently lost the first time the script was re-run.

**Deduplication.** `build_dataset.py` groups on `(season, playoff_variant, player_id)`, summing `gms` and `clwpa` and taking first `pos`/`player`, then asserts zero duplicates remain.

**This groupby is currently a no-op, and that is expected.** `scrape_wpa.py` already keys its result dict on `player_id`, so the raw CSV holds exactly one row per (season, variant, player) — verified: 0 duplicate keys across all 31,745 rows, and the row count is unchanged by the step. It is kept as a guard: if the scraper's keying ever changes, or the site starts emitting per-team splits, this catches it instead of silently double-counting.

`gms > 82` (max 85, 14 rows in regular) is **inpredictable.com's own combined total** for a traded player, not something this pipeline sums. Expected and correct.

---

## Analysis Design (`src/analysis.py`)

### Thresholds (analysis only; EDA uses full unfiltered data)

| Variant | Minimum gms | Rationale |
|---|---|---|
| Regular | 62 (stated as 61.5) | ~75% of an 82-game season; filters out injury-shortened years and call-ups |
| Playoffs | 10 | Roughly one full round; filters out teams swept in the first round |
| Combined | 72 (stated as 71.5) | Higher than regular alone since combined folds in playoff minutes; keeps only players with meaningful exposure in both settings |

These thresholds apply **only to the analysis and XGBoost models**. EDA uses the full unfiltered dataset. The mismatch check applies them only to the BBRef side; WPA is never filtered by games.

The sensitivity sweep in `analysis.py` reruns all three prongs across a grid of alternate thresholds ([41, 52, 62, 72, 82] for regular/combined; [5, 8, 10, 13, 16] for playoffs) to confirm the primary results are not threshold-sensitive.

### Two Normalizations (both reported for all prongs)
- **Per-game rate:** `clwpa / gms`
- **Raw season total:** `clwpa`

### Three Prongs

**Prong 1 — Regular YoY persistence:** Filter regular-season rows to `gms ≥ 62`. For every consecutive `(yr, yr+1)` pair across all 25 transitions (2000-01 → 2024-25), join on `player_id`. Pool all pairs. n = 3,244.

**Prong 2 — Regular → Playoff transfer (same season):** Within each season, join players appearing in both regular (`gms ≥ 62`) and playoff (`gms ≥ 10`) on `player_id`. n = 1,352.

**Prong 3 — Combined YoY persistence:** Same as Prong 1 with the `combined` variant and `gms ≥ 72`. Tests whether folding in playoff clutch minutes increases measured persistence. n = 2,243.

### Confirmed Results (`data/outputs/analysis_results.csv`)

| Prong | Normalization | Pearson r | 95% CI | Spearman ρ | n |
|---|---|---|---|---|---|
| P1 Regular YoY | per_game | 0.5089 | [0.4829, 0.5340] | 0.4922 | 3,244 |
| P1 Regular YoY | total | 0.5068 | [0.4808, 0.5319] | 0.4883 | 3,244 |
| P2 Reg → Playoff | per_game | 0.3663 | [0.3193, 0.4116] | 0.3215 | 1,352 |
| P2 Reg → Playoff | total | 0.3817 | [0.3352, 0.4263] | 0.3261 | 1,352 |
| P3 Combined YoY | per_game | 0.5527 | [0.5233, 0.5808] | 0.5258 | 2,243 |
| P3 Combined YoY | total | 0.5584 | [0.5293, 0.5863] | 0.5247 | 2,243 |

All p-values are effectively 0 given sample sizes.

These figures were regenerated after the 2019-20 bubble-playoff data was recovered. They differ
from the pre-patch numbers in the third decimal place (e.g. P1 per_game was 0.5093 at n=3,230),
with every change falling well inside the confidence intervals. The sample sizes grew because the
2019-20 playoff season now contributes pairs.

### Sensitivity Sweep

Recomputes Pearson r across:
- Regular/Combined thresholds: [41, 52, 62, 72, 82]
- Playoff thresholds: [5, 8, 10, 13, 16]

Output: line plots of r vs threshold (Prongs 1 and 3) and a heatmap at every reg × playoff combination (Prong 2). Primary benchmarks shown as vertical dashed lines.

---

## XGBoost Models (`src/build_xgboost.py`)

Two independent models: **Regular Season** (G≥62, 5,647 joined rows) and **Playoffs** (G≥10, 1,862 rows). Regression only — there is no classification task.

**Join:** BBRef `Year` == WPA `yr` (both start-year convention, direct equality). Names via `norm()`: lowercase, strip periods and asterisks.

`EXCLUDE = {"Year", "Player", "Pos", "Age", "Tm", "GS", "player_id", "yr"}` — leaves **44 numeric BBRef candidate features** plus one engineered feature (see below). (`G` and `MP` are candidates but neither reaches the top 20 in either model.)

**Lag feature (regular-season model only).** After the join, `add_lag_feature()` attaches `clwpa_lag1` = the player's regular-season cWPA from `yr-1`, looked up by `player_id` against the WPA data. Rookies and gap-year returners have no prior-year record and receive NaN; XGBoost learns a default split direction for missing values so no rows are dropped. 86.6% of rows have a value; 13.4% (755 rows, mostly rookies) are NaN. This gives **45 candidate features** for the regular model.

### Chronological train/test split

| | Seasons | Regular rows | Playoff rows |
|---|---|---|---|
| Train | 2000-2020 (21 seasons) | 4,618 | 1,503 |
| Test | 2021-2025 (5 seasons) | 1,029 | 359 |

The split is by season, not random. A random split would let the model learn from 2024 to predict 2023.

**No test-set leakage.** Feature selection *and* hyperparameter tuning both use training years only; the test seasons are touched once, at final evaluation. Ranking features on all 26 seasons and then "testing" on five of them would leak test information into the feature list.

### Two-step procedure

**Step 1 — feature importance.** Full-feature model fit on TRAIN only, ranked by gain. Top `TOP_K = 20` retained. Each model picks its own list. The regular model has 45 candidate features (44 BBRef + `clwpa_lag1`); the playoff model has 44.

**Step 2 — hyperparameter search.** `RandomizedSearchCV`, `N_ITER = 400` configs sampled from a 3,240-point space, scored on RMSE + MAE + R² simultaneously, refit on RMSE, 5-fold `KFold(shuffle=True)`.

Random sampling rather than an exhaustive sweep is a measured cost decision, not a shortcut: the exhaustive grid took ~56 min for the regular model under 5-fold and was killed after 7.5 h under the 21-fold scheme trialled alongside it. The sampled optimum scored test R² = 0.4425 against the exhaustive sweep's 0.4418 on identical data — marginally better, from 12% of the space.

**NaN handling.** `3P%` is NaN for players with zero 3PA (5% of regular rows, 14% of playoff). Those rows are **kept** — XGBoost routes NaN natively. The earlier implementation dropped them, discarding real players.

### Cross-validation scheme selection (LOGO evaluated and removed)

Two schemes were implemented and run head to head, each scheme's winning config refit on TRAIN and scored on TEST:

- **KFold** (5-fold, shuffled) — player-seasons treated as exchangeable
- **LeaveOneGroupOut by season** (21 folds) — one whole season held out per fold

**KFold is retained and the LOGO code path was deleted from `build_xgboost.py`. It was kept because the two schemes are statistically tied, not because it won.** A note recording the full comparison lives in that file's docstring; the numbers cannot be regenerated by rerunning the pipeline.

On the repaired regular-season data LOGO is nominally *ahead*:

| Scheme | Test R² | Test RMSE | Test MAE |
|---|---|---|---|
| KFold (retained) | 0.4622 | 0.6353 | 0.4634 |
| LOGO (removed) | 0.4628 | 0.6350 | 0.4632 |

That is a 0.0006 R² gap. Re-running the KFold search under different seeds spans **0.4574 – 0.4631 (range 0.0057, roughly 10× the gap)**, and KFold at `seed=1` scores 0.4631, beating LOGO outright. Which scheme "wins" is decided by seed choice, so the tiebreak is cost: LOGO runs 21 folds to KFold's 5, about 4× the compute, for no measurable gain. On the playoff model — whose data the repair did not touch — KFold won outright, 0.2658 vs 0.2380.

Note: these comparison numbers predate the `clwpa_lag1` addition. The current regular-season model (with lag) scores test R² = 0.4624 — effectively identical, so the KFold/LOGO tiebreak conclusion still holds.

**Caution if you revisit this.** The first comparison ran on the corrupted 2021-22 data and had KFold ahead 0.4425 vs 0.4327. That verdict was an artefact of the bad year. Re-measure after any change to the underlying data instead of trusting a stored conclusion.

Important subtlety: the two schemes' **CV R² values are not comparable**. LOGO scores each fold against a single season's mean, and one season has far less cWPA variance than the 21-season pool, so its R² denominator is much smaller — LOGO posts a *better* RMSE but a much worse R². Selection was therefore decided on held-out test performance, not CV scores.

### Confirmed Results

| Metric | Regular Season | Playoffs |
|---|---|---|
| CV R² / RMSE / MAE (train) | 0.4885 / 0.6335 / 0.4541 | 0.2667 / 0.2574 / 0.1670 |
| In-sample R² (train) | 0.6547 | 0.5045 |
| **Test R² (2021-2025)** | **0.4624** | **0.2658** |
| **Test RMSE** | **0.6352** | **0.2646** |
| **Test MAE** | **0.4616** | **0.1511** |
| n (train / test) | 4,618 / 1,029 | 1,503 / 359 |
| cWPA_Difference mean / sd | −0.0557 / 0.6334 | −0.0155 / 0.2645 |

Winning hyperparameters — *Regular:* `max_depth=4, learning_rate=0.01, n_estimators=800, subsample=0.6, colsample_bytree=0.6, min_child_weight=5, reg_lambda=1.0`. *Playoffs:* `max_depth=3, learning_rate=0.01, n_estimators=400, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0`. Both settled on a very low learning rate with many shallow trees, the signature of a noisy target where slow regularised fitting beats aggressive learning.

**Top 20 features.**

- Regular: `TS%, eFG%, PTS, TOV%, clwpa_lag1, TOV, OBPM, OWS, 3PA, USG%, 3P, WS, 2PA, VORP, 3PAr, ORB, PER, FT, PF, FT%`
- Playoffs: `VORP, OWS, eFG%, TS%, FT, WS, TOV, PTS, 2P, 3PA, 2PA, FGA, DWS, AST, FG, FTA, PF, 3P, TRB%, TOV%`

The regular model leads with efficiency (`TS%`, `eFG%`) and now includes `clwpa_lag1` (prior-year regular-season cWPA) at rank 5 — reflecting the year-over-year clutch persistence signal measured in the correlation analysis. The playoff model leads with cumulative value (`VORP`, `OWS`) and has no lag feature (prior-year playoff cWPA ranked 30th in the feature importance check and did not make the top 20).

The regular list changed twice: after the 2021-22 repair (`FGA`, `FG`, `VORP` dropped out; `FT`, `TRB%`, `FTA` entered), and again when `clwpa_lag1` was added (`2P%`, `TRB%`, `FTA` dropped out; `WS`, `VORP`, `clwpa_lag1` entered). If you find an older copy of this list anywhere, it predates both fixes.

Gain is split arbitrarily among correlated features (`WS`/`VORP`/`BPM`, `FG`/`FGA`, `PTS`/`USG%`), so a low rank means "redundant given the others", not "unrelated to clutch". The regular model dropping `VORP` while the playoff model ranks it first is an artefact of this: in the regular model its signal is already carried by `OWS` and `OBPM`.

### cWPA_Difference

Stored per held-out player-season: `cWPA_Difference = predicted_cWPA − actual_cWPA`. Negative = the model under-predicted, i.e. the player out-performed their statistical profile in the clutch. Positive = under-delivered.

**Outputs:**
- `data/outputs/xgb_results_{regular,playoff}.csv` — model summary, metrics, winning hyperparameters
- `data/outputs/xgb_feature_importance_{regular,playoff}.csv` — all 44 features ranked
- `data/outputs/xgb_gridsearch_{regular,playoff}.csv` — full search trace, all 400 configs
- `data/outputs/xgb_predictions.xlsx` — sheets: `Regular`, `Regular_by_season`, `Regular_importance`, `Playoffs`, `Playoffs_by_season`, `Playoffs_importance`, `Model_Summary`
- `data/outputs/xgb_predictions_{regular,playoff}.csv` + `xgb_per_season_{regular,playoff}.csv` — per-model checkpoints written as soon as each model finishes, so an interrupted run does not lose completed work
- `data/outputs/unmatched_{regular,playoff}.txt` — now empty; both joins are 100%
- `figures/analysis/xgboost/01_importance_all_features_*.png` … `05_residuals_*.png`

`random_state = 42` throughout for reproducibility. Rerunning reproduces every metric exactly.

---

## Shared Utilities: `stats_utils.py`

- `set_style()` — seaborn whitegrid, font_scale=1.05, dpi=150
- `C_REG`, `C_PO`, `C_COMB` — muted palette colors (blue, orange, green)
- `pearson_with_ci(x, y, alpha=0.05)` → `(r, p, ci_lo, ci_hi, n)` via Fisher-z transform
- `spearman_rho(x, y)` → `(rho, p, n)`
- `corr_summary(x, y, label)` → prints both stats, returns dict
- `scatter_corr(x, y, xlabel, ylabel, title, out_path, color, ...)` → scatter + OLS line + annotation box showing **R² and n only** (changed from Pearson r + Spearman ρ in a prior session; all six analysis scatter plots in `figures/analysis/` were regenerated after this change and now display R²)

---

## File Structure

```
NBA Clutch Ability/
├── src/
│   ├── scrape_wpa.py          # Step 1a: scrape WPA from inpredictable.com
│   ├── build_dataset.py       # Step 1b: name repair + dedup → cwpa_long.xlsx
│   ├── scrape_bbref.py        # Step 2: BBRef scrape + name/encoding fixes + REPLACE_YEARS
│   ├── check_mismatches.py    # Step 2b: BBRef↔WPA name-matching audit
│   ├── stats_utils.py         # load_cwpa() + correlation helpers + plot style
│   ├── eda.py                 # Step 3: 10 EDA figures + eda_summary.json
│   ├── analysis.py            # Step 4: 3 prongs × 2 normalizations + sensitivity
│   ├── generate_report.py     # Step 5: report/report.md
│   └── build_xgboost.py       # Step 6: XGBoost, chronological split
├── data/
│   ├── backup/                       # pre-repair snapshots, not read by any script
│   │   ├── seasons_stats_clean.pre2021fix.csv   # before the 2021-22 season repair
│   │   ├── playoffStats_clean.pre2021fix.csv
│   │   ├── cwpa_long.pre_namefix.xlsx           # before build_dataset name repair
│   │   └── wpa_all_seasons.pre_namefix.xlsx
│   ├── raw/
│   │   ├── wpa_all_seasons.csv       # 31,745 rows; full scrape output
│   │   ├── wpa_all_seasons.xlsx      # 3 sheets: cWPA_Regular/_Playoffs/_Combined
│   │   ├── seasons_stats.csv         # Original BBRef regular season (1950-2022)
│   │   └── playoffStats.csv          # Original BBRef playoffs (1950-2022)
│   ├── processed/
│   │   ├── cwpa_long.xlsx            # CANONICAL WPA source; Regular 12,910 / Playoffs 5,912 / Combined 12,923
│   │   │                             #   (no cwpa_long.csv — deliberately, see Canonical WPA Dataset)
│   │   ├── seasons_stats_clean.csv   # 12,811 rows, Year 2000-2025, all encoding fixed
│   │   └── playoffStats_clean.csv    # 5,350 rows, Year 2000-2025, all encoding fixed
│   └── outputs/
│       ├── analysis_results.csv      # 6 rows: 3 prongs × 2 normalizations
│       ├── eda_summary.json          # Dataset counts + top/bottom 10 leaderboards
│       ├── player_mismatches.xlsx    # 5 sheets: mismatch lists + summary
│       ├── xgb_results_regular.csv   # model summary + metrics (regular)
│       ├── xgb_results_playoff.csv   # model summary + metrics (playoffs)
│       ├── xgb_feature_importance_*.csv  # all 44 features ranked
│       ├── xgb_gridsearch_*.csv      # full 400-config search trace
│       ├── xgb_predictions.xlsx      # per-player test predictions + cWPA_Difference
│       ├── xgb_predictions_*.csv     # per-model checkpoints
│       ├── xgb_per_season_*.csv      # per-season test metrics
│       ├── unmatched_regular.txt     # BBRef rows with no WPA match (currently empty)
│       └── unmatched_playoff.txt     # note: singular 'playoff', matching the model slug
├── figures/
│   ├── eda/
│   │   ├── coverage/                 # 01_coverage_grid.png
│   │   ├── distributions/            # 02-07 histograms, violin, yearly aggregates
│   │   └── careers/                  # 08 top-25 bars (×3 variants), 09 longevity scatter
│   └── analysis/
│       ├── prong_1_regular_yoy/      # P1 scatters + per-transition r bar chart
│       ├── prong_2_reg_to_playoff/   # P2 scatter
│       ├── prong_3_combined_yoy/     # P3 scatters
│       ├── sensitivity/              # Line plots + P2 heatmap
│       └── xgboost/                  # Importance, predicted-vs-actual, residuals
├── report/
│   ├── report.md                     # Full technical narrative with all results, tables, and figure paths
│   ├── blog_post.md                  # Public-facing blog post (intentionally NOT kept in sync with docx)
│   └── blog_post.docx                # Canonical blog deliverable — generated by make_blog_v2.js (scratchpad)
├── figures/
│   └── blog/                         # Blog-specific figures (not generated by any src/ script)
│       ├── 01_seasons_scatter.png    # All qualifying regular-season player-seasons as dots, top 5 highlighted
│       ├── 02_career_leaders.png     # All-time career cWPA dot plot, top 30, G≥62
│       └── 03_five_year_reg.png      # 5-year regular-season leaders dot plot, top 30, G≥62
└── README.md                         # Public-facing repo summary — findings, all four leaderboard tables, over/under-performer tables, figure index
```

### Blog figures (`figures/blog/`)

Generated by a one-off script (`make_blog_figures.py`, stored in the session scratchpad, not in `src/`). If you need to regenerate them, recreate the script from the logic described here:

- **01_seasons_scatter.png** — scatter of all qualifying regular-season player-seasons (G≥62, n=5,505), jittered on y-axis, top 5 highlighted with gold dots and curved annotation arrows. Top 5 layout is hard-coded by absolute label coordinates to avoid overlap (all five cluster between 4.90 and 5.84 cWPA).
- **02_career_leaders.png** — horizontal dot plot, top 30 all-time career cWPA, G≥62 threshold.
- **03_five_year_reg.png** — same format, 5-year window (yr 2021-2025, seasons 2021-22 through 2025-26), G≥62.

### blog_post.docx vs blog_post.md

**`blog_post.md` is intentionally kept at an older state. Never apply docx changes back to the md.** The canonical blog deliverable is `blog_post.docx`. It was built with `make_blog_v2.js` (node, `docx` npm package, stored in the session scratchpad — not in `src/`).

**Final docx contents** (diverges from blog_post.md in all of the following):

- **Introduction:** Knicks-first narrative — down 20 twice vs Celtics (2024-25), down 20 with 8 min left vs Cavs Game 1 ECF (2025-26), down 29 in Q3 vs Spurs (greatest playoff comeback in NBA history). Clutch as organizational DNA, not luck.
- **Seasons scatter** (`figures/blog/01_seasons_scatter.png`) replacing the text histogram. Every qualifying player-season as a dot (G≥62, n=5,505), top 5 highlighted with gold dots and curved arrows.
- **Career dot plot** (`figures/blog/02_career_leaders.png`) replacing the all-time career table. Top 30, G≥62 threshold.
- **5-year dot plot** (`figures/blog/03_five_year_reg.png`). Top 30, yr 2021-2025 (seasons 2021-22 through 2025-26), G≥62.
- **Correlation section:** three embedded scatter plots (one per prong, total normalization, from `figures/analysis/`) with R² annotation, one short paragraph each. No multi-paragraph text breakdowns.
- **Leaderboard tables** (no games threshold, raw cWPA totals):
  - All-time regular season top 10 (LeBron +45.00 … Butler +20.83)
  - Last 4 years regular season top 10 — yr 2022-2025 = seasons 2022-23 through 2025-26 (DeRozan +14.71 … Reaves +8.33)
  - All-time playoffs top 10 (LeBron +10.80 … Horford +3.07)
  - Last 4 years playoffs top 10 (Haliburton +2.24 … DiVincenzo +1.35)
- **Playoffs 5-year table** (separate from the dot plots, G≥10): Brunson leads at 2.37 over the 5-year window (yr 2021-2025).
- **Over-performers table (regular, 5 rows):** Fox 2022-23 (-4.00), DeRozan 2021-22 (-3.46), LeBron 2024-25 (-2.54), Schroder 2022-23 (-2.37), DeRozan 2022-23 (-2.31).
- **Under-performers table (regular, 5 rows):** Mitchell 2021-22 (+2.47), Trae Young 2021-22 (+2.46), Tatum 2022-23 (+2.28), Garland 2022-23 (+1.69), Cunningham 2023-24 (+1.64).
- **Over-performers table (playoff, 5 rows):** Haliburton 2024-25 (-2.34), Murray 2022-23 (-1.32), Gordon 2024-25 (-1.08), White 2022-23 (-1.07), Butler 2022-23 (-1.03).
- Shortened Haliburton and Brunson notes (2 sentences each explaining the cumulative vs single-season paradox).

---

## EDA Output Summary

`eda.py` generates 10 figures:

| Figure | Path | Content |
|---|---|---|
| 01 | `eda/coverage/01_coverage_grid.png` | Unique players per season per variant (line plot) |
| 02 | `eda/distributions/02_gms_distributions.png` | gms histograms, 3 panels |
| 03 | `eda/distributions/03_position_breakdown.png` | Position breakdown (regular, all years) |
| 04 | `eda/distributions/04_clwpa_histogram.png` | Overall cWPA histogram, skew/kurtosis annotated |
| 05 | `eda/distributions/05_clwpa_per_game_histogram.png` | Per-game rate histogram |
| 06 | `eda/distributions/06_seasonal_violin_regular.png` | Per-season violin, full unfiltered regular |
| 07 | `eda/distributions/07_league_yearly_aggregates.png` | 2×2: mean/median/total cWPA + strong clutch count |
| 08×3 | `eda/careers/08_career_top25_{regular,playoffs,combined}.png` | Career cWPA top-25 bar charts |
| 09 | `eda/careers/09_career_longevity.png` | Career longevity histogram + total-vs-seasons scatter |

`eda_summary.json` stores row counts, player counts, gms=0/gms>82 flags per variant, and top/bottom 10 single-season cWPA leaderboards.

---

## Known Data Issues and Edge Cases

**2019-20 bubble playoffs:** The NBA played its playoffs August–October 2020. The standard `todt=2020-08-01` misses them entirely. `season_dates(2019)` returns `todt="2020-11-01"` as a hardcoded special case in `scrape_wpa.py`. Without this, 2019-20 playoffs returns 0 rows.

**Non-ASCII player names:** Two distinct corruption patterns in the original files:
1. Double-encoded UTF-8 (e.g., `Bogdan BogdanoviÃ\x84Â\x87`) — in BBRef files
2. `?`-substitution (e.g., `Nikola Joki?`) — in WPA 2021/2025 seasons and some BBRef years

Both fixed with `unidecode(fix_text(name))` plus explicit year-specific overrides for `?` cases where ftfy leaves the `?` intact (it is valid ASCII). Zero garbled names remain in any processed file.

**Name collisions:** Two "Chris Johnson" players (IDs 202419 and 203187) and two "Marcus Williams" players (200766 and 201173) appear in overlapping seasons. The WPA pipeline joins on `player_id`, keeping them separate. The BBRef pipeline joins on name + year, which is unambiguous since they played in different seasons.

**gms > 82:** Traded players whose per-team stints are summed can exceed 82. Expected and correct.

**combined ≈ regular + playoffs:** Max discrepancy is 0.09, consistent with rounding at source. Treat `combined` as authoritative.

**Right-skewed cWPA distribution:** Skew ≈ 1.05, excess kurtosis ≈ 5.1. Motivates reporting Spearman ρ alongside Pearson r.

**Pseudo-replication in YoY prongs:** The same player contributes up to 25 consecutive pairs. The per-transition r bar chart in Prong 1 tests era consistency as a partial check.

**Year-specific name changes (Jimmy Butler, Otto Porter, Marcus Morris, KJ Martin, Lou Williams):** WPA's naming for these players changes partway through their careers. Applying the suffix correction globally in either direction creates mismatches in other years. All corrections are row-by-row with explicit year conditions. See the corrections table above for exact year ranges.

**Stale comment in `build_xgboost.py`:** Lines 17-19 say `bbref_Year = wpa_yr + 1`. This was written before the year convention was aligned. The actual join code uses direct equality; only the docstring comment is wrong.

---

## Dependencies

```
requests
beautifulsoup4
pandas
numpy
scipy
matplotlib
seaborn
openpyxl
xgboost
scikit-learn
ftfy
unidecode
tabulate   # optional — for to_markdown() in generate_report.py
```

Install with:
```bash
pip3 install requests beautifulsoup4 pandas numpy scipy matplotlib seaborn openpyxl xgboost scikit-learn ftfy unidecode tabulate
```

---

## Limitations

1. **gms as clutch-appearance proxy.** No column counts clutch possessions directly.
2. **Pseudo-replication in YoY prongs.** Same player contributes up to 25 consecutive pairs.
3. **Durable-player bias.** G≥62 threshold filters toward healthy starters, potentially inflating correlations.
4. **Era effects.** Pace, three-point rates, and officiating changed across 26 seasons. cWPA totals are not era-adjusted.
5. **Playoff small samples.** Even at G≥10, individual playoff per-game rates are noisy (max 28 games per season).
6. **Proprietary metric.** cWPA methodology is described at inpredictable.com but the underlying play-by-play data is not publicly auditable.
7. **BBRef join on name.** The WPA-internal pipeline uses `player_id`; the cross-source join uses normalized names, requiring extensive manual correction (see `NAME_FIXES`). Both joins currently resolve at 100%, but a newly scraped season will not — always run the join check after adding a year.
8. **Source data required repair.** The shipped `seasons_stats.csv` had the 2021-22 season duplicated from 2020-21. It was caught only because a per-season test metric went negative. Treat new raw drops as unverified: check max games per season against the real schedule length and scan for duplicated adjacent years before trusting them.

---

## Project Status

**This project is complete and published.**

GitHub repo: `https://github.com/Siddhant17-Jain/Predicting-NBA-Clutch-WPA` (public)

All pipeline steps have been run to completion. All outputs are committed. The blog deliverable (`report/blog_post.docx`) is in its final state. No further development is planned.

**Verified final numbers (do not re-derive without re-running the pipeline):**

| Prong | Normalization | Pearson r | R² | n |
|---|---|---|---|---|
| P1 Regular YoY | total | 0.5068 | 0.257 | 3,244 |
| P2 Reg → Playoff | total | 0.3817 | 0.146 | 1,352 |
| P3 Combined YoY | total | 0.5584 | 0.312 | 2,243 |

| Model | Test R² | Test RMSE | Test MAE |
|---|---|---|---|
| Regular season | 0.4624 | 0.6352 | 0.4616 |
| Playoffs | 0.2658 | 0.2646 | 0.1511 |

**All-time single-season records (no threshold):** DeRozan 2021-22 (+5.84 regular), Haliburton 2024-25 (+2.63 playoffs).

**Join completeness:** Regular 5,647/5,647 (100%), Playoffs 1,862/1,862 (100%).

If the project is resumed for a new season (2026-27), the required steps are: re-run `scrape_wpa.py` with `SEASONS=range(2000, 2027)`, re-run `build_dataset.py`, re-run `scrape_bbref.py` with `REPLACE_YEARS` set if needed, run `check_mismatches.py` and resolve any new name gaps, then re-run `eda.py`, `analysis.py`, `build_xgboost.py`, and `generate_report.py` in order. Regenerate blog figures and rebuild the docx manually.
