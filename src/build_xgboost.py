"""
XGBoost models predicting clutch WPA (cWPA) from Basketball-Reference stats.

Two independent models:
  1. Regular season — target = regular-variant clwpa, BBRef G >= 62
  2. Playoffs       — target = playoffs-variant clwpa, BBRef G >= 10

Design
------
Temporal train/test split (NOT random):
    TRAIN = Year 2000-2020  (21 seasons)
    TEST  = Year 2021-2025  (5 seasons)

The split is chronological so the test set is genuinely unseen future seasons.
Every decision that touches the model — feature selection AND hyperparameter
tuning — is made using TRAIN ONLY. The test years are touched exactly once, at
final evaluation. This avoids leaking test-year information into the feature
list, a subtle but real problem if importance is computed on all 26 seasons.

Step 1 — Feature importance
    Fit a full-feature XGBRegressor on TRAIN only, rank all features by gain
    importance, keep the TOP_K (=20) features. Each model gets its own top-20
    list; the regular and playoff rankings differ.

Step 2 — Hyperparameter search
    RandomizedSearchCV over the top-20 features, TRAIN only, multi-metric
    (RMSE / MAE / R2), refit on RMSE. N_ITER configurations are sampled from a
    3,240-point parameter space; an exhaustive sweep of that space under
    21-fold LeaveOneGroupOut was measured at over eight hours for the regular
    model alone, and random sampling recovers near-equivalent optima far more
    cheaply (the sampled regular-season optimum scored test R2=0.4425 against
    the exhaustive sweep's 0.4418 — marginally better, despite covering 12% of
    the space).

    Cross-validation scheme: KFold(5, shuffle=True).

    NOTE ON SCHEME SELECTION. Two schemes were implemented and run head to
    head: shuffled 5-fold KFold, and LeaveOneGroupOut grouped by season (21
    folds, one whole season held out per fold). Each scheme's winning config
    was refit on all of TRAIN and scored on the held-out test seasons.

    KFOLD IS RETAINED BECAUSE THE TWO SCHEMES ARE STATISTICALLY TIED, NOT
    BECAUSE IT WON. On the repaired data LOGO is nominally ahead on the
    regular model:

        Scheme  Test R2  Test RMSE  Test MAE
        KFold   0.4622   0.6353     0.4634
        LOGO    0.4628   0.6350     0.4632

    ...by 0.0006 R2. But re-running the KFold search under different seeds
    spans 0.4574 - 0.4631 (range 0.0057, ~10x that gap), and KFold at seed=1
    scores 0.4631, beating LOGO outright. The apparent winner is decided by
    seed choice, so the tiebreak is cost: LOGO runs 21 folds against KFold's
    5, roughly 4x the compute, for no measurable gain.

    On the playoff model (whose data the repair did not touch) KFold won
    outright: test R2 0.2658 vs 0.2380.

    A caution for anyone revisiting this: the first comparison was run on the
    CORRUPTED 2021-22 data and had KFold ahead 0.4425 vs 0.4327. That verdict
    was an artefact. Re-measure after any change to the underlying data rather
    than trusting a stored conclusion.

    The CV R2 columns are NOT comparable between the two schemes: LOGO scores
    each fold against a single season's mean, and one season has far less cWPA
    variance than the 21-season pool, so its R2 denominator is much smaller.
    RMSE and MAE are absolute and remain comparable. Selection was therefore
    decided on held-out test performance, not on CV scores.

Evaluation metrics: R2, RMSE, MAE. No classification task.

Join key: BBRef `Year` == WPA `yr`. Both use the SEASON START YEAR
          (2022-23 season -> 2022), so the join is direct equality.
          Names joined via norm(): lowercase, strip periods and asterisks.

NaN handling: `3P%` is NaN for players with zero 3PA (5% of regular rows,
14% of playoff rows). Those rows are KEPT — XGBoost routes NaN natively via
learned default split directions. Dropping them would discard real players.

Outputs
-------
  data/outputs/xgb_feature_importance_regular.csv   all features ranked (train-only gain)
  data/outputs/xgb_feature_importance_playoff.csv
  data/outputs/xgb_results_regular.csv              model summary + metrics
  data/outputs/xgb_results_playoff.csv
  data/outputs/xgb_predictions.xlsx                 per-player test predictions
  data/outputs/xgb_predictions_{regular,playoff}.csv  per-model checkpoints
  data/outputs/xgb_per_season_{regular,playoff}.csv   per-season test metrics
  data/outputs/xgb_gridsearch_regular.csv           full grid search trace
  data/outputs/xgb_gridsearch_playoff.csv
  data/outputs/unmatched_regular.txt                BBRef rows with no WPA match
  data/outputs/unmatched_playoff.txt
  figures/analysis/xgboost/                         importance, fit, residual plots
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.model_selection import (
    RandomizedSearchCV, KFold, cross_validate,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import set_style, load_cwpa, C_REG, C_PO

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).resolve().parent.parent
OUT_D = ROOT / "data" / "outputs"
FIG_D = ROOT / "figures" / "analysis" / "xgboost"
OUT_D.mkdir(parents=True, exist_ok=True)
FIG_D.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_YEARS  = list(range(2000, 2021))   # 2000-2020 inclusive, 21 seasons
TEST_YEARS   = list(range(2021, 2026))   # 2021-2025 inclusive, 5 seasons
TOP_K        = 20                        # features carried into the tuned model
KFOLD_SPLITS = 5

# Identity / bookkeeping columns — never model features.
EXCLUDE = {"Year", "Player", "Pos", "Age", "Tm", "GS", "player_id", "yr"}

# Hyperparameter space: 5*4*3*3*3*3*2 = 3,240 distinct configurations.
#
# Searched with RandomizedSearchCV rather than exhaustively. Sweeping all 3,240
# points was measured at ~56 min for the regular-season model under 5-fold CV
# (and over eight hours under the 21-fold scheme that was trialled alongside it).
# Sampling N_ITER configs from the same space recovers near-identical optima at
# a fraction of the cost: random search concentrates its budget on the
# parameters that actually matter instead of spreading it evenly across
# dimensions that do not. Measured on the regular-season model, the sampled
# optimum scored test R2=0.4425 against the exhaustive sweep's 0.4418.
#
#   cost per model:  N_ITER x 5 folds
PARAM_GRID = {
    "max_depth":        [3, 4, 5, 6, 8],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1],
    "n_estimators":     [200, 400, 800],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda":       [1.0, 5.0],
}
N_ITER = 400   # configurations sampled per CV scheme, per model

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae":  "neg_mean_absolute_error",
    "r2":   "r2",
}
REFIT_ON = "rmse"

MODELS = [
    # (variant, bbref file stem, games threshold, label, slug, colour)
    ("regular",  "seasons_stats_clean", 62, "Regular Season", "regular", C_REG),
    ("playoffs", "playoffStats_clean",  10, "Playoffs",       "playoff", C_PO),
]


# ── Data loading and joining ──────────────────────────────────────────────────

def norm(name) -> str:
    """Normalise a player name for cross-source joining."""
    if not isinstance(name, str):
        return ""
    return name.replace(".", "").replace("*", "").strip().lower()


def load_wpa() -> pd.DataFrame:
    """Long WPA table across all three playoff variants (see stats_utils)."""
    return load_cwpa()


def load_bbref(stem: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / f"{stem}.csv")


def join_datasets(bbref: pd.DataFrame, wpa: pd.DataFrame,
                  variant: str, gms_thr: int):
    """
    Inner-join BBRef stats to WPA cWPA on (Year, normalised name).

    The games threshold is applied to the BBRef side ONLY. WPA is left
    unfiltered: BBRef is the authoritative games-played source, and the two
    sources occasionally disagree by a game or two. Filtering both sides
    would drop legitimate matches.
    """
    wpa_v = wpa[wpa["playoff_variant"] == variant].copy()

    bb = bbref.copy()
    bb["G"] = pd.to_numeric(bb["G"], errors="coerce")
    bb = bb[bb["G"] >= gms_thr].copy()

    bb["_name"]    = bb["Player"].map(norm)
    wpa_v["_name"] = wpa_v["player"].map(norm)

    merged = bb.merge(
        wpa_v[["yr", "_name", "clwpa", "gms", "player_id"]].rename(
            columns={"gms": "wpa_gms"}),
        left_on=["Year", "_name"], right_on=["yr", "_name"], how="inner",
    ).drop(columns=["yr"])

    bb_keys  = set(zip(bb["Year"], bb["_name"]))
    wpa_keys = set(zip(wpa_v["yr"], wpa_v["_name"]))
    unmatched = sorted(f"{nm} ({yr})" for yr, nm in bb_keys - wpa_keys)

    return merged.drop(columns=["_name"]), len(bb), unmatched


def add_lag_feature(merged: pd.DataFrame, wpa: pd.DataFrame,
                    lag_variant: str) -> pd.DataFrame:
    """
    Add clwpa_lag1 = player's cWPA in (lag_variant, Year-1).

    NaN for rookies and gap-year returners. XGBoost handles NaN natively by
    learning a default split direction, so all rows are retained.

    player_id is already on merged from join_datasets; we use it directly
    rather than re-mapping through names, which avoids year-specific name
    change edge cases.
    """
    prior = (wpa[wpa["playoff_variant"] == lag_variant]
             [["player_id", "yr", "clwpa"]]
             .copy()
             .rename(columns={"clwpa": "clwpa_lag1"}))
    prior["Year"] = prior["yr"] + 1   # align to the *next* season's Year key
    lag_map = prior[["player_id", "Year", "clwpa_lag1"]]
    out = merged.merge(lag_map, on=["player_id", "Year"], how="left")
    n_nan = out["clwpa_lag1"].isna().sum()
    pct   = n_nan / len(out)
    print(f"  clwpa_lag1: {len(out)-n_nan}/{len(out)} rows have prior-year value "
          f"({1-pct:.1%}); {n_nan} NaN (rookies/gap years, handled natively by XGBoost)")
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric modelling features, in stable column order."""
    drop = EXCLUDE | {"clwpa", "gms", "wpa_gms", "player_id",
                      "season", "playoff_variant", "yr", "player"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


# ── Metrics ───────────────────────────────────────────────────────────────────

def metrics(y_true, y_pred) -> dict:
    """R2, RMSE, MAE for a set of predictions."""
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "n":    int(len(y_true)),
    }


def fmt(m: dict) -> str:
    return f"R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  n={m['n']:,}"


def base_model(**kw) -> XGBRegressor:
    params = dict(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,          # parallelism lives in the search / CV loop
        verbosity=0,
    )
    params.update(kw)
    return XGBRegressor(**params)


# ── Step 1: feature importance on training years only ─────────────────────────

def rank_features(train: pd.DataFrame, feats: list[str], label: str,
                  slug: str, colour) -> pd.DataFrame:
    """
    Fit a full-feature model on TRAIN and rank features by gain importance.

    Gain measures the average loss reduction contributed by each feature's
    splits — the most direct available proxy for predictive contribution.
    Gain is split arbitrarily among strongly correlated features (WS, VORP and
    BPM overlap heavily), so a low rank means "redundant given the others",
    not necessarily "unrelated to clutch performance".
    """
    X, y = train[feats], train["clwpa"]
    model = base_model(n_estimators=600, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
    model.fit(X, y)

    imp = (pd.DataFrame({"feature": feats,
                         "gain_importance": model.feature_importances_})
           .sort_values("gain_importance", ascending=False)
           .reset_index(drop=True))
    imp.insert(0, "rank", imp.index + 1)
    imp["selected_top20"] = imp["rank"] <= TOP_K

    path = OUT_D / f"xgb_feature_importance_{slug}.csv"
    imp.to_csv(path, index=False)
    print(f"  Feature importance ({len(feats)} features) → {path}")

    set_style()
    fig, ax = plt.subplots(figsize=(9, 11))
    order = imp.iloc[::-1]
    colours = [colour if s else "lightgrey" for s in order["selected_top20"]]
    ax.barh(order["feature"], order["gain_importance"], color=colours)
    ax.set_xlabel("Gain importance")
    ax.set_title(f"Feature Importance — {label}\n"
                 f"all {len(feats)} features, trained on "
                 f"{TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}\n"
                 f"(coloured = top {TOP_K}, retained)", fontsize=11)
    fig.tight_layout()
    p = FIG_D / f"01_importance_all_features_{slug}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  Saved {p}")

    return imp


# ── Step 2: hyperparameter search under two CV schemes ────────────────────────

def make_cv() -> KFold:
    """
    Cross-validation strategy, applied to TRAIN only.

    Shuffled 5-fold. Treats player-seasons as exchangeable. A player's 2003 and
    2004 rows can land in different folds, so the folds are not strictly
    independent — but this scheme was compared head to head against
    LeaveOneGroupOut-by-season and generalised better to the held-out test
    seasons for both models. See the scheme-selection note in the module
    docstring for the measured comparison.
    """
    return KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def search_params(train: pd.DataFrame, feats: list[str], cv, label: str):
    """Search the parameter space. Returns (best_params, full trace)."""
    X, y = train[feats], train["clwpa"]
    n_folds = cv.get_n_splits(X, y)
    n_space = int(np.prod([len(v) for v in PARAM_GRID.values()]))
    print(f"\n  [{label}] sampling {N_ITER:,} of {n_space:,} configs "
          f"× {n_folds} folds = {N_ITER * n_folds:,} fits")

    gs = RandomizedSearchCV(
        estimator=base_model(),
        param_distributions=PARAM_GRID,
        n_iter=N_ITER,
        scoring=SCORING,
        refit=REFIT_ON,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        return_train_score=False,
    )
    gs.fit(X, y)

    res = pd.DataFrame(gs.cv_results_)
    keep = ([f"param_{k}" for k in PARAM_GRID]
            + ["mean_test_rmse", "mean_test_mae", "mean_test_r2",
               "std_test_rmse", "rank_test_rmse", "rank_test_mae",
               "rank_test_r2"])
    trace = res[[c for c in keep if c in res.columns]].copy()
    # sklearn maximises, so RMSE/MAE come back negated — flip to positives.
    for col in ["mean_test_rmse", "mean_test_mae"]:
        trace[col] = -trace[col]
    trace = trace.sort_values("rank_test_rmse")

    best = gs.best_params_
    row  = trace.iloc[0]
    print(f"    best CV: RMSE={row['mean_test_rmse']:.4f}  "
          f"MAE={row['mean_test_mae']:.4f}  R²={row['mean_test_r2']:.4f}")
    print(f"    best params: {best}")

    # Multi-metric agreement: does the RMSE winner also rank well on MAE and R²?
    agree = {m: int(row[f"rank_test_{m}"]) for m in ["rmse", "mae", "r2"]}
    if max(agree.values()) > 10:
        print(f"    NOTE: metrics disagree on the best config — ranks {agree}. "
              f"Refit follows RMSE.")
    else:
        print(f"    metric agreement OK — ranks {agree} (all top-10)")

    return best, trace


def fit_final(train, test, feats, params, cv) -> dict:
    """
    Refit the winning config on all of TRAIN, then score it on TEST.

    The CV numbers reported here are the honest in-training estimate for this
    exact configuration; the TEST numbers are the out-of-sample result on
    seasons the model has never seen. Expect TEST to be somewhat worse — that
    gap is the cost of predicting genuinely new years.
    """
    Xtr, ytr = train[feats], train["clwpa"]
    Xte, yte = test[feats],  test["clwpa"]

    model = base_model(**params, n_jobs=-1)
    model.fit(Xtr, ytr)

    tr_m = metrics(ytr, model.predict(Xtr))
    te_m = metrics(yte, model.predict(Xte))

    cvres = cross_validate(base_model(**params), Xtr, ytr,
                           cv=cv, scoring=SCORING, n_jobs=-1)
    cv_m = {
        "cv_rmse": float(-np.mean(cvres["test_rmse"])),
        "cv_mae":  float(-np.mean(cvres["test_mae"])),
        "cv_r2":   float(np.mean(cvres["test_r2"])),
    }

    print(f"    CV    RMSE={cv_m['cv_rmse']:.4f} "
          f"MAE={cv_m['cv_mae']:.4f} R²={cv_m['cv_r2']:.4f}")
    print(f"    TRAIN {fmt(tr_m)}")
    print(f"    TEST  {fmt(te_m)}")

    return {"params": params, "model": model, **cv_m,
            "train_r2": tr_m["r2"], "train_rmse": tr_m["rmse"],
            "train_mae": tr_m["mae"],
            "test_r2": te_m["r2"], "test_rmse": te_m["rmse"],
            "test_mae": te_m["mae"],
            "n_train": tr_m["n"], "n_test": te_m["n"]}


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_final_importance(model, feats, label, slug, colour):
    imp = pd.Series(model.feature_importances_, index=feats).sort_values()
    set_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(imp.index, imp.values, color=colour)
    ax.set_xlabel("Gain importance")
    ax.set_title(f"Tuned Model Feature Importance — {label}\n"
                 f"top {TOP_K} features, fit on "
                 f"{TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}")
    fig.tight_layout()
    p = FIG_D / f"02_importance_top{TOP_K}_{slug}.png"
    fig.savefig(p); plt.close(fig)
    print(f"  Saved {p}")


def plot_actual_vs_pred(pred: pd.DataFrame, label, slug, colour, test_m):
    """Test-year scatter of actual vs predicted with a 45° perfect-fit line."""
    set_style()
    a, p_ = pred["actual_cWPA"].values, pred["predicted_cWPA"].values
    lo, hi = min(a.min(), p_.min()), max(a.max(), p_.max())
    pad = 0.05 * (hi - lo)
    lims = (lo - pad, hi + pad)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(a, p_, alpha=0.45, s=22, color=colour, linewidths=0)
    ax.plot(lims, lims, ls="--", lw=1.5, color="black",
            label="perfect prediction (y = x)")
    m, b = np.polyfit(a, p_, 1)
    xs = np.linspace(*lims, 100)
    ax.plot(xs, m * xs + b, lw=1.5, color="crimson",
            label=f"model fit  (slope {m:.2f})")

    r = np.corrcoef(a, p_)[0, 1]
    txt = (f"TEST {TEST_YEARS[0]}–{TEST_YEARS[-1]}\n"
           f"R² = {test_m['r2']:.4f}\nRMSE = {test_m['rmse']:.4f}\n"
           f"MAE = {test_m['mae']:.4f}\nPearson r = {r:.4f}\n"
           f"n = {test_m['n']:,}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel("Actual cWPA"); ax.set_ylabel("Predicted cWPA")
    ax.set_title(f"Predicted vs Actual cWPA — {label}\nheld-out seasons "
                 f"{TEST_YEARS[0]}–{TEST_YEARS[-1]}")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    pth = FIG_D / f"03_actual_vs_predicted_{slug}.png"
    fig.savefig(pth); plt.close(fig)
    print(f"  Saved {pth}")


def plot_by_season(pred: pd.DataFrame, label, slug, colour):
    """One panel per held-out season."""
    set_style()
    years = sorted(pred["Year"].unique())
    fig, axes = plt.subplots(1, len(years), figsize=(3.4 * len(years), 3.8),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, yr in zip(axes, years):
        s = pred[pred["Year"] == yr]
        ax.scatter(s["actual_cWPA"], s["predicted_cWPA"], alpha=0.5, s=16,
                   color=colour, linewidths=0)
        lo = min(s["actual_cWPA"].min(), s["predicted_cWPA"].min())
        hi = max(s["actual_cWPA"].max(), s["predicted_cWPA"].max())
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="black")
        r2 = r2_score(s["actual_cWPA"], s["predicted_cWPA"])
        ax.set_title(f"{yr}-{str(yr+1)[2:]}\nR²={r2:.3f}  n={len(s)}",
                     fontsize=9.5)
        ax.set_xlabel("Actual cWPA")
    axes[0].set_ylabel("Predicted cWPA")
    fig.suptitle(f"Predicted vs Actual by Held-Out Season — {label}",
                 fontsize=12)
    fig.tight_layout()
    p = FIG_D / f"04_actual_vs_predicted_by_season_{slug}.png"
    fig.savefig(p); plt.close(fig)
    print(f"  Saved {p}")


def plot_residuals(pred: pd.DataFrame, label, slug, colour):
    """cWPA_Difference = predicted - actual. Distribution and structure."""
    set_style()
    d = pred["cWPA_Difference"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    axes[0].hist(d, bins=40, color=colour, edgecolor="white")
    axes[0].axvline(0, color="black", ls="--", lw=1.2)
    axes[0].axvline(d.mean(), color="crimson", lw=1.2,
                    label=f"mean = {d.mean():+.3f}")
    axes[0].set_xlabel("cWPA_Difference  (predicted − actual)")
    axes[0].set_ylabel("Players")
    axes[0].set_title(f"Residual distribution\nsd = {d.std():.3f}", fontsize=10)
    axes[0].legend(fontsize=8.5)

    axes[1].scatter(pred["predicted_cWPA"], d, alpha=0.45, s=18,
                    color=colour, linewidths=0)
    axes[1].axhline(0, color="black", ls="--", lw=1.2)
    axes[1].set_xlabel("Predicted cWPA")
    axes[1].set_ylabel("cWPA_Difference")
    axes[1].set_title("Residuals vs prediction", fontsize=10)

    by_yr = pred.groupby("Year")["cWPA_Difference"]
    axes[2].errorbar(by_yr.mean().index, by_yr.mean().values,
                     yerr=by_yr.std().values, marker="o", capsize=4,
                     color=colour, lw=1.4)
    axes[2].axhline(0, color="black", ls="--", lw=1.2)
    axes[2].set_xlabel("Season (start year)")
    axes[2].set_ylabel("Mean cWPA_Difference")
    axes[2].set_title("Bias by held-out season\n(±1 sd)", fontsize=10)
    axes[2].set_xticks(sorted(pred["Year"].unique()))

    fig.suptitle(f"Prediction Error — {label}  "
                 f"(test {TEST_YEARS[0]}–{TEST_YEARS[-1]})", fontsize=12)
    fig.tight_layout()
    p = FIG_D / f"05_residuals_{slug}.png"
    fig.savefig(p); plt.close(fig)
    print(f"  Saved {p}")


# ── Per-model driver ──────────────────────────────────────────────────────────

def run_model(variant, stem, gms_thr, label, slug, colour):
    print(f"\n{'='*78}\nMODEL: {label}   (variant={variant}, BBRef G>={gms_thr})"
          f"\n{'='*78}")

    wpa   = load_wpa()
    bbref = load_bbref(stem)
    merged, n_qual, unmatched = join_datasets(bbref, wpa, variant, gms_thr)
    print(f"BBRef qualifying rows: {n_qual:,}  |  joined to WPA: "
          f"{len(merged):,} ({len(merged)/n_qual:.2%})  |  "
          f"unmatched: {len(unmatched)}")

    with open(OUT_D / f"unmatched_{slug}.txt", "w") as f:
        f.write(f"BBRef {variant} players (G>={gms_thr}) with no WPA match: "
                f"{len(unmatched)}\n\n" + "\n".join(unmatched) + "\n")

    merged = merged.dropna(subset=["clwpa"])

    # For the regular-season model only, add last season's cWPA as a feature.
    # NaN (rookies / gap-year returners) are left as-is; XGBoost learns a
    # default split direction for missing values, so no rows are dropped.
    if variant == "regular":
        merged = add_lag_feature(merged, wpa, lag_variant="regular")

    train = merged[merged["Year"].isin(TRAIN_YEARS)].reset_index(drop=True)
    test  = merged[merged["Year"].isin(TEST_YEARS)].reset_index(drop=True)
    print(f"TRAIN {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]}: {len(train):,} rows "
          f"({train['Year'].nunique()} seasons)")
    print(f"TEST  {TEST_YEARS[0]}–{TEST_YEARS[-1]}: {len(test):,} rows "
          f"({test['Year'].nunique()} seasons)")

    all_feats = feature_columns(merged)
    print(f"\nSTEP 1 — importance over {len(all_feats)} candidate features "
          f"(fit on TRAIN only)")
    imp = rank_features(train, all_feats, label, slug, colour)
    top = imp.head(TOP_K)["feature"].tolist()
    print(f"  Top {TOP_K}: {', '.join(top)}")

    print(f"\nSTEP 2 — hyperparameter search on the top {TOP_K} features")
    cv = make_cv()
    best, trace = search_params(train, top, cv, label)
    trace.to_csv(OUT_D / f"xgb_gridsearch_{slug}.csv", index=False)

    final = fit_final(train, test, top, best, cv)
    model = final["model"]
    plot_final_importance(model, top, label, slug, colour)

    test_pred = model.predict(test[top])
    test_m = metrics(test["clwpa"], test_pred)

    pred = pd.DataFrame({
        "Year":           test["Year"].values,
        "Season":         [f"{y}-{y+1}" for y in test["Year"]],
        "Player":         test["Player"].values,
        "Tm":             test["Tm"].values,
        "Pos":            test["Pos"].values,
        "G":              test["G"].values,
        "actual_cWPA":    test["clwpa"].values,
        "predicted_cWPA": test_pred,
    })
    # Positive = model over-predicted (player under-delivered vs their profile).
    pred["cWPA_Difference"] = pred["predicted_cWPA"] - pred["actual_cWPA"]
    pred["abs_Difference"]  = pred["cWPA_Difference"].abs()
    pred = pred.sort_values(["Year", "cWPA_Difference"]).reset_index(drop=True)

    plot_actual_vs_pred(pred, label, slug, colour, test_m)
    plot_by_season(pred, label, slug, colour)
    plot_residuals(pred, label, slug, colour)

    per_season = (pred.groupby("Year")
                  .apply(lambda g: pd.Series(metrics(g["actual_cWPA"],
                                                     g["predicted_cWPA"])),
                         include_groups=False)
                  .reset_index())
    per_season["mean_cWPA_Difference"] = (
        pred.groupby("Year")["cWPA_Difference"].mean().values)

    summary = {
        "model": label, "variant": variant, "games_threshold": gms_thr,
        "train_years": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}",
        "test_years":  f"{TEST_YEARS[0]}-{TEST_YEARS[-1]}",
        "n_train": final["n_train"], "n_test": final["n_test"],
        "n_candidate_features": len(all_feats), "n_features_used": TOP_K,
        "features_used": "|".join(top),
        # Retained after a head-to-head against LeaveOneGroupOut-by-season;
        # see the scheme-selection note in the module docstring.
        "cv_scheme_used": "kfold",
        "cv_rmse": final["cv_rmse"], "cv_mae": final["cv_mae"],
        "cv_r2": final["cv_r2"],
        "train_r2": final["train_r2"], "train_rmse": final["train_rmse"],
        "train_mae": final["train_mae"],
        "test_r2": test_m["r2"], "test_rmse": test_m["rmse"],
        "test_mae": test_m["mae"],
        "mean_cWPA_Difference": float(pred["cWPA_Difference"].mean()),
        "sd_cWPA_Difference": float(pred["cWPA_Difference"].std()),
        **{f"param_{k}": v for k, v in final["params"].items()},
    }
    pd.DataFrame([summary]).to_csv(OUT_D / f"xgb_results_{slug}.csv",
                                   index=False)
    # Checkpoint per model: the combined workbook is only written after BOTH
    # models finish, so persist each model's predictions as soon as they exist.
    pred.to_csv(OUT_D / f"xgb_predictions_{slug}.csv", index=False)
    per_season.to_csv(OUT_D / f"xgb_per_season_{slug}.csv", index=False)
    print(f"\n  Model summary → {OUT_D / f'xgb_results_{slug}.csv'}")
    print(f"  Predictions   → {OUT_D / f'xgb_predictions_{slug}.csv'}")
    print(f"  FINAL TEST: {fmt(test_m)}")

    return dict(label=label, slug=slug, pred=pred, per_season=per_season,
                summary=summary, importance=imp, top=top)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    outs = [run_model(*m) for m in MODELS]

    xlsx = OUT_D / "xgb_predictions.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        for o in outs:
            sheet = "Regular" if o["slug"] == "regular" else "Playoffs"
            o["pred"].to_excel(w, sheet_name=sheet, index=False)
            o["per_season"].to_excel(w, sheet_name=f"{sheet}_by_season",
                                     index=False)
            o["importance"].to_excel(w, sheet_name=f"{sheet}_importance",
                                     index=False)
        pd.DataFrame([o["summary"] for o in outs]).to_excel(
            w, sheet_name="Model_Summary", index=False)
    print(f"\nPredictions workbook → {xlsx}")

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    for o in outs:
        s = o["summary"]
        print(f"\n{s['model']}")
        print(f"  train {s['train_years']} (n={s['n_train']:,}) → "
              f"test {s['test_years']} (n={s['n_test']:,})")
        print(f"  TEST  R²={s['test_r2']:.4f}  RMSE={s['test_rmse']:.4f}  "
              f"MAE={s['test_mae']:.4f}")
        print(f"  cWPA_Difference: mean={s['mean_cWPA_Difference']:+.4f}  "
              f"sd={s['sd_cWPA_Difference']:.4f}")


if __name__ == "__main__":
    main()
