"""Statistical helpers and the canonical WPA loader, shared across scripts."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent

SHEETS = [("regular", "Regular"), ("playoffs", "Playoffs"), ("combined", "Combined")]


# ── Canonical dataset loader ──────────────────────────────────────────────────
def load_cwpa() -> pd.DataFrame:
    """
    Load the canonical long cWPA table from data/processed/cwpa_long.xlsx.

    This workbook is the SINGLE source of truth for WPA data. Every consumer —
    eda.py, analysis.py, build_xgboost.py — reads it, so they cannot drift apart.

    A cwpa_long.csv used to sit alongside it and was read by eda.py and
    analysis.py. It was deleted: inpredictable.com serves some player names with
    corrupted encoding (Nikola Joki?, Dario <fffd>ari?), and the repair had only
    ever been applied to the workbook. The CSV still carried 52 corrupted rows,
    so the two files disagreed about 27 players. One clean source removes the
    whole class of problem.

    Returns the three variants stacked, with `playoff_variant` restored (the
    per-sheet copies drop it) and `yr` as an integer season-start key.
    """
    xlsx = ROOT / "data" / "processed" / "cwpa_long.xlsx"
    frames = []
    for variant, sheet in SHEETS:
        d = pd.read_excel(xlsx, sheet_name=sheet, dtype={"player_id": str})
        d["playoff_variant"] = variant
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if "yr" not in df.columns:
        df["yr"] = df["season"].str[:4].astype(int)
    return df

# ── Styling ──────────────────────────────────────────────────────────────────
PALETTE = sns.color_palette("muted")
C_REG   = PALETTE[0]   # blue  — regular
C_PO    = PALETTE[1]   # orange — playoffs
C_COMB  = PALETTE[2]   # green  — combined

def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


# ── Correlation helpers ───────────────────────────────────────────────────────
def pearson_with_ci(x, y, alpha=0.05):
    """Return (r, p, ci_lo, ci_hi, n) with 95% Fisher-z CI."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    r, p = stats.pearsonr(x, y)
    # Fisher z
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lo = np.tanh(z - z_crit * se)
    ci_hi = np.tanh(z + z_crit * se)
    return r, p, ci_lo, ci_hi, n


def spearman_rho(x, y):
    """Return (rho, p, n)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    rho, p = stats.spearmanr(x, y)
    return rho, p, len(x)


def corr_summary(x, y, label=""):
    """Print and return a dict with Pearson + Spearman stats."""
    r, p, ci_lo, ci_hi, n = pearson_with_ci(np.array(x, float), np.array(y, float))
    rho, p_s, _ = spearman_rho(np.array(x, float), np.array(y, float))
    d = dict(label=label, n=n,
             pearson_r=round(r, 4), pearson_p=round(p, 6),
             ci_lo=round(ci_lo, 4), ci_hi=round(ci_hi, 4),
             spearman_rho=round(rho, 4), spearman_p=round(p_s, 6))
    print(f"[{label}] n={n}  Pearson r={r:.4f} [{ci_lo:.4f},{ci_hi:.4f}] p={p:.2e}"
          f"  |  Spearman ρ={rho:.4f} p={p_s:.2e}")
    return d


# ── Scatter / regression plot ─────────────────────────────────────────────────
def scatter_corr(x, y, xlabel, ylabel, title, out_path,
                 color=C_REG, alpha=0.35, size=18, annotate=True):
    """Scatter + OLS line with Pearson r and n annotated."""
    set_style()
    x, y = np.array(x, float), np.array(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    r, p, ci_lo, ci_hi, n = pearson_with_ci(x, y)
    rho, p_s, _ = spearman_rho(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x, y, alpha=alpha, s=size, color=color, linewidths=0)

    # OLS line
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, m * xs + b, color="black", lw=1.5, ls="--")

    if annotate:
        r2 = r ** 2
        txt = (f"R² = {r2:.2f}   n = {n:,}")
        ax.text(0.04, 0.96, txt, transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")
