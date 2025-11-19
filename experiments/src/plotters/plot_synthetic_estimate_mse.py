import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from constants import DIR_SYN_RESULTS, DIR_PLOTS, APPROACH_TO_TITLES
from utilities import load_all_parquets

sns.set_theme(style="whitegrid")
sns.set_context("paper")

greens = sns.color_palette("Greens", 4)
grays = sns.color_palette("Greys", 2)
purples = sns.color_palette("Purples", 5)
PALETTE = {
    "Uniform": purples[2],
    "Dynamic": purples[3],
    "Discretization": greens[2],
    "Dynamic Discretization": greens[3],
}

ID_COLS   = ["k", "Budget"]
TRUE_COL  = "True Pass@k"
METHOD_ORDER = ["Uniform", "Dynamic", "Discretization", "Dynamic Discretization"]
METHOD_CAT = CategoricalDtype(METHOD_ORDER, ordered=True)
METHOD_COLS = [f"{m} Estimate" for m in METHOD_ORDER]

def _long_mse(df: pd.DataFrame) -> pd.DataFrame:
    long = df.melt(
        id_vars=[c for c in ID_COLS if c in df.columns] + [TRUE_COL],
        value_vars=[c for c in METHOD_COLS if c in df.columns],
        var_name="Method",
        value_name="Estimate",
    )
    # strip " Estimate" to get Method names that match your palette/order
    long["Method"] = long["Method"].str.replace(" Estimate", "", regex=False).astype(METHOD_CAT)
    se = (pd.to_numeric(long["Estimate"], errors="coerce")
          - pd.to_numeric(long[TRUE_COL], errors="coerce")) ** 2
    return long.assign(SE=se)

def bootstrap_mse(df: pd.DataFrame, B: int = 1000, seed: int | None = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    EPS = 1e-12
    long = _long_mse(df)

    rows = []
    group_cols = ID_COLS + ["Method"]
    for keys, g in long.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple): keys = (keys,)
        rec = dict(zip(group_cols, keys))
        x = g["SE"].to_numpy(dtype=float)
        n = x.size
        if n == 0:
            rec.update(center=np.nan, low=np.nan, high=np.nan, n=0)
        elif n == 1:
            m = float(x.mean())
            rec.update(center=max(m, EPS), low=max(m, EPS), high=max(m, EPS), n=1)
        else:
            m = float(x.mean())
            idx = rng.integers(0, n, size=(B, n))
            boot = x[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rec.update(center=max(m, EPS), low=max(lo, EPS), high=max(hi, EPS), n=n)
        rows.append(rec)

    out = pd.DataFrame(rows)
    if "Method" in out.columns:
        out["Method"] = out["Method"].astype(METHOD_CAT)
    return out

def plot_mse_single_k(stats: pd.DataFrame, out_dir=DIR_PLOTS):
    """Plot MSE vs Budget for a single k value."""
    stats = stats.copy()
    stats["Method"] = pd.Categorical(stats["Method"], METHOD_ORDER, ordered=True)

    # Sort for clean plotting
    stats = stats.sort_values(["Budget", "Method"])

    fig, ax = plt.subplots(figsize=(6, 4))

    # Draw each method
    for method, gm in stats.groupby("Method", sort=True, observed=True):
        gm = gm.dropna(subset=["center", "low", "high"]).sort_values("Budget")
        if gm.empty:
            continue
        color = PALETTE.get(str(method), "black")
        ax.plot(
            gm["Budget"], gm["center"],
            marker="o", linewidth=1.6,
            label=APPROACH_TO_TITLES.get(str(method), str(method)), color=color
        )
        ax.fill_between(
            gm["Budget"], gm["low"], gm["high"],
            color=color, alpha=0.15, linewidth=0
        )

    # Log scales
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=100, right=10000)

    ax.set_xlabel("Budget")
    ax.set_ylabel("MSE vs True Pass@k")

    # Title and legend
    k_value = stats["k"].iloc[0]
    ax.set_title(f"Prediction Mean Squared Error (MSE) on Synthetic Data", fontsize=12)

    present_methods = [m for m in METHOD_ORDER if m in stats["Method"].unique()]
    ax.legend(
        handles=[
            Line2D([0], [0], color=PALETTE[m], marker="o", linewidth=1.6, label=APPROACH_TO_TITLES[m])
            for m in present_methods
        ],
        frameon=False,
        loc="best"
    )

    # Save
    out_dir = out_dir / "syn"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"estimate_mse-k={k_value}"
    fig.savefig(out_dir / f"{base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)

def run():
    print(f"Loading results from: {DIR_SYN_RESULTS.resolve()}")
    df = load_all_parquets(DIR_SYN_RESULTS)

    print("Bootstrapping MSE for CIs...")
    stats = bootstrap_mse(df, B=10000)

    print("Plotting MSE facets by k...")
    plot_mse_single_k(stats, DIR_PLOTS)

    print(f"Saved figures → {DIR_PLOTS.resolve()}")
