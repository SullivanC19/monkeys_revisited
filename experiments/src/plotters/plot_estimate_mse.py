import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from constants import DIR_EST_RESULTS, DIR_PLOTS, SOURCES_TO_TITLES, APPROACH_TO_TITLES
from utilities import load_all_parquets

sns.set_theme(style="whitegrid")
sns.set_context("paper")

greens = sns.color_palette("Greens", 4)
purples = sns.color_palette("Purples", 5)
PALETTE = {
    "Regression": greens[1],
    "Discretization": greens[3],
    "Dynamic": purples[3],
}

ID_COLS   = ["Problem", "Model", "k", "Budget"]
TRUE_COL  = "True Pass@k"
METHOD_ORDER = ["Regression", "Discretization", "Dynamic"]
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


def _facet_panel(data: pd.DataFrame, **kwargs):
    ax = plt.gca()
    for method, gm in data.groupby("Method", sort=True, observed=True):
        gm = gm.sort_values("Budget").dropna(subset=["center","low","high"])
        if gm.empty:
            continue
        color = PALETTE.get(str(method))
        ax.plot(gm["Budget"], gm["center"], marker="o", linewidth=1.6,
                label=str(method), color=color)
        ax.fill_between(gm["Budget"], gm["low"], gm["high"],
                        color=color, alpha=0.15, linewidth=0)

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

def plot_mse_facets_by_k(stats: pd.DataFrame, out_dir=DIR_PLOTS, col_wrap=4):
    stats = stats.copy()
    stats["Method"] = pd.Categorical(stats["Method"], METHOD_ORDER, ordered=True)
    stats["Pair"] = stats["Model"].astype(str) + " — " + stats["Problem"].astype(str)

    for (k, problem), gkp in stats.groupby(["k", "Problem"], dropna=False):
        gkp = gkp.sort_values(["Model", "Budget", "Method"])
        models = gkp["Model"].dropna().unique().tolist()
        keep = models[:len(models) // col_wrap * col_wrap]
        gkp = gkp[gkp["Model"].isin(keep)]

        g = sns.FacetGrid(
            gkp, col="Model", col_wrap=col_wrap, sharex=True, sharey=True,
            despine=True
        )
        g.map_dataframe(_facet_panel)

        # Log y-axis and labels per facet
        for ax in g.axes.flat:
            ax.set_xlim(left=100, right=10000)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel("Budget")
            ax.set_ylabel("MSE vs True Pass@k")

        # Titles
        g.set_titles(col_template="{col_name}")
        g.figure.suptitle(f"Prediction Mean Squared Error (MSE) for {SOURCES_TO_TITLES[problem]}", y=1.02, fontsize=14)

        # Shared legend (one for all facets)
        handles = [Line2D([0],[0], color=PALETTE[m], marker="o", linewidth=1.6, label=m)
                   for m in METHOD_ORDER if m in stats["Method"].unique()]
        labels = [APPROACH_TO_TITLES[h.get_label()] for h in handles]
        g.figure.legend(handles=handles, labels=labels, loc="center", frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.01))

        plt.tight_layout()
        mse_out_dir = out_dir / "mse"
        mse_out_dir.mkdir(parents=True, exist_ok=True)
        base = f"estimate_mse-k={k}-problem={problem}"
        g.savefig(mse_out_dir / f"{base}.png", dpi=200, bbox_inches="tight")
        g.savefig(mse_out_dir / f"{base}.pdf", bbox_inches="tight")
        plt.close(g.figure)

def run():
    print(f"Loading results from: {DIR_EST_RESULTS.resolve()}")
    df = load_all_parquets(DIR_EST_RESULTS)

    print("Bootstrapping MSE for CIs...")
    stats = bootstrap_mse(df, B=10000)

    print("Plotting MSE facets by k...")
    plot_mse_facets_by_k(stats, DIR_PLOTS, col_wrap=3)

    print(f"Saved figures → {DIR_PLOTS.resolve()}")
