import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from constants import DIR_RESULTS, DIR_PLOTS, SOURCES_TO_TITLES, APPROACH_TO_TITLES

sns.set_theme(style="whitegrid")
sns.set_context("paper")

greens = sns.color_palette("Greens", 4)
purple = sns.color_palette("Purples", 5)[3]
PALETTE = {"Regression": greens[1], "Discretization": greens[3], "Dynamic": purple}

ID_COLS   = ["Problem","Model","k","Budget"]
TRUE_COL  = "True Pass@k"
METHOD_ORDER = ["Regression", "Discretization", "Dynamic"]
METHOD_CAT = CategoricalDtype(METHOD_ORDER, ordered=True)
METHOD_COLS = [f"{m} Estimate" for m in METHOD_ORDER]

def load_all_parquets(in_dir=DIR_RESULTS) -> pd.DataFrame:
    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {in_dir}")
    df = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in files], ignore_index=True)
    # ensure numeric
    for c in ("Budget","k"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

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


def compute_mse_long(df: pd.DataFrame) -> pd.DataFrame:
    long = df.melt(
        id_vars=[c for c in ["Problem","Model","k","Budget"] if c in df.columns] + ["True Pass@k"],
        value_vars=[c for c in ["Regression Estimate","Discretization Estimate","Dynamic Estimate"] if c in df.columns],
        var_name="Method",
        value_name="Estimate",
    )
    long["Method"] = long["Method"].str.replace(" Estimate", "", regex=False)
    long["SE"] = (pd.to_numeric(long["Estimate"], errors="coerce")
                  - pd.to_numeric(long["True Pass@k"], errors="coerce")) ** 2
    agg = (long.groupby(["Problem","Model","k","Budget","Method"], dropna=False)["SE"]
                 .agg(MSE="mean", SD="std", N="count")
                 .reset_index())
    agg["SD"] = agg["SD"].fillna(0.0)  # if only one seed, std is NaN → 0
    return agg

def plot_mse_facets_by_k(stats: pd.DataFrame, out_dir=DIR_PLOTS, col_wrap=4):
    stats = stats.copy()
    stats["Method"] = pd.Categorical(stats["Method"], METHOD_ORDER, ordered=True)
    stats["Pair"] = stats["Model"].astype(str) + " — " + stats["Problem"].astype(str)

    for (k_val, problem), gkp in stats.groupby(["k", "Problem"], dropna=False):
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
            ax.set_yscale("log")
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
        base = f"estimate_mse-k={k_val}-problem={problem}"
        g.savefig(out_dir / f"{base}.png", dpi=200, bbox_inches="tight")
        g.savefig(out_dir / f"{base}.pdf", bbox_inches="tight")
        plt.close(g.figure)

def run():
    print(f"Loading results from: {DIR_RESULTS.resolve()}")
    df = load_all_parquets(DIR_RESULTS)
    stats = bootstrap_mse(df, B=1000)
    plot_mse_facets_by_k(stats, DIR_PLOTS, col_wrap=3)
    print(f"Saved figures → {DIR_PLOTS.resolve()}")
