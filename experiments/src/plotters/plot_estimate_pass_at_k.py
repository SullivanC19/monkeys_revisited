import pandas as pd
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from constants import DIR_EST_RESULTS, DIR_PLOTS, SOURCES_TO_TITLES, APPROACH_TO_TITLES
from utilities import load_all_parquets

greens = sns.color_palette("Greens", 4)
purples = sns.color_palette("Purples", 5)
PALETTE = {"Regression": greens[1], "Discretization": greens[3], "Dynamic": purples[3]}

ID_COLS   = ["Problem", "Model", "k", "Budget"]
TRUE_COL  = "True Pass@k"
SUBP_COL  = "Subproblems"
METHOD_ORDER = ["Regression", "Discretization", "Dynamic"]
METHOD_CAT = CategoricalDtype(METHOD_ORDER, ordered=True)
METHOD_COLS = [f"{m} Estimate" for m in METHOD_ORDER]

def _long_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long form with per-method estimate and the true pass@k retained for each row.
    """
    long = df.melt(
        id_vars=[c for c in ID_COLS if c in df.columns] + [TRUE_COL, SUBP_COL],
        value_vars=[c for c in METHOD_COLS if c in df.columns],
        var_name="Method",
        value_name="Estimate",
    )
    long["Method"] = long["Method"].str.replace(" Estimate", "", regex=False).astype(METHOD_CAT)
    long["Estimate"] = pd.to_numeric(long["Estimate"], errors="coerce")
    long[TRUE_COL] = pd.to_numeric(long[TRUE_COL], errors="coerce")
    long[SUBP_COL] = pd.to_numeric(long[SUBP_COL], errors="coerce")
    return long

def bootstrap_estimates(df: pd.DataFrame, B: int = 1000, seed: int | None = 0) -> pd.DataFrame:
    """
    Bootstrap the mean pass@k estimate across seeds/runs for CIs, per (Problem, Model, k, Budget, Method).
    Also carries a representative True Pass@k for each group (median across rows, but it should be constant).
    """
    rng = np.random.default_rng(seed)
    long = _long_estimates(df)

    rows = []
    group_cols = ID_COLS + ["Method"]
    for keys, g in long.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple): keys = (keys,)
        rec = dict(zip(group_cols, keys))

        x = g["Estimate"].dropna().to_numpy(dtype=float)
        n = x.size
        # Take a robust representative for true (should be constant)
        true_val = float(g[TRUE_COL].dropna().median()) if g[TRUE_COL].notna().any() else np.nan
        n_subproblems = float(g[SUBP_COL].iloc[0]) if g[SUBP_COL].notna().any() else np.nan
        rec[TRUE_COL] = true_val
        rec[SUBP_COL] = n_subproblems

        if n == 0:
            rec.update(center=np.nan, low=np.nan, high=np.nan, n=0)
        elif n == 1:
            m = float(x.mean())
            rec.update(center=m, low=m, high=m, n=1)
        else:
            m = float(x.mean())
            idx = rng.integers(0, n, size=(B, n))
            los, his = np.percentile(x[idx], [2.5, 97.5], axis=1)
            rec.update(center=m, low=float(los.mean()), high=float(his.mean()), n=n)
        rows.append(rec)

    out = pd.DataFrame(rows)
    if "Method" in out.columns:
        out["Method"] = out["Method"].astype(METHOD_CAT)
    return out

def _facet_panel_over_k_with_truth(data: pd.DataFrame, **kwargs):
    """
    Plot method estimates vs k (x-axis), with CI ribbons, plus dashed true pass@k vs k.
    Assumes columns: k, Method, center, low, high, True Pass@k.
    """
    ax = plt.gca()

    # Plot method lines and ribbons
    for method, gm in data.groupby("Method", sort=True, observed=True):
        gm = gm.sort_values("k").dropna(subset=["center", "low", "high"])
        if gm.empty:
            continue
        color = PALETTE.get(str(method))
        ax.plot(gm["k"], gm["center"], linewidth=0.8,
                label=str(method), color=color)
        ax.fill_between(gm["k"], gm["low"], gm["high"],
                        color=color, alpha=0.15, linewidth=0)

    # True pass@k vs k (allowing truth to vary with k)
    if TRUE_COL in data.columns and data[TRUE_COL].notna().any():
        gt = (data[["k", TRUE_COL]]
              .dropna()
              .groupby("k", as_index=False)[TRUE_COL].median()
              .sort_values("k"))
        if not gt.empty:
            ax.plot(gt["k"], gt[TRUE_COL], linestyle="--", linewidth=0.8,
                    color="black", label="True Pass@k")
            
    # Highlight low-k region
    k_th = data[SUBP_COL].mode().iloc[0]
    ax.axvspan(xmin=ax.get_xlim()[0], xmax=k_th, 
               facecolor="gray", alpha=0.2)
    ax.axvline(k_th, color="gray", linestyle="--")

def plot_passk_over_k_by_budget(stats_est: pd.DataFrame, out_dir=DIR_PLOTS, col_wrap=4, budgets: list | None = None):
    """
    For each (Problem, Budget), make a figure faceted by Model (columns),
    plotting pass@k estimates vs k (x-axis) with bootstrap CIs and the true pass@k curve.
    """
    stats = stats_est.copy()
    stats["Method"] = pd.Categorical(stats["Method"], METHOD_ORDER, ordered=True)

    # Optional budget subset
    if budgets is not None:
        stats = stats[stats["Budget"].isin(budgets)]

    # Build one figure per (Problem, Budget)
    for (problem, budget), gb in stats.groupby(["Problem", "Budget"], dropna=False):
        if gb.empty:
            continue

        gb = gb.sort_values(["Model", "k", "Method"])
        models = gb["Model"].dropna().unique().tolist()
        keep = models[:len(models) // col_wrap * col_wrap] if len(models) >= col_wrap else models
        gb = gb[gb["Model"].isin(keep)]

        g = sns.FacetGrid(
            gb, col="Model", col_wrap=col_wrap, sharex=True, sharey=True, despine=True
        )
        g.map_dataframe(_facet_panel_over_k_with_truth)

        # Axes formatting
        for ax in g.axes.flat:
            # k is typically integer and not huge; keep linear x-scale
            xmin = max(1, int(np.nanmin(gb["k"]))) if np.isfinite(gb["k"]).any() else 1
            xmax = int(np.nanmax(gb["k"])) if np.isfinite(gb["k"]).any() else 1
            if xmax <= xmin:
                xmax = xmin + 1
            ax.set_xscale("log")
            ax.set_xlim(left=xmin, right=xmax)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("k")
            ax.set_ylabel("Estimated pass@k")

        # Titles & legend
        g.set_titles(col_template="{col_name}")
        g.figure.suptitle(f"Prediction pass@k for {SOURCES_TO_TITLES[problem]}",
                          y=1.02, fontsize=14)

        method_handles = [Line2D([0],[0], color=PALETTE[m], linewidth=0.8, label=m)
                          for m in METHOD_ORDER if m in stats["Method"].unique()]
        method_labels  = [APPROACH_TO_TITLES[h.get_label()] for h in method_handles]
        truth_handle   = Line2D([0],[0], color="black", linestyle="--", linewidth=0.8, label="True Pass@k")
        handles = method_handles + [truth_handle]
        labels  = method_labels  + ["True Pass@k"]

        g.figure.legend(handles=handles, labels=labels, loc="center", frameon=False,
                        ncol=min(4, len(handles)), bbox_to_anchor=(0.5, -0.01))

        plt.tight_layout()
        pass_at_k_dir = out_dir / "pass_at_k"
        pass_at_k_dir.mkdir(parents=True, exist_ok=True)
        base = f"problem={problem}-budget={int(budget) if pd.notna(budget) else budget}"
        g.savefig(pass_at_k_dir / f"{base}.png", dpi=200, bbox_inches="tight")
        g.savefig(pass_at_k_dir / f"{base}.pdf", bbox_inches="tight")
        plt.close(g.figure)

def run():
    print(f"Loading results from: {DIR_EST_RESULTS.resolve()}")
    df = load_all_parquets(DIR_EST_RESULTS)

    print("Bootstrapping estimates for CIs...")
    stats_est = bootstrap_estimates(df, B=1000)

    print("Plotting pass@k vs k by budget...")
    plot_passk_over_k_by_budget(stats_est, DIR_PLOTS, col_wrap=3)

    print(f"Saved figures → {DIR_PLOTS.resolve()}")
