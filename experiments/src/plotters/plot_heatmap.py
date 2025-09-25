# heatmaps_by_problem.py
from pathlib import Path
import re, unicodedata
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

from constants import DIR_EST_RESULTS, DIR_PLOTS, SOURCES_TO_TITLES, APPROACH_TO_TITLES
from utilities import load_all_parquets

sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=0.85)

METHOD_ORDER = ["Regression", "Discretization", "Dynamic"]
METHOD_COLS  = [f"{m} Estimate" for m in METHOD_ORDER]
GP_CMAP = cm.get_cmap("Purples_r")

ID_COLS  = ["Problem", "Model", "k", "Budget"]
TRUE_COL = "True Pass@k"

def compute_mean_mse(df: pd.DataFrame) -> pd.DataFrame:
    long = df.melt(
        id_vars=[c for c in ID_COLS if c in df.columns] + [TRUE_COL],
        value_vars=[c for c in METHOD_COLS if c in df.columns],
        var_name="Method",
        value_name="Estimate",
    )
    long["Method"] = long["Method"].str.replace(" Estimate", "", regex=False)
    se = (pd.to_numeric(long["Estimate"], errors="coerce")
          - pd.to_numeric(long[TRUE_COL], errors="coerce")) ** 2
    long = long.assign(MSE=se)
    mse = (long.groupby(ID_COLS + ["Method"], dropna=False)["MSE"]
               .mean()
               .reset_index())
    return mse

def plot_mse_heatmaps_by_problem(mse: pd.DataFrame, out_dir: Path, model_order: list[str] | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    mse = mse.copy()
    mse["Method"] = pd.Categorical(mse["Method"], METHOD_ORDER, ordered=True)

    for problem, gp in mse.groupby("Problem", dropna=False):
        models = (sorted(gp["Model"].dropna().unique().tolist())
                  if model_order is None else model_order)
        n_rows, n_cols = len(METHOD_ORDER), len(models)  # rows = approaches, cols = models
        if n_rows == 0 or n_cols == 0:
            continue

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(2*n_cols, 3.0*n_rows),
            sharex=True, sharey=True, constrained_layout=True
        )
        axes = np.atleast_2d(axes)

        # Column headers = models (top row)
        for c, model in enumerate(models):
            axes[0, c].set_title(str(model))

        # Compute vmin/vmax for each column (per-model scaling)
        col_norms: dict[str, LogNorm] = {}
        for c, model in enumerate(models):
            col_vals = gp.loc[gp["Model"] == model, "MSE"].to_numpy(dtype=float)
            col_vals = col_vals[np.isfinite(col_vals) & (col_vals > 0)]
            if col_vals.size == 0:
                for r in range(n_rows):
                    axes[r, c].set_visible(False)
                continue
            vmin = max(1e-12, float(np.nanmin(col_vals)))
            vmax = float(np.nanmax(col_vals))
            if not np.isfinite(vmax) or vmax <= 0:
                for r in range(n_rows):
                    axes[r, c].set_visible(False)
                continue
            if vmax <= vmin:
                vmax = vmin * 10.0
            col_norms[model] = LogNorm(vmin=vmin, vmax=vmax)

        # Track a mappable in each column
        last_col_im = {model: None for model in models}

        for r, method in enumerate(METHOD_ORDER):
            for c, model in enumerate(models):
                ax = axes[r, c]
                if model not in col_norms:
                    ax.set_visible(False)
                    continue

                gm = gp[(gp["Method"] == method) & (gp["Model"] == model)]
                if gm.empty:
                    ax.set_visible(False)
                    continue

                pivot = (
                    gm.pivot_table(index="k", columns="Budget", values="MSE", aggfunc="mean")
                      .sort_index(axis=0).sort_index(axis=1)
                )
                if pivot.size == 0 or pivot.isna().all().all():
                    ax.set_visible(False)
                    continue

                pivot = pivot.astype("float64")
                im = sns.heatmap(
                    pivot, ax=ax, cmap=GP_CMAP, norm=col_norms[model], cbar=False,
                    mask=pivot.isna(), linewidths=0,
                    yticklabels=1,
                )
                last_col_im[model] = im

                # Clean labeling
                if c == 0:
                    ax.set_ylabel("k")
                else:
                    ax.set_ylabel("")
                if r == n_rows - 1:
                    ax.set_xlabel("Budget")
                else:
                    ax.set_xlabel("")

                ax.invert_yaxis()

            right_ax = axes[r, -1]
            right_ax.text(
                1.12, 0.5, APPROACH_TO_TITLES.get(method, method),
                transform=right_ax.transAxes,
                rotation=-90, va="center", ha="right",
                fontsize=10, clip_on=False
            )


        # --- Add a horizontal colorbar for each column ---
        for c, model in enumerate(models):
            im = last_col_im.get(model)
            if im is None:
                continue
            fig.colorbar(
                im.collections[0],
                ax=axes[:, c].ravel().tolist(),
                orientation="horizontal",
                location="bottom",
                fraction=0.046,  # controls thickness
                pad=0.03,        # controls distance from the heatmaps
                label="Mean Squared Error"
            )

        fig.suptitle(SOURCES_TO_TITLES.get(problem, str(problem)), y=1.04, fontsize=12)
        heat_out_dir = out_dir / "heatmap"
        heat_out_dir.mkdir(parents=True, exist_ok=True)
        base = f"problem={problem}"
        fig.savefig(heat_out_dir / f"{base}.png", dpi=200, bbox_inches="tight")
        fig.savefig(heat_out_dir / f"{base}.pdf", bbox_inches="tight")
        plt.close(fig)


def run():
    print(f"Loading results from: {DIR_EST_RESULTS.resolve()}")
    df  = load_all_parquets(DIR_EST_RESULTS)
    df = df[(df["k"] >= 100)]

    print("Computing statistics...")
    mse = compute_mean_mse(df)

    print("Plotting heatmaps by problem...")
    plot_mse_heatmaps_by_problem(mse, DIR_PLOTS)

    print(f"Saved figures → {DIR_PLOTS.resolve()}")
