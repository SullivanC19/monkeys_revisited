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

from constants import DIR_RESULTS, DIR_PLOTS, SOURCES_TO_TITLES, APPROACH_TO_TITLES
from utilities import sanitize  # for filenames

# --- theme ---
sns.set_theme(style="whitegrid")
sns.set_context("paper", font_scale=0.85)

METHOD_ORDER = ["Regression", "Discretization", "Dynamic"]
METHOD_COLS  = [f"{m} Estimate" for m in METHOD_ORDER]
GP_CMAP = cm.get_cmap("Purples_r")

ID_COLS  = ["Problem", "Model", "k", "Budget"]
TRUE_COL = "True Pass@k"

def _slug(s: str, sep: str = "-") -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", sep, s).strip(sep) or "x"

def load_all_parquets(in_dir: Path) -> pd.DataFrame:
    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {in_dir}")
    df = pd.concat([pd.read_parquet(p, engine="pyarrow") for p in files], ignore_index=True)
    for c in ("Budget","k"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

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

def _pivot_mse(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["Budget"] = pd.to_numeric(g["Budget"], errors="coerce")
    g["k"] = pd.to_numeric(g["k"], errors="coerce")
    pivot = (g.pivot_table(index="k", columns="Budget", values="MSE", aggfunc="mean")
               .sort_index(axis=0)
               .sort_index(axis=1))
    return pivot.astype("float64")

def plot_mse_heatmaps_by_problem(mse: pd.DataFrame, out_dir: Path, model_order: list[str] | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    mse = mse.copy()
    mse["Method"] = pd.Categorical(mse["Method"], METHOD_ORDER, ordered=True)

    for problem, gp in mse.groupby("Problem", dropna=False):
        # facet transpose: rows = models, cols = methods
        models = (sorted(gp["Model"].dropna().unique().tolist())
                  if model_order is None else model_order)
        n_rows, n_cols = len(models), len(METHOD_ORDER)
        if n_rows == 0 or n_cols == 0:
            continue

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(3.6*n_cols, 3.0*n_rows),
            sharex=True, sharey=True, constrained_layout=True
        )
        axes = np.atleast_2d(axes)

        # Column headers = methods (top row)
        for c, method in enumerate(METHOD_ORDER):
            axes[0, c].set_title(APPROACH_TO_TITLES[method])

        for r, model in enumerate(models):
            # --- per-row scaling: collect all valid MSE values for this model across all methods ---
            row_vals = gp.loc[gp["Model"] == model, "MSE"].to_numpy(dtype=float)
            row_vals = row_vals[np.isfinite(row_vals) & (row_vals > 0)]
            if row_vals.size == 0:
                # hide whole row if nothing to plot
                for c in range(n_cols):
                    axes[r, c].set_visible(False)
                continue

            vmin = max(1e-12, float(np.nanmin(row_vals)))
            vmax = float(np.nanmax(row_vals))
            if not np.isfinite(vmax) or vmax <= 0:
                for c in range(n_cols):
                    axes[r, c].set_visible(False)
                continue
            if vmax <= vmin:
                vmax = vmin * 10.0
            row_norm = LogNorm(vmin=vmin, vmax=vmax)

            last_row_im = None  # track a mappable for the row colorbar

            for c, method in enumerate(METHOD_ORDER):
                ax = axes[r, c]
                gm = gp[(gp["Method"] == method) & (gp["Model"] == model)]
                if gm.empty:
                    ax.set_visible(False)
                    continue

                pivot = (gm.pivot_table(index="k", columns="Budget", values="MSE", aggfunc="mean")
                           .sort_index(axis=0).sort_index(axis=1))
                # ensure numeric + mask NaNs
                if pivot.size == 0 or pivot.isna().all().all():
                    ax.set_visible(False)
                    continue
                pivot = pivot.astype("float64")

                im = sns.heatmap(
                    pivot, ax=ax, cmap=GP_CMAP, norm=row_norm, cbar=False,
                    mask=pivot.isna(), linewidths=0
                )
                last_row_im = im

                # axes labels: only put y on first col, x on bottom row to avoid clutter
                if c == 0:
                    ax.set_ylabel("k")
                else:
                    ax.set_ylabel("")
                if r == n_rows - 1:
                    ax.set_xlabel("Budget")
                else:
                    ax.set_xlabel("")

            # --- add a right-side colorbar for this row only ---
            if last_row_im is not None:
                # attach the colorbar to all axes in the row so its height matches the row
                fig.colorbar(
                    last_row_im.collections[0],
                    ax=axes[r, :].ravel().tolist(),
                    location="right",
                    fraction=0.046,
                    pad=0.04,
                    label="Mean Squared Error"
                )

            # --- add the row label (model) on the left margin of the first visible cell ---
            left_ax = None
            for c in range(n_cols):
                if axes[r, c].get_visible():
                    left_ax = axes[r, c]
                    break
            if left_ax is not None:
                left_ax.text(
                    -0.20, 0.5, str(model),
                    transform=left_ax.transAxes,
                    rotation=90, va="center", ha="right",
                    fontsize=11, color="black", clip_on=False
                )

        fig.suptitle(SOURCES_TO_TITLES.get(problem, str(problem)), y=1.02, fontsize=12)
        base = f"mse_heat_by_problem__{sanitize(problem)}__rows-models_cols-methods"
        fig.savefig(out_dir / f"{base}.png", dpi=200, bbox_inches="tight")
        fig.savefig(out_dir / f"{base}.pdf", bbox_inches="tight")
        plt.close(fig)


def run():
    print(f"Loading results from: {DIR_RESULTS.resolve()}")
    df  = load_all_parquets(DIR_RESULTS)
    mse = compute_mean_mse(df)
    plot_mse_heatmaps_by_problem(mse, DIR_PLOTS)
    print(f"Saved figures → {DIR_PLOTS.resolve()}")
