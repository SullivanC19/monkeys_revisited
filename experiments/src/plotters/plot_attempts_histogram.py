import math
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from constants import DIR_ATT_RESULTS, DIR_PLOTS, SOURCES_TO_TITLES, SAMPLER_TO_TITLES
from utilities import load_all_parquets

GREEN  = sns.color_palette("Greens", 4)[1]
PURPLE = sns.color_palette("Purples", 5)[3]
GRAY  = sns.color_palette("Greys", 4)[2]

PALETTE = {
    "Uniform": GREEN,
    "Dynamic": PURPLE,
    "Optimal": GRAY,
}

PROBLEM_COL   = "Problem"
MODEL_COL     = "Model"
BUDGET_COL    = "Budget"
SUBP_COL      = "Subproblems"
HARDNESS_COL  = "Hardness"
DYN_ATT_COL   = "Dynamic Attempts"

def _as_pylist(v) -> list | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        if hasattr(v, "to_pylist"):
            return v.to_pylist()
    except Exception:
        pass
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    return None

def get_optimal_allocation(hardness: list[float], k: int=1000, budget: int=1000) -> list[float]:
    r_hardness = 1 - np.array(hardness)
    log_hardness = np.log(np.clip(r_hardness, 1e-8, 1 - 1e-8))
    log_weights = ((2 * k - 1) * log_hardness + np.log(1 - np.exp(log_hardness))) * 1 / 2
    log_weights -= np.max(log_weights)  # numerical stability
    weights = np.exp(log_weights - np.max(log_weights)).tolist()
    total = sum(weights)
    return [budget * (w / total) for w in weights]

def _expand_row_to_long(row: pd.Series) -> pd.DataFrame:
    problem = row.get(PROBLEM_COL)
    model   = row.get(MODEL_COL)
    budget  = row.get(BUDGET_COL)
    n_subp  = row.get(SUBP_COL)

    hardness = _as_pylist(row.get(HARDNESS_COL))
    dyn_atts = _as_pylist(row.get(DYN_ATT_COL))
    opt_atts = _as_pylist(get_optimal_allocation(hardness, budget=float(budget))) if hardness is not None else None

    if hardness is None or dyn_atts is None or n_subp is None or budget is None or opt_atts is None:
        return pd.DataFrame(columns=[PROBLEM_COL, MODEL_COL, BUDGET_COL, "Hardness",
                                     "Uniform Weight", "Dynamic Weight", "Optimal Weight"])

    L = min(len(hardness), len(dyn_atts), len(opt_atts), int(n_subp) if pd.notna(n_subp) else 0)
    if L <= 0:
        return pd.DataFrame(columns=[PROBLEM_COL, MODEL_COL, BUDGET_COL, "Hardness",
                                     "Uniform Weight", "Dynamic Weight", "Optimal Weight"])

    uniform_weight = float(budget) / float(n_subp) if float(n_subp) > 0 else 0.0

    out = pd.DataFrame({
        PROBLEM_COL: [problem] * L,
        MODEL_COL:   [model]   * L,
        BUDGET_COL:  [budget]  * L,
        "Hardness":        [float(hardness[i]) for i in range(L)],
        "Uniform Weight":  [uniform_weight]     * L,
        "Dynamic Weight":  [float(dyn_atts[i]) for i in range(L)],
        "Optimal Weight":  [float(opt_atts[i]) for i in range(L)],
    })
    return out[np.isfinite(out["Hardness"].to_numpy())]

def build_long_attempts(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, row in df.iterrows():
        piece = _expand_row_to_long(row)
        if not piece.empty:
            pieces.append(piece)
    if not pieces:
        return pd.DataFrame(columns=[PROBLEM_COL, MODEL_COL, BUDGET_COL,
                                     "Hardness", "Uniform Weight", "Dynamic Weight", "Optimal Weight"])
    out = pd.concat(pieces, ignore_index=True)
    out[BUDGET_COL] = pd.to_numeric(out[BUDGET_COL], errors="coerce", downcast="integer")
    for c in ["Hardness", "Uniform Weight", "Dynamic Weight", "Optimal Weight"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _hist_two_layers(ax: plt.Axes, data: pd.DataFrame, bins: int = 10):
    sns.histplot(
        data=data, x="Hardness", weights="Uniform Weight", log_scale=True,
        bins=bins, color=PALETTE["Uniform"], alpha=0.5, stat="percent", element="step", ax=ax
    )
    sns.histplot(
        data=data, x="Hardness", weights="Dynamic Weight", log_scale=True,
        bins=bins, color=PALETTE["Dynamic"], alpha=0.5, stat="percent", element="step", ax=ax
    )
    sns.histplot(
        data=data, x="Hardness", weights="Optimal Weight", log_scale=True,
        bins=bins, color=PALETTE["Optimal"], alpha=1, stat="percent", element="step",
        fill=False, linewidth=0.8, ax=ax, zorder=10, linestyle="--"
    )
    ax.set_xlabel("Hardness (log scale)")
    ax.set_ylabel("% Attempts Allocated")
    ax.set_ylim(bottom=0)

def _legend(fig: plt.Figure):
    """Make a shared legend using APPROACH_TO_TITLES and PALETTE."""
    approaches = ["Uniform", "Dynamic", "Optimal"]
    handles = [
        Line2D([0], [0], color=PALETTE[a], linewidth=0.8 if a == "Optimal" else 2, linestyle="--" if a == "Optimal" else "-", label=a)
        for a in approaches
    ]
    labels = [SAMPLER_TO_TITLES.get(a, a) for a in approaches]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))

def plot_attempts_histograms_per_problem_budget(
    long_df: pd.DataFrame,
    out_dir: Path,
    col_wrap: int = 3,
    bins: int = 10,
):
    """
    Create ONE histogram figure per (Problem, Budget), with facets (subplots) per Model.
    """
    out_dir = Path(out_dir) / "attempts_over_hardness"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = long_df.copy()
    if df.empty:
        print("No rows to plot after filtering."); return

    for (problem, budget), gp in df.groupby([PROBLEM_COL, BUDGET_COL], dropna=False):
        models = gp[MODEL_COL].dropna().unique().tolist()
        if not models:
            continue

        # keep a full grid (optional); otherwise drop this trimming
        if col_wrap > 0 and len(models) >= col_wrap:
            keep = models[: (len(models) // col_wrap) * col_wrap]
            if keep:
                gp = gp[gp[MODEL_COL].isin(keep)]
                models = keep

        n = len(models)
        ncols = min(col_wrap, n) if col_wrap else n
        nrows = math.ceil(n / ncols) if ncols else 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 3.2*nrows), squeeze=False)
        axes = axes.flatten()

        for i, model in enumerate(models):
            ax = axes[i]
            gpm = gp[gp[MODEL_COL] == model]
            if gpm.empty:
                ax.axis("off"); continue
            _hist_two_layers(ax, gpm, bins=bins)
            ax.set_title(str(model))

        # hide any unused axes
        for j in range(len(models), len(axes)):
            axes[j].axis("off")

        problem_title = SOURCES_TO_TITLES.get(problem, str(problem))
        fig.suptitle(f"Attempt Allocation for {problem_title}", y=0.98, fontsize=14)

        # Add shared legend at the bottom
        _legend(fig)

        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        base = f"attempts_hist-problem={problem}-budget={int(budget)}"
        fig.savefig(out_dir / f"{base}.png", dpi=200, bbox_inches="tight")
        fig.savefig(out_dir / f"{base}.pdf", dpi=200, bbox_inches="tight")
        plt.close(fig)

def run():
    print(f"Scanning results → {DIR_ATT_RESULTS.resolve()}")
    df = load_all_parquets(DIR_ATT_RESULTS)
    if df.empty:
        print("No rows found in results directory."); return

    print("Expanding rows to per-subproblem records...")
    long_df = build_long_attempts(df)
    if long_df.empty:
        print("No per-subproblem records were constructed (check Hardness / Dynamic Attempts)."); return

    print("Plotting histograms per (Problem, Budget)...")
    plot_attempts_histograms_per_problem_budget(
        long_df=long_df,
        out_dir=DIR_PLOTS,
        col_wrap=3,
        bins=10,
    )

    print(f"Saved figures → {(Path(DIR_PLOTS) / 'attempts_over_hardness').resolve()}")

if __name__ == "__main__":
    run()
