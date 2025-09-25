import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from monkeys.analyze import (
    fit_discretized_beta_three_parameters_to_num_samples_and_num_successes,
    fit_beta_binomial_two_parameters_to_num_samples_and_num_successes,
)

PURPLE = sns.color_palette("Purples", 5)[3]
GREEN = sns.color_palette("Greens", 4)[2]
GRAY = sns.color_palette("Greys", 3)[1]

from constants import DIR_PLOTS

def _beta_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Beta(a,b) pdf on [0,1] computed stably via log-space.
    """
    x = np.clip(x, 1e-12, 1 - 1e-12)  # avoid log(0)
    logB = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return np.exp((alpha - 1) * np.log(x) + (beta - 1) * np.log(1 - x) - logB)

def run():
    n_samples = 10000
    b = 100

    rng = np.random.default_rng(0)
    sampled_hardness = rng.uniform(0, 1, n_samples)

    # draw b Bernoulli trials per item
    samples = rng.uniform(0, 1, (n_samples, b)) < sampled_hardness[:, None]
    n_successes = samples.sum(axis=1)

    df = pd.DataFrame({
        "Num. Samples Total": np.full(n_samples, b),
        "Num. Samples Correct": n_successes,
    })

    # ---- Fit both models ----
    params_dsc = fit_discretized_beta_three_parameters_to_num_samples_and_num_successes(df)
    alpha_dsc, beta_dsc, scale_dsc = (
        float(params_dsc['alpha']),
        float(params_dsc['beta']),
        float(params_dsc['scale'])
    )

    params_our = fit_beta_binomial_two_parameters_to_num_samples_and_num_successes(df)
    alpha_our, beta_our = float(params_our['alpha']), float(params_our['beta'])

    # ---- Plot ----
    sns.set_theme(style="whitegrid")
    sns.set_context("paper")

    fig, ax = plt.subplots(figsize=(6, 4))

    # Histogram of true latent hardness
    sns.histplot(
        sampled_hardness,
        bins=20,
        stat="density",
        alpha=0.35,
        edgecolor="none",
        ax=ax,
        color=GRAY,
        label="True sampled hardness"
    )

    # x domain: match the three-parameter scale
    x = np.linspace(1e-3, 1-1e-3, 500)

    # Discretized Beta pdf scaled to [0, scale_dsc]
    y_dsc = (1/scale_dsc) * _beta_pdf(x / scale_dsc, alpha_dsc, beta_dsc)
    ax.plot(x, y_dsc, color=GREEN, linewidth=1, linestyle='-', label=f"Discretized fit (α={alpha_dsc:.3f}, β={beta_dsc:.3f})")

    # Our two-parameter Beta pdf (assumes support [0,1])
    y_our = _beta_pdf(x, alpha_our, beta_our)
    ax.plot(x, y_our, color=PURPLE, linewidth=1, linestyle='--', label=f"Our fit (α={alpha_our:.3f}, β={beta_our:.3f})")

    # Labels and legend
    ax.set_xlabel("Latent success probability")
    ax.set_ylabel("Density")
    ax.set_title("Comparison of Beta Fits")
    ax.legend(frameon=False)

    plt.tight_layout()
    out_dir = DIR_PLOTS / "syn"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"distribution-beta_sampling"
    fig.savefig(out_dir / f"{base}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / f"{base}.pdf", bbox_inches="tight")
