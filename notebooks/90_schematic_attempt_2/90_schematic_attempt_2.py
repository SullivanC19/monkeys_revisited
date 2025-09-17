from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import pandas as pd
import scipy.stats
import seaborn as sns
from typing import Any, Dict, List, Tuple

import src.plot
import src.utils

# Setup
data_dir, results_dir = src.utils.setup_notebook_dir(
    notebook_dir=os.path.dirname(os.path.abspath(__file__)),
    refresh=False,
)

# Parameters
alpha = 0.3
beta = 3.5
pass_at_1 = np.logspace(-5, 0, 10000)
palette = sns.color_palette("cool", n_colors=len(pass_at_1))
palette_dict = dict(zip(pass_at_1, palette))

# Create the data
beta_distributution_df = pd.DataFrame.from_dict(
    {
        r"$x$": pass_at_1,
        r"$\alpha$": np.full_like(pass_at_1, fill_value=alpha),
        r"$\beta$": np.full_like(pass_at_1, fill_value=beta),
        r"$p(x)$": scipy.stats.beta.pdf(pass_at_1, alpha + 1, beta),
    }
)

# Setup for pass@k calculations
select_pass_at_1_values = pass_at_1[:: (len(pass_at_1) // 5)]
select_palette = palette[:: (len(pass_at_1) // 5)]
k = np.linspace(1, 10000, 10000)
pass_at_k = 1.0 - np.power(
    1.0 - select_pass_at_1_values.reshape(-1, 1), k.reshape(1, -1)
)
neg_log_pass_at_k = -np.log(pass_at_k)
# Create schematic figure.
plt.close()
fig, axes = plt.subplots(
    figsize=(18, 12),  # Slightly wider figure
    nrows=2,
    ncols=2,
    sharex=False,
    sharey=False,
)
fig.subplots_adjust(wspace=1.0, hspace=1.0)  # Increased spacing between subplots
# First subplot - Beta distribution
ax1 = axes[0, 0]
ax1.plot(k, np.power(beta / k, alpha), color="k", linewidth=5)
ax1.set(
    xscale="log",
    xlim=(1, k.max()),
    yscale="log",
    ylim=(3.162e-2, 1.5e1),
    xlabel=r"Number of Attempts $k$",
    ylabel=r"$-\log(\operatorname{pass_{\mathcal{D}}@k})$",
)
# Second subplot - Pass@k for different pass@1 values
ax2 = axes[1, 0]
for idx, select_pass_at_1 in enumerate(select_pass_at_1_values):
    ax2.plot(
        k,
        neg_log_pass_at_k[idx],
        label=f"pass@1={select_pass_at_1:.2e}",
        color=select_palette[idx],
        linewidth=5,
    )
    # Add 'X' marker at x=1 for each line
    ax2.scatter(
        1.0,  # x value
        neg_log_pass_at_k[idx][0],  # y value at k=1
        marker="d",  # X marker
        s=200,  # size of marker
        color=select_palette[idx],  # same color as line
        zorder=5,  # ensure markers appear on top
    )
ax2.set(
    xscale="log",
    yscale="log",
    xlim=(1.0, k.max()),
    ylim=(3.162e-2, 1.5e1),
    xlabel=r"Number of Attempts $k$",
    ylabel=r"$-\log (\operatorname{pass_i@k})$",
)
# Third subplot - Expected pass@k
ax3 = axes[1, 1]
positive_prob_beta_distribution = beta_distributution_df[
    beta_distributution_df[r"$p(x)$"] > 0
]
sns.scatterplot(
    data=positive_prob_beta_distribution,
    x=r"$x$",
    y=r"$p(x)$",
    hue=r"$x$",
    legend=False,
    palette="cool",
    hue_norm=LogNorm(),
    linewidth=0,
    ax=ax3,
)
min_y = positive_prob_beta_distribution[r"$p(x)$"].min()
for idx, select_pass_at_1 in enumerate(select_pass_at_1_values):
    # Add 'X' marker at x=1 for each line
    ax3.scatter(
        np.exp(-neg_log_pass_at_k[idx][0]),  # y value at k=1,
        3.162e-2,
        marker="d",  # X marker
        s=200,  # size of marker
        color=select_palette[idx],  # same color as line
        zorder=5,  # ensure markers appear on top
        # linewidth=2  # marker edge width
    )
ax3.set(
    xscale="log",
    xlim=(1e-5, 1.05),
    xlabel=r"$\operatorname{pass_i@1}$",
    ylim=(3.162e-2, 1.5e1),
    ylabel=r"$p_{\mathcal{D}}(\operatorname{pass_i@1})$",
    yscale="log",
)
# Create text in axes.
fig.text(
    0.33,
    0.87,
    r"$-\log(\operatorname{pass_{\mathcal{D}}@k}) \propto k^{-b}$",
    fontsize=40,
    ha="center",
    va="center",
)
fig.text(
    0.76,
    0.41,
    r"$p_{\mathcal{D}}(\operatorname{pass_i@k}) \propto (\operatorname{pass_i@k})^{b-1}$",
    fontsize=40,
    ha="center",
    va="center",
)
# Create arrows between axes.
fig.text(0.13, 0.52, r"$\Downarrow$", fontsize=40, ha="center", va="center")
fig.text(0.44, 0.52, r"$\Downarrow$", fontsize=40, ha="center", va="center")
fig.text(0.51, 0.16, r"$\Rightarrow$", fontsize=40, ha="center", va="center")
fig.text(0.51, 0.38, r"$\Rightarrow$", fontsize=40, ha="center", va="center")
fig.text(0.61, 0.52, r"$\Uparrow$", fontsize=40, ha="center", va="center")
fig.text(0.93, 0.52, r"$\Uparrow$", fontsize=40, ha="center", va="center")
# Add A, B, C, D labels counterclockwise
# Position the labels slightly outside the plots
fig.text(0.01, 0.98, "A", fontweight="bold")  # Top left
fig.text(0.01, 0.48, "B", fontweight="bold")  # Bottom left
fig.text(0.51, 0.48, "C", fontweight="bold")  # Bottom right
fig.text(0.51, 0.98, "D", fontweight="bold")  # Top right
# Save the combined plot
src.plot.save_plot_with_multiple_extensions(
    plot_dir=results_dir, plot_filename="combined_statistical_plots"
)
plt.show()
print("Finished0 notebooks/90_schematic_attempt_2!")
