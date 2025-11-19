import numpy as np
import matplotlib.pyplot as plt

from constants import DATA_SOURCES, DIR_PLOTS
from utilities import (
    extract_models_with_samples,
    sanitize,
)

def run():

    if not (DIR_PLOTS / "hardness").exists():
        (DIR_PLOTS / "hardness").mkdir()

    for problem, gen in DATA_SOURCES:
        for model, samples in extract_models_with_samples(gen()):
            avgs = [sum(s) / len(s) for s in samples]
            frac_zeros = sum(1 for a in avgs if a == 0.0) / len(avgs)
            print(f"Problem: {problem}, Model: {model}, Subtasks: {len(samples)}, Impossible Subtasks: {frac_zeros}")

            print(sum(1 for a in avgs if a >= 0.01) / len(avgs))

            avgs_nonzero = [a for a in avgs if a > 0.0]

            plt.figure(figsize=(6, 4))
            plt.hist(np.log10(avgs_nonzero), bins=30, color='blue', alpha=0.7)
            plt.title(f"Histogram of Log-Scale Subtask Hardness\n{problem} - {model}")
            plt.xlabel("Log10(Subtask Hardness)")
            plt.ylabel("Frequency")
            
            # Add 'impossible task' count annotation
            plt.annotate(f"Impossible Subtasks: {100 * frac_zeros:.2f}%", xy=(0.7, 0.9), xycoords='axes fraction', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5))

            plt.grid(axis='y', alpha=0.75)
            plt.tight_layout()


            plt.savefig(DIR_PLOTS / "hardness" / f"histogram_{sanitize(problem)}_{sanitize(model)}.png")
            plt.close()
            print(f"Saved histogram for Problem: {problem}, Model: {model}")
