from monkeys import analyze
from pathlib import Path
import numpy as np
from math import log10
import os
import seaborn as sns

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

sns.set_theme(style="whitegrid")
sns.set_context("paper")

DATA_SOURCES = [
    ('jailbreaking', analyze.create_or_load_bon_jailbreaking_text_individual_outcomes_df),
    ('code_contests', analyze.create_or_load_large_language_monkeys_code_contests_individual_outcomes_df),
    ('math', analyze.create_or_load_large_language_monkeys_pythia_math_individual_outcomes_df),
]

SOURCES_TO_TITLES = {
    'jailbreaking': 'AdvBench',
    'math': 'MATH',
    'code_contests': 'Code Contests',
}

APPROACH_TO_TITLES = {
    "Regression": "Regression",
    "Discretization": "Discretized Beta [Schaeffer et al.]",
    "Dynamic": "Dynamic + Beta-Binomial [Ours]",
}

SAMPLER_TO_TITLES = {
    "Dynamic": "Dynamic Allocation [Ours]",
    "Uniform": "Uniform Allocation",
    "Optimal": "Optimal Allocation [Oracle]",
}

K_VALUES = np.logspace(1, 3, 40, dtype=int, base=10).tolist()
BUDGET_VALUES = np.logspace(2, 4, 20, dtype=int, base=10).tolist()
N_TRIALS = 100
SEEDS = list(range(N_TRIALS))

DIR_EST_RESULTS = Path(__file__).parent.parent / 'results' / 'estimates'
DIR_ATT_RESULTS = Path(__file__).parent.parent / 'results' / 'attempts'
DIR_SYN_RESULTS = Path(__file__).parent.parent / 'results' / 'synthetic'
DIR_EST_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_ATT_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_SYN_RESULTS.mkdir(parents=True, exist_ok=True)

DIR_PLOTS = Path(__file__).parent.parent / 'figs'
DIR_PLOTS.mkdir(exist_ok=True)