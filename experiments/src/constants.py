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
    ('math', analyze.create_or_load_large_language_monkeys_pythia_math_individual_outcomes_df),
    ('code_contests', analyze.create_or_load_large_language_monkeys_code_contests_individual_outcomes_df),
]

SOURCES_TO_TITLES = {
    'jailbreaking': 'BON Jailbreaking',
    'math': 'MATH',
    'code_contests': 'Code Contests',
}

APPROACH_TO_TITLES = {
    "Regression": "Linear Regression [OpenAI]",
    "Discretization": "Uniform + Discretized Beta [Schaeffer et al. 2025]",
    "Dynamic": "Dynamic + Beta-Binomial [Ours]",
}

SAMPLER_TO_TITLES = {
    "Dynamic": "Dynamic Allocation [Ours]",
    "Uniform": "Uniform Allocation",
    "Optimal": "Optimal Allocation [Oracle]",
}

K_VALUES = np.logspace(0, 4, 100, dtype=int, base=10).tolist()
BUDGET_VALUES = np.logspace(log10(200), 4, 30, dtype=int, base=10).tolist()
N_TRIALS = 3
SEEDS = list(range(N_TRIALS))

DIR_EST_RESULTS = Path(__file__).parent.parent / 'results' / 'estimates'
DIR_ATT_RESULTS = Path(__file__).parent.parent / 'results' / 'attempts'
DIR_EST_RESULTS.mkdir(parents=True, exist_ok=True)
DIR_ATT_RESULTS.mkdir(parents=True, exist_ok=True)

DIR_PLOTS = Path(__file__).parent.parent / 'figs'
DIR_PLOTS.mkdir(exist_ok=True)