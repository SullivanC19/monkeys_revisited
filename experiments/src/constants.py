from monkeys import analyze
from pathlib import Path
import numpy as np
from math import log10
import os

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

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

K_VALUES = list(range(10, 1001, 10))
BUDGET_VALUES = np.logspace(log10(200), 4, 30, dtype=int, base=10).tolist()  # 200 to 10_000 --> need at least 1 per problem
N_TRIALS = 3
SEEDS = list(range(N_TRIALS))

DIR_RESULTS = Path(__file__).parent.parent / 'results'
DIR_RESULTS.mkdir(exist_ok=True)

DIR_PLOTS = Path(__file__).parent.parent / 'figs'
DIR_PLOTS.mkdir(exist_ok=True)