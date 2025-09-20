from monkeys import analyze
from pathlib import Path

DATA_SOURCES = [
    ('jailbreaking', analyze.create_or_load_bon_jailbreaking_text_pass_at_k_df),
    ('math', analyze.create_or_load_large_language_monkeys_pythia_math_pass_at_k_df),
    ('code_contests', analyze.create_or_load_large_language_monkeys_code_contests_pass_at_k_df),
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

K_VALUES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
BUDGET_VALUES = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

N_TRIALS = 100
SEEDS = list(range(N_TRIALS))

DIR_RESULTS = Path(__file__).parent.parent / 'results'
DIR_RESULTS.mkdir(exist_ok=True)

DIR_PLOTS = Path(__file__).parent.parent / 'figs'
DIR_PLOTS.mkdir(exist_ok=True)