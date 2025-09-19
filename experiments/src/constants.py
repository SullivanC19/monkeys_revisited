from monkeys import analyze
from pathlib import Path

DATA_SOURCES = [
    ('jailbreaking', analyze.create_or_load_bon_jailbreaking_text_pass_at_k_df),
    ('math', analyze.create_or_load_large_language_monkeys_pythia_math_pass_at_k_df),
    ('code_contests', analyze.create_or_load_large_language_monkeys_code_contests_pass_at_k_df),
]

K_VALUES = [100, 200]
BUDGET_VALUES = [1000, 2000]

N_TRIALS = 3
SEEDS = list(range(N_TRIALS))

DIR_RESULTS = Path(__file__).parent.parent / 'results'
DIR_RESULTS.mkdir(exist_ok=True)