import numpy as np
import pandas as pd

from tqdm import tqdm

from constants import BUDGET_VALUES, SEEDS, DIR_SYN_RESULTS

from .simulate_pass_at_k_estimation import (
    simulate_estimate_of_pass_at_k,
    simulate_discretization_estimate_of_pass_at_k,
    compute_hardness,
    compute_true_pass_at_k,
)


def create_synthetic_samples(n_subtasks: int, n_samples: int=10000, k: int=1000, seed: int=0) -> list[list[bool]]:
    """
    Create synthetic samples for a problem with n_subtasks.
    Each subtask has n_samples attempts, with success probability determined by hardness.
    Returns a list of lists of booleans indicating success (True) or failure (False).
    """
    rng = np.random.default_rng(seed)
    samples = []
    n_hard = int(n_subtasks / 2)
    n_easy = n_subtasks - n_hard
    p_success = [0] * n_hard + [0.3] * n_easy
    for p in p_success:
        subtask_samples = rng.uniform(0, 1, n_samples) < p
        samples.append(subtask_samples.tolist())
    return samples

def create_uar_synthetic_samples(n_subtasks: int, n_samples: int=10000, k: int=1000, seed: int=0) -> list[list[bool]]:
    """
    Create synthetic samples for a problem with n_subtasks.
    Each subtask has n_samples attempts, with success probability drawn uniformly at random.
    Returns a list of lists of booleans indicating success (True) or failure (False).
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_subtasks):
        p = rng.uniform(0, 1)
        subtask_samples = rng.uniform(0, 1, n_samples) < p
        samples.append(subtask_samples.tolist())
    return samples

def run():
    k = 1000
    n_samples = 10000
    n_subtasks = 64

    print("Running synthetic best-case scenario simulation...")

    # samples = create_synthetic_samples(n_subtasks=n_subtasks, n_samples=n_samples, k=k)
    samples = create_uar_synthetic_samples(n_subtasks=n_subtasks, n_samples=n_samples, k=k)
    hardness = compute_hardness(samples)

    res = []
    for budget in tqdm(BUDGET_VALUES):
        for seed in SEEDS[:20]:
            true_pass_at_k = compute_true_pass_at_k(hardness, [k])[0]
            dynamic_estimates, _, _ = simulate_estimate_of_pass_at_k(
                samples=samples,
                budget=budget,
                k_values=[k],
                dynamic=True,
                seed=seed,
            )
            uniform_estimates, _, _ = simulate_estimate_of_pass_at_k(
                samples=samples,
                budget=budget,
                k_values=[k],
                dynamic=False,
                seed=seed,
            )
            uniform_discretization_estimates, _, _ = simulate_discretization_estimate_of_pass_at_k(
                samples=samples,
                budget=budget,
                k_values=[k],
                dynamic=False,
                seed=seed,
            )
            dynamic_discretization_estimates, _, _ = simulate_discretization_estimate_of_pass_at_k(
                samples=samples,
                budget=budget,
                k_values=[k],
                dynamic=True,
                seed=seed,
            )
            res.append({
                "k": k,
                "Seed": seed,
                "Budget": budget,
                "True Pass@k": true_pass_at_k,
                "Uniform Estimate": uniform_estimates[0],
                "Dynamic Estimate": dynamic_estimates[0],
                "Discretization Estimate": uniform_discretization_estimates[0],
                "Dynamic Discretization Estimate": dynamic_discretization_estimates[0],
            })

    df = pd.DataFrame(res)
    df.to_parquet(DIR_SYN_RESULTS / "synthetic_best_case_simulation.parquet", engine="pyarrow", index=False)
    print("Done! Results saved to synthetic_best_case_simulation.parquet")