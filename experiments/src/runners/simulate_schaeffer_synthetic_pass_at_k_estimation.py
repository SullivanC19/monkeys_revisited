from itertools import product

import numpy as np
import pandas as pd

from tqdm import tqdm

from constants import BUDGET_VALUES, SEEDS, DIR_SYN_RESULTS

from .simulate_pass_at_k_estimation import (
    simulate_estimate_of_pass_at_k,
    simulate_discretization_estimate_of_pass_at_k,
    compute_true_pass_at_k,
)


def create_synthetic_samples(
    distribution: str,
    alpha: float,
    beta: float,
    scale: float,
    n_subtasks: int,
    n_samples: int=10000,
    seed: int=0,
) -> tuple[np.ndarray, list[list[bool]]]:
    """Generate problem success probabilities and repeated outcomes."""
    problem_seed, sample_seed = np.random.SeedSequence([seed, n_subtasks]).spawn(2)
    rng_p = np.random.default_rng(problem_seed)
    rng = np.random.default_rng(sample_seed)

    if distribution == "beta":
        probabilities = scale * rng_p.beta(alpha, beta, size=n_subtasks)
    else:
        u = rng_p.uniform(0, 1, n_subtasks)
        probabilities = scale * (
            -np.expm1(np.log1p(-u) / beta)
        ) ** (1 / alpha)

    samples = []
    for p in probabilities:
        subtask_samples = rng.uniform(0, 1, n_samples) < p
        samples.append(subtask_samples.tolist())
    return probabilities, samples


def run():
    k = 1000
    n_samples = 10000
    output_dir = DIR_SYN_RESULTS / "schaeffer"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running Schaeffer synthetic distribution simulations...")

    settings = list(product(
        ["beta", "kumaraswamy"],
        [0.15, 0.4],
        [3.5, 5.0],
        [0.1, 0.8],
        [64, 128],
    ))

    for distribution, alpha, beta, scale, n_subtasks in tqdm(settings):
        res = []
        for seed in SEEDS[:30]:
            probabilities, samples = create_synthetic_samples(
                distribution=distribution,
                alpha=alpha,
                beta=beta,
                scale=scale,
                n_subtasks=n_subtasks,
                n_samples=n_samples,
                seed=seed,
            )
            true_pass_at_k = compute_true_pass_at_k(probabilities, [k])[0]

            for budget in BUDGET_VALUES:
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
                    "Distribution": distribution,
                    "Alpha": alpha,
                    "Beta": beta,
                    "Scale": scale,
                    "Subproblems": n_subtasks,
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
        filename = f"{distribution}-a={alpha}-b={beta}-scale={scale}-m={n_subtasks}.parquet"
        df.to_parquet(output_dir / filename, engine="pyarrow", index=False)
        print(f"Done! Results saved to {filename}")


if __name__ == "__main__":
    run()
