from itertools import chain
import re
import unicodedata
import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
from pathlib import Path

from constants import DATA_SOURCES, BUDGET_VALUES, K_VALUES, SEEDS, DIR_RESULTS
from utilities import (
    extract_models_with_subtask_hardness,
    attempt,
    evenly_distribute_budget,
    sanitize,
)

from monkeys.analyze import (
    estimate_pass_at_k,
    fit_discretized_beta_three_parameters_to_num_samples_and_num_successes,
    fit_beta_binomial_two_parameters_to_num_samples_and_num_successes,
)
from monkeys.EM import compute_estimate

def compute_true_pass_at_k(hardness: np.ndarray, k: int) -> float:
    """
    Compute the true pass@k value given an array of subtask hardness values.

    Args:
        hardness (np.ndarray): An array of subtask hardness values (between 0 and 1).
        k (int): The number of successful attempts required to pass.

    Returns:
        float: The true pass@k value.
    """
    return 1 - np.mean((1 - hardness) ** k)

def simulate_regression_estimate_of_pass_at_k(subtask_hardness: np.ndarray, budget: int, k: int) -> float:
    """
    Simulate the regression estimate of pass@k given subtask hardness, budget, and k.

    Args:
        subtask_hardness (np.ndarray): An array of subtask hardness values (between 0 and 1).
        budget (int): The total number of attempts allowed.
        k (int): The number of successful attempts required to pass.

    Returns:
        float: The regression estimate of pass@k.
    """

    # Uniformly distribute sampling budget across subproblems
    n_subproblems = len(subtask_hardness)
    problem_budgets = evenly_distribute_budget(budget, n_subproblems)
    n_successes = attempt(subtask_hardness, n=problem_budgets)
    if np.sum(n_successes) == 0:
        return 0.0, problem_budgets, n_successes

    # Estimate pass@k for small k values
    small_ks = []
    estimated_pass_at_small_ks = []
    for small_k in range(1, budget // n_subproblems):
        est = np.mean(estimate_pass_at_k(problem_budgets, n_successes, small_k))
        small_ks.append(small_k)
        estimated_pass_at_small_ks.append(est)

    # Extrapolate to desired k using linear regression on log-log scale
    X = np.log(np.array(small_ks, dtype=np.float64)).reshape(-1, 1)
    with np.errstate(divide='ignore'):
        y = -np.log(np.array(estimated_pass_at_small_ks, dtype=np.float64))
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    estimate = np.minimum(np.exp(-model.predict(np.log(np.array(k)).reshape(1, -1))), 1).item()

    return estimate, problem_budgets, n_successes

def simulate_discretization_estimate_of_pass_at_k(subtask_hardness: np.ndarray, budget: int, k: int) -> float:
    """
    Simulate the discretization estimate of pass@k given subtask hardness, budget, and k.

    Args:
        subtask_hardness (np.ndarray): An array of subtask hardness values (between 0 and 1).
        budget (int): The total number of attempts allowed.
        k (int): The number of successful attempts required to pass.

    Returns:
        float: The discretization estimate of pass@k.
    """

    # Uniformly distribute sampling budget across subproblems
    n_subproblems = len(subtask_hardness)
    problem_budgets = evenly_distribute_budget(budget, n_subproblems)
    n_successes = attempt(subtask_hardness, n=problem_budgets)
    if np.sum(n_successes) == 0:
        return 0.0, problem_budgets, n_successes

    # Fit discretized beta distribution to (num_samples, num_successes) data
    num_samples_and_num_successes_df = pd.DataFrame({
        "Num. Samples Total": problem_budgets,
        "Num. Samples Correct": n_successes,
    })
    beta_3_discretized_params = fit_discretized_beta_three_parameters_to_num_samples_and_num_successes(num_samples_and_num_successes_df)

    # Compute estimate of pass@k using fitted (alpha, beta, scale) parameters
    estimate = compute_estimate(beta_3_discretized_params, k)

    return estimate, problem_budgets, n_successes

def simulate_dynamic_estimate_of_pass_at_k(subtask_hardness: np.ndarray, budget: int, k: int) -> float:
    """
    Simulate the dynamic estimate of pass@k given subtask hardness, budget, and k.

    Args:
        subtask_hardness (np.ndarray): An array of subtask hardness values (between 0 and 1).
        budget (int): The total number of attempts allowed.
        k (int): The number of successful attempts required to pass.

    Returns:
        float: The dynamic estimate of pass@k.
    """

    # Dynamically select subproblems with fewest successes
    n_subproblems = len(subtask_hardness)
    permutation = np.random.permutation(n_subproblems)
    inverse_permutation = np.argsort(permutation)
    subtask_hardness = subtask_hardness[permutation]  # shuffle subproblems to avoid argmin ordering bias
    problem_budgets = np.zeros(n_subproblems, dtype=np.int32)
    n_successes = np.zeros(n_subproblems, dtype=np.int32)
    for _ in range(budget):
        selected = np.argmin(n_successes + 1e-6 * problem_budgets)  # break ties by preferring subproblems with fewer attempts
        problem_budgets[selected] += 1
        n_successes[selected] += attempt(subtask_hardness[selected])

    # Find maximum likelihood parameters for beta-binomial distribution
    num_samples_and_num_successes_df = pd.DataFrame({
        "Num. Samples Total": problem_budgets,
        "Num. Samples Correct": n_successes,
    })
    beta_2_params = fit_beta_binomial_two_parameters_to_num_samples_and_num_successes(num_samples_and_num_successes_df)

    # Compute estimate of pass@k using fitted (alpha, beta, scale) parameters
    estimate = compute_estimate(beta_2_params, k)

    # Undo permutation of subproblems
    problem_budgets = problem_budgets[inverse_permutation]
    n_successes = n_successes[inverse_permutation]

    return estimate, problem_budgets, n_successes

def run_single(
    problem: str,
    model: str,
    subtask_hardness: np.ndarray,
    budget: int=1000,
    k: int=100,
    seed: int=0,
):
    """
    Simulate the pass@k estimation process for a given model and subtask hardness array.

    Args:
        model (str): The name of the model being simulated.
        subtask_hardness (np.ndarray): An array of subtask hardness values (between 0 and 1).
        budget (int): The total number of attempts allowed.
        k (int): The number of successful attempts required to pass.
        seed (int): Random seed for reproducibility.
    """
    np.random.seed(seed)
    pass_at_k = compute_true_pass_at_k(subtask_hardness, k)
    regression_estimate, regression_n_attempts, regression_n_successes = simulate_regression_estimate_of_pass_at_k(subtask_hardness, budget, k)
    discretization_estimate, discretization_n_attempts, discretization_n_successes = simulate_discretization_estimate_of_pass_at_k(subtask_hardness, budget, k)
    dynamic_estimate, dynamic_n_attempts, dynamic_n_successes = simulate_dynamic_estimate_of_pass_at_k(subtask_hardness, budget, k)

    return {
        "Problem": problem,
        "Model": model,
        "Subproblems": len(subtask_hardness),
        "Budget": budget,
        "k": k,
        "Seed": seed,
        "True Pass@k": pass_at_k,
        "Regression Estimate": regression_estimate,
        "Regression Num. Attempts": regression_n_attempts.tolist(),
        "Regression Num. Successes": regression_n_successes.tolist(),
        "Discretization Estimate": discretization_estimate,
        "Discretization Num. Attempts": discretization_n_attempts.tolist(),
        "Discretization Num. Successes": discretization_n_successes.tolist(),
        "Dynamic Estimate": dynamic_estimate,
        "Dynamic Num. Attempts": dynamic_n_attempts.tolist(),
        "Dynamic Num. Successes": dynamic_n_successes.tolist(),
    }

def run():
    tasks = chain.from_iterable(
        extract_models_with_subtask_hardness(problem, gen())
        for problem, gen in DATA_SOURCES
    )
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for problem, gen in DATA_SOURCES:
        for model, subtask_hardness in extract_models_with_subtask_hardness(gen()):
            print(f"Simulating {model} on {problem} with {len(subtask_hardness)} subproblems... (~10 minutes)")
            tasks = (
                (problem, model, subtask_hardness, budget, k, seed)
                for budget in BUDGET_VALUES
                for k in K_VALUES
                for seed in SEEDS
            )
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.starmap(run_single, tasks)

            out_path = DIR_RESULTS / f"simulate_pass_at_k_estimation_{sanitize(problem)}_{sanitize(model)}.parquet"
            df = pd.DataFrame(results).convert_dtypes(dtype_backend="pyarrow")
            df.to_parquet(out_path, engine="pyarrow", index=False, compression="snappy")
            print(f"Wrote {len(df)} rows → {out_path}")