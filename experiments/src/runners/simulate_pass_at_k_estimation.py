import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.linear_model import LinearRegression

from constants import DATA_SOURCES, BUDGET_VALUES, K_VALUES, SEEDS, DIR_EST_RESULTS, DIR_ATT_RESULTS
from utilities import (
    extract_models_with_samples,
    attempt,
    shuffled_attempt_generators,
    evenly_distribute_budget,
    sanitize,
)

from monkeys.analyze import (
    estimate_pass_at_k,
    fit_discretized_beta_three_parameters_to_num_samples_and_num_successes,
    fit_beta_binomial_two_parameters_to_num_samples_and_num_successes,
)
from monkeys.EM import compute_estimate

def compute_hardness(samples: list[list[bool]]) -> list[float]:
    """
    Compute the hardness of each subtask given an array of subtask samples.

    Args:
        samples (list[list[bool]]): An array of subtask samples (between 0 and 1).

    Returns:
        list[float]: The hardness of each subtask.
    """
    return [sum(st_samples) / len(st_samples) for st_samples in samples]

def compute_true_pass_at_k(hardness: list[float], k_values: list[int]) -> list[float]:
    """
    Compute the true pass@k value for each k value given an array of subtask samples.

    Args:
        samples (list[list[bool]]): An array of subtask samples (between 0 and 1).
        k_values (list[int]): The list of k values for which to compute pass@k.

    Returns:
        float: The true pass@k value.
    """
    return [(1 - np.mean((1 - np.array(hardness)) ** k)).item() for k in k_values]

def simulate_regression_estimate_of_pass_at_k(samples: list[list[bool]], budget: int, k_values: list[int], seed: int=0) -> list[float]:
    """
    Simulate the regression estimate of pass@k given subtask hardness, budget, and k values.

    Args:
        samples (list[list[bool]]): An list of samples for each subtask.
        budget (int): The total number of attempts allowed.
        k_values (list[int]): The list of k values for which to estimate pass@k.

    Returns:
        float: The regression estimate of pass@k.
    """
    n_subtasks = len(samples)
    attempt_generators = shuffled_attempt_generators(samples, seed)

    # Uniformly distribute sampling budget across subtasks
    subtask_budgets = np.array(evenly_distribute_budget(budget, n_subtasks))
    subtask_budgets = np.minimum(subtask_budgets, np.array([len(s) for s in samples]))
    subtask_successes = np.array(attempt(attempt_generators, amt=subtask_budgets.tolist()))
    if np.sum(subtask_successes) == 0:
        return [0.0] * len(k_values), subtask_budgets.tolist(), subtask_successes.tolist()
    subtask_successes = subtask_successes[subtask_budgets > 0].tolist()
    subtask_budgets = subtask_budgets[subtask_budgets > 0].tolist()

    # Estimate pass@k for small k values
    small_ks = []
    estimated_pass_at_small_ks = []
    for small_k in range(1, budget // len(subtask_budgets) + 1):
        est = np.mean(estimate_pass_at_k(subtask_budgets, subtask_successes, small_k))
        small_ks.append(small_k)
        estimated_pass_at_small_ks.append(est)

    # Extrapolate to desired k using linear regression on log-log scale
    X = np.log(np.array(small_ks, dtype=np.float64)).reshape(-1, 1)
    with np.errstate(divide='ignore'):
        y = -np.log(np.array(estimated_pass_at_small_ks, dtype=np.float64))
    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)
    estimates = np.minimum(np.exp(-model.predict(np.log(np.array(k_values)).reshape(-1, 1))), 1).flatten().tolist()

    return estimates, subtask_budgets, subtask_successes

def simulate_discretization_estimate_of_pass_at_k(samples: list[list[bool]], budget: int, k_values: list[int], seed: int=0) -> list[float]:
    """
    Simulate the discretization estimate of pass@k given subtask hardness, budget, and k values.

    Args:
        samples (list[list[bool]]): A list of samples for each subtask.
        budget (int): The total number of attempts allowed.
        k_values (list[int]): The list of k values for which to estimate pass@k.

    Returns:
        float: The discretization estimate of pass@k.
    """
    n_subtasks = len(samples)
    attempt_generators = shuffled_attempt_generators(samples, seed)

    # Uniformly distribute sampling budget across subtasks
    subtask_budgets = np.array(evenly_distribute_budget(budget, n_subtasks))
    subtask_budgets = np.minimum(subtask_budgets, np.array([len(s) for s in samples]))
    subtask_successes = np.array(attempt(attempt_generators, amt=subtask_budgets.tolist()))
    if np.sum(subtask_successes) == 0:
        return [0.0] * len(k_values), subtask_budgets.tolist(), subtask_successes.tolist()
    subtask_successes = subtask_successes[subtask_budgets > 0].tolist()
    subtask_budgets = subtask_budgets[subtask_budgets > 0].tolist()

    # Fit discretized beta distribution to (num_samples, num_successes) data
    num_samples_and_num_successes_df = pd.DataFrame({
        "Num. Samples Total": subtask_budgets,
        "Num. Samples Correct": subtask_successes,
    })
    beta_3_discretized_params = fit_discretized_beta_three_parameters_to_num_samples_and_num_successes(num_samples_and_num_successes_df)

    # Compute estimate of pass@k using fitted (alpha, beta, scale) parameters
    estimates = [compute_estimate(beta_3_discretized_params, k) for k in k_values]

    return estimates, subtask_budgets, subtask_successes

def simulate_dynamic_estimate_of_pass_at_k(samples: list[list[bool]], budget: int, k_values: list[int], seed: int=0) -> list[float]:
    """
    Simulate the dynamic estimate of pass@k given subtask hardness, budget, and k.

    Args:
        samples (list[list[bool]]): A list of samples for each subtask.
        budget (int): The total number of attempts allowed.
        k_values (list[int]): The list of k values for which to estimate pass@k.

    Returns:
        float: The dynamic estimate of pass@k.
    """
    n_subtasks = len(samples)
    attempt_generators = shuffled_attempt_generators(samples, seed)

    # Dynamically select subtasks with fewest successes, then fewest attempts, breaking ties randomly
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_subtasks)
    subtask_budgets = [0] * n_subtasks
    subtask_successes = [0] * n_subtasks
    for _ in range(budget):
        selected = np.lexsort((permutation, subtask_budgets, subtask_successes))[0]
        if subtask_budgets[selected] < len(samples[selected]):
            subtask_budgets[selected] += 1
            subtask_successes[selected] += attempt(attempt_generators[selected])

    # Find maximum likelihood parameters for beta-binomial distribution
    num_samples_and_num_successes_df = pd.DataFrame({
        "Num. Samples Total": subtask_budgets,
        "Num. Samples Correct": subtask_successes,
    })
    beta_2_params = fit_beta_binomial_two_parameters_to_num_samples_and_num_successes(num_samples_and_num_successes_df)

    # Compute estimate of pass@k using fitted (alpha, beta, scale) parameters
    estimates = [compute_estimate(beta_2_params, k) for k in k_values]

    return estimates, subtask_budgets, subtask_successes

def simulate_uniform_estimate_of_pass_at_k(samples: list[list[bool]], budget: int, k_values: list[int], seed: int=0) -> list[float]:
    """
    Simulate the uniform estimate of pass@k given subtask hardness, budget, and k.

    Args:
        samples (list[list[bool]]): A list of samples for each subtask.
        budget (int): The total number of attempts allowed.
        k_values (list[int]): The list of k values for which to estimate pass@k.

    Returns:
        float: The dynamic estimate of pass@k.
    """
    n_subtasks = len(samples)
    attempt_generators = shuffled_attempt_generators(samples, seed)

    # Uniformly distribute sampling budget across subtasks
    subtask_budgets = evenly_distribute_budget(budget, n_subtasks)
    subtask_budgets = [min(b, len(samples[i])) for i, b in enumerate(subtask_budgets)]
    subtask_successes = attempt(attempt_generators, amt=subtask_budgets)
    if np.sum(subtask_successes) == 0:
        return [0.0] * len(k_values), subtask_budgets, subtask_successes

    # Find maximum likelihood parameters for beta-binomial distribution
    num_samples_and_num_successes_df = pd.DataFrame({
        "Num. Samples Total": subtask_budgets,
        "Num. Samples Correct": subtask_successes,
    })
    beta_2_params = fit_beta_binomial_two_parameters_to_num_samples_and_num_successes(num_samples_and_num_successes_df)

    # Compute estimate of pass@k using fitted (alpha, beta, scale) parameters
    estimates = [compute_estimate(beta_2_params, k) for k in k_values]

    return estimates, subtask_budgets, subtask_successes

def get_optimal_allocation(hardness: list[float], k: int=1000) -> list[float]:
    r_hardness = 1 - np.array(hardness)
    log_hardness = np.log(np.clip(r_hardness, 1e-8, 1 - 1e-8))
    log_weights = ((2 * k - 1) * log_hardness + np.log(1 - np.exp(log_hardness))) * 1 / 2
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights - np.max(log_weights)).tolist()
    total = sum(weights)
    return [w / total for w in weights]

def run_single(
    problem: str,
    model: str,
    samples: list[list[bool]],
    budget: int=1000,
    seed: int=0,
):
    """
    Simulate the pass@k estimation process for a given model and subtask hardness array.

    Args:
        model (str): The name of the model being simulated.
        samples (list[list[bool]]): A list of samples for each subtask.
        budget (int): The total number of attempts allowed.
        seed (int): Random seed for reproducibility.
    """
    hardness = compute_hardness(samples)
    pass_at_k = compute_true_pass_at_k(hardness, K_VALUES)

    regression_estimates, regression_n_attempts, regression_n_successes = simulate_regression_estimate_of_pass_at_k(samples, budget, K_VALUES, seed)
    discretization_estimates, discretization_n_attempts, discretization_n_successes = simulate_discretization_estimate_of_pass_at_k(samples, budget, K_VALUES, seed)
    dynamic_estimates, dynamic_n_attempts, dynamic_n_successes = simulate_dynamic_estimate_of_pass_at_k(samples, budget, K_VALUES, seed)

    return {
        "est_entries": [{
            "Problem": problem,
            "Model": model,
            "Subproblems": len(samples),
            "Budget": budget,
            "k": k,
            "Seed": seed,
            "True Pass@k": pass_at_k[i],
            "Regression Estimate": regression_estimates[i],
            "Discretization Estimate": discretization_estimates[i],
            "Dynamic Estimate": dynamic_estimates[i],
        } for i, k in enumerate(K_VALUES)],
        "att_entry": {
            "Problem": problem,
            "Model": model,
            "Subproblems": len(samples),
            "Hardness": hardness,
            "Budget": budget,
            "Seed": seed,
            "Regression Attempts": regression_n_attempts,
            "Regression Successes": regression_n_successes,
            "Discretization Attempts": discretization_n_attempts,
            "Discretization Successes": discretization_n_successes,
            "Dynamic Attempts": dynamic_n_attempts,
            "Dynamic Successes": dynamic_n_successes,
        }
    }

def run():
    for problem, gen in DATA_SOURCES:
        for model, samples in extract_models_with_samples(gen()):
            f_base = f"{sanitize(problem)}_{sanitize(model)}.parquet"
            tasks = (
                (problem, model, samples, budget, seed)
                for budget in BUDGET_VALUES
                for seed in SEEDS
            )

            # Run simulations in parallel
            print(f"Running simulations for {model} on {problem} with {len(samples)} subtasks... (~10 minutes)")
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.starmap(run_single, tasks)

            # Extract and save entries
            print(f"  Writing estimation results...")
            all_est_entries = [item for res in results for item in res["est_entries"]]
            df = pd.DataFrame(all_est_entries).convert_dtypes(dtype_backend="pyarrow")
            df.to_parquet(DIR_EST_RESULTS / f_base, engine="pyarrow", index=False, compression="snappy")
            print(f"  Wrote {len(df)} rows → {DIR_EST_RESULTS / f_base}")

            # Extract and save attempts
            print(f"  Writing attempts...")
            all_att_entries = [res["att_entry"] for res in results]
            attempts_df = pd.DataFrame(all_att_entries).convert_dtypes(dtype_backend="pyarrow")
            attempts_df.to_parquet(DIR_ATT_RESULTS / f_base, engine="pyarrow", index=False)
            print(f"  Wrote {len(attempts_df)} rows → {DIR_ATT_RESULTS / f_base}")