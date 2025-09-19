import re
import unicodedata
import pandas as pd
import numpy as np
from typing import Generator, Tuple, Union

def sanitize(title: str, sep: str = "-") -> str:
    s = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", sep, s)
    s = re.sub(re.escape(sep) + r"{2,}", sep, s).strip(sep + " ._")
    return s

def attempt(hardness: Union[float, np.ndarray], n: Union[int, np.ndarray]=1) -> int:
    """
    Simulate n attempts with success probability hardness.
    Returns the number of successful attempts.
    """
    assert np.array(0.0 <= hardness).all() and np.array(hardness <= 1.0).all(), "Hardness must be between 0 and 1."
    return np.random.binomial(n, hardness)

def evenly_distribute_budget(budget: int, n_subproblems: int) -> np.ndarray:
    """
    Evenly distribute a total budget across a number of subproblems.
    Returns an array of budgets for each subproblem.
    """
    problem_budgets = np.full(n_subproblems, budget // n_subproblems)
    problem_budgets[:budget % n_subproblems] += 1  # distribute remaining attempts
    return np.random.permutation(problem_budgets)

def extract_models_with_subtask_hardness(df: pd.DataFrame) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Given a DataFrame with columns ['Score', 'Scaling Parameter', 'Problem Idx', 'Model'],
    yield tuples of (model_name, subtask_hardness_array) for each unique model.
    """
    required_columns = {'Score', 'Scaling Parameter', 'Problem Idx', 'Model'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Only keep rows with all 10k samples
    df = df[df['Scaling Parameter'] == 1.0]

    # Provide subtask hardness arrays for each model
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        model_df = model_df.sort_values(by='Problem Idx')
        subtask_hardness = model_df['Score'].to_numpy(dtype=np.float64)
        yield (model, subtask_hardness)