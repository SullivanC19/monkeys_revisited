import re
import random
import unicodedata
import pandas as pd
import numpy as np
from typing import Generator, Tuple, Union, List
from pathlib import Path
import tqdm

def load_all_parquets(in_dir: Union[str, Path]) -> pd.DataFrame:
    in_dir = Path(in_dir)
    files = sorted(in_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {in_dir}")
    df = pd.concat(tqdm.tqdm([pd.read_parquet(p, engine="pyarrow") for p in files]), ignore_index=True)
    return df

def sanitize(title: str, sep: str = "-") -> str:
    s = unicodedata.normalize("NFKD", str(title)).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", sep, s)
    s = re.sub(re.escape(sep) + r"{2,}", sep, s).strip(sep + " ._")
    return s

def attempt(generator: Union[Generator[bool, None, None], List[Generator[bool, None, None]]], amt: Union[int, List[int]]=1) -> Union[int, List[int]]:
    """
    Simulate several attempts using the provided generator(s).
    Returns the number of successful attempts.
    """
    if isinstance(generator, list):
        assert isinstance(amt, list) and len(generator) == len(amt), "If a list of generators is provided, n must be a list of the same length."
        return [attempt(gen, n) for gen, n in zip(generator, amt)]
    return sum(next(generator) for _ in range(amt))

def shuffled_attempt_generators(samples: list[list[bool]], seed: int) -> List[Generator[bool, None, None]]:
    """
    Create a list of generators, each simulating attempts for a subtask using the provided samples.
    """
    gen = np.random.default_rng(seed)
    return [(s for s in gen.permutation(sb_samples)) for sb_samples in samples]

def evenly_distribute_budget(budget: int, n_subtasks: int) -> List[int]:
    """
    Evenly distribute a total budget across a number of subtasks.
    Returns an array of budgets for each subtask.
    """
    problem_budgets = np.full(n_subtasks, budget // n_subtasks)
    problem_budgets[:budget % n_subtasks] += 1  # distribute remaining attempts
    return np.random.permutation(problem_budgets).tolist()

def extract_models_with_samples(df: pd.DataFrame) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Given a DataFrame with columns ['Score', 'Scaling Parameter', 'Problem Idx', 'Model'],
    yield tuples of (model_name, samples) for each unique model.
    """
    required_columns = {'Score', 'Problem Idx', 'Model'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Provide subtask samples for each model
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        samples = [
            problem_df['Score'].values.astype(bool).tolist()
            for _, problem_df in model_df.groupby('Problem Idx')
        ]
        yield (model, samples)