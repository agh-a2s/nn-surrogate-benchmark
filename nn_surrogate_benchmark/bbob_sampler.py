import numpy as np
import pandas as pd
from tqdm import tqdm
from pyDOE2 import lhs
from cocoex import Suite
import os


def generate_lhs_samples(
    n_samples: int,
    bounds: tuple[float, float],
    dim: int,
    criterion: str | None = "maximin",
    random_state: int | None = None,
):
    if random_state is not None:
        np.random.seed(random_state)

    if criterion is None:
        lhs_raw = lhs(dim, samples=n_samples)
    else:
        lhs_raw = lhs(dim, samples=n_samples, criterion=criterion)

    lower, upper = bounds
    scale = upper - lower

    lhs_scaled = lower + lhs_raw * scale
    return lhs_scaled


def main():
    BOUNDS = [-5.0, 5.0]
    N_SAMPLES = 100_000
    DIMENSIONS = 2
    INSTANCE_ID = 1
    FUNCTION_ID = 1
    DATA_DIR = "data"

    suite = Suite(
        "bbob",
        f"instances: {FUNCTION_ID}",
        f"dimensions: {DIMENSIONS} instance_indices: {INSTANCE_ID}",
    )

    os.makedirs(DATA_DIR, exist_ok=True)

    for function in suite:
        print(
            f"Processing function {function.id}, dimension {function.dimension} instance_indices: {INSTANCE_ID}"
        )
        combinations = generate_lhs_samples(
            n_samples=N_SAMPLES,
            bounds=BOUNDS,
            dim=function.dimension,
            criterion=None,
            random_state=42,
        )
        rows = []
        for x in tqdm(combinations):
            f_value = function(x)
            rows.append(
                {
                    **{f"x{i+1}": float(val) for i, val in enumerate(x)},
                    "y": float(f_value),
                }
            )

        df = pd.DataFrame(rows)
        filename = f"{DATA_DIR}/{function.id}_samples.csv"
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")


if __name__ == "__main__":
    main()
