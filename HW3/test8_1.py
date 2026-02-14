from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


def fit_normal(x):
    values = np.asarray(x, dtype=float).reshape(-1)
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    return mu, sigma


def VaR_normal(mu, sigma, alpha=0.05):
    return -NormalDist(mu=mu, sigma=sigma).inv_cdf(alpha)


base_dir = Path(__file__).resolve().parent
input_candidates = [
    base_dir / "test7_1.csv",
    base_dir.parent / "HW1" / "test7_1.csv",
]

input_path = next((p for p in input_candidates if p.exists()), None)
if input_path is None:
    raise FileNotFoundError("Could not find test7_1.csv in HW3 or HW1.")

cin = pd.read_csv(input_path).to_numpy()
mu, sigma = fit_normal(cin[:, 0])

out = pd.DataFrame(
    {
        "VaR Absolute": [VaR_normal(mu, sigma)],
        "VaR Diff from Mean": [VaR_normal(0.0, sigma)],
    }
)

out.to_csv(base_dir / "testout_8.1.csv", index=False)
