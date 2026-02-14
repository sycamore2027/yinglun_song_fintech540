from pathlib import Path

import numpy as np
import pandas as pd


def fit_general_t(x):
    values = np.asarray(x, dtype=float).reshape(-1)
    mu = float(np.mean(values))
    s = float(np.std(values, ddof=1))
    if s <= 0:
        raise ValueError("Input series must have positive standard deviation.")

    z = (values - mu) / s
    ex_kurt = float(np.mean(z**4) - 3.0)
    if np.isfinite(ex_kurt) and ex_kurt > 1e-8:
        nu = max(2.1, 6.0 / ex_kurt + 4.0)
    else:
        nu = 100.0

    sigma = s * np.sqrt((nu - 2.0) / nu)
    return mu, sigma, nu


def t_quantile(alpha, nu, seed=4, n_sim=1_000_000):
    rng = np.random.default_rng(seed)
    sim = rng.standard_t(df=nu, size=n_sim)
    return float(np.quantile(sim, alpha))


base_dir = Path(__file__).resolve().parent
input_candidates = [
    base_dir / "test7_2.csv",
    base_dir.parent / "test7_2.csv",
]

input_path = next((p for p in input_candidates if p.exists()), None)
if input_path is None:
    raise FileNotFoundError("Could not find test7_2.csv in HW3 or HW1.")

cin = pd.read_csv(input_path).to_numpy()
mu, sigma, nu = fit_general_t(cin[:, 0])
q05 = t_quantile(0.05, nu, seed=4)

out = pd.DataFrame(
    {
        "VaR Absolute": [-(mu + sigma * q05)],
        "VaR Diff from Mean": [-(sigma * q05)],
    }
)

out.to_csv(base_dir / "testout_8.2.csv", index=False)
