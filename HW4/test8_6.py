from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def ES(returns, alpha=0.05):
    x = np.sort(np.asarray(returns, dtype=float).reshape(-1))
    n = x.size
    if n == 0:
        raise ValueError("Input array cannot be empty.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")

    nup = max(1, min(n, int(np.ceil(n * alpha))))
    ndn = max(1, min(n, int(np.floor(n * alpha))))
    var_level = 0.5 * (x[nup - 1] + x[ ndn - 1])
    return -float(np.mean(x[x <= var_level]))


def main():
    seed = 4
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "test7_2.csv"

    cin = pd.read_csv(input_path).to_numpy()
    values = np.asarray(cin[:, 0], dtype=float).reshape(-1)
    df, mu, sigma = t.fit(values)
    sim = t.rvs(df, loc=mu, scale=sigma, size=10000, random_state=seed)

    out = pd.DataFrame(
        {
            "ES Absolute": [ES(sim)],
            "ES Diff from Mean": [ES(sim - np.mean(sim))],
        }
    )
    out.to_csv(base_dir / "testout8_6.csv", index=False)


if __name__ == "__main__":
    main()
