from pathlib import Path

import numpy as np
import pandas as pd


class GeneralTFit:
    def __init__(self, mu, sigma, nu, seed=4):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.nu = float(nu)
        self._rng = np.random.default_rng(seed)

    def eval(self, u):
        n = np.asarray(u).size
        return self.mu + self.sigma * self._rng.standard_t(self.nu, size=n)


def fit_general_t(x, seed=4):
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
    return GeneralTFit(mu=mu, sigma=sigma, nu=nu, seed=seed)


def VaR(a, alpha=0.05):
    x = np.sort(np.asarray(a, dtype=float).reshape(-1))
    n = x.size
    nup = max(1, min(n, int(np.ceil(n * alpha))))
    ndn = max(1, min(n, int(np.floor(n * alpha))))
    v = 0.5 * (x[nup - 1] + x[ndn - 1])
    return -float(v)


seed = 4
rng = np.random.default_rng(seed)

base_dir = Path(__file__).resolve().parent
input_candidates = [
    base_dir / "test7_2.csv",
    base_dir.parent / "test7_2.csv",
]
input_path = next((p for p in input_candidates if p.exists()), None)
if input_path is None:
    raise FileNotFoundError("Could not find test7_2.csv in HW3 or HW1.")

cin = pd.read_csv(input_path).to_numpy()
fd = fit_general_t(cin[:, 0], seed=seed)
sim = fd.eval(rng.random(10000))

out = pd.DataFrame(
    {
        "VaR Absolute": [VaR(sim)],
        "VaR Diff from Mean": [VaR(sim - np.mean(sim))],
    }
)
out.to_csv(base_dir / "testout_8.3.csv", index=False)
