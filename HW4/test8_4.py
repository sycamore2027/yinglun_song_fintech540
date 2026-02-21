from math import exp, pi, sqrt
from scipy.stats import norm
import numpy as np
import pandas as pd


def fit_normal(x):
    values = np.asarray(x, dtype=float).reshape(-1)
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    return mu, sigma


def ES_normal(mu, sigma, alpha=0.05):
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1.")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")
    if sigma == 0.0:
        return -float(mu)

    z = norm.ppf(alpha)
    phi = exp(-0.5 * z * z) / sqrt(2.0 * pi)
    tail_mean = mu - sigma * phi / alpha
    return -float(tail_mean)

cin = pd.read_csv("yinglun_song_fintech540/HW4/test7_1.csv").to_numpy()
mu, sigma = fit_normal(cin[:, 0])

out = pd.DataFrame(
    {
        "ES Absolute": [ES_normal(mu, sigma)],
        "ES Diff from Mean": [ES_normal(0.0, sigma)],
    }
)
out.to_csv("testout8_4.csv", index=False)
