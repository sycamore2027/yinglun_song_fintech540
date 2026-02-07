import numpy as np
import pandas as pd

def exponential_weights(n, lam):
    i = np.arange(n)
    raw = (1 - lam) * (lam ** (n - 1 - i))
    weights = raw / raw.sum()
    return weights

def ew_covariance(returns, lam):
    R = np.asarray(returns, dtype = float)
    n_obs, n_assets = R.shape
    w = exponential_weights(n_obs, lam)
    mu = (w[:, None] * R).sum(axis = 0)
    D = R - mu
    cov = D.T @ (w[:, None] * D)
    return cov

df = pd.read_csv('HW2/test2.csv')

cov_97 = ew_covariance(df, lam=0.97)
sd1 = np.sqrt(np.diag(cov_97))
diagm1 = np.diag(sd1)

cov_94 = ew_covariance(df, lam=0.94)
sd2 = 1.0 / np.sqrt(np.diag(cov_94))
diagm2 = np.diag(sd2)

combined = pd.DataFrame(diagm1 @ diagm2 @ cov_94 @ diagm2 @ diagm1, index=df.columns, columns=df.columns)

combined.to_csv('HW2/test2_3_output.csv', index = False, header = False)
print(combined)