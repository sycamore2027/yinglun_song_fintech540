import numpy as np
import pandas as pd

def ew_covariance_df(df, lam=0.97):

    R = df.values
    n_obs, n_assets = R.shape

    i = np.arange(n_obs)
    raw_w = (1 - lam) * lam**(n_obs - 1 - i)
    w = raw_w / raw_w.sum()

    mu = (w[:, None] * R).sum(axis=0)

    D = R - mu

    cov = D.T @ (w[:, None] * D)

    return pd.DataFrame(cov, index=df.columns, columns=df.columns)

df = pd.read_csv('HW2/test2.csv')
cov_ew = ew_covariance_df(df, lam=0.97)
cov_ew.to_csv('HW2/test2_1_output.csv', index=False)
print(cov_ew)