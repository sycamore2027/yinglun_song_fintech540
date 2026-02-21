import numpy as np
import pandas as pd
from scipy.stats import t

def fit_tDist(returns):
    df, mu, sigma = t.fit(returns)
    return df, mu, sigma

def ES_tDist(df, mu, sigma, alpha=0.05):
    if not (0.0 < alpha < 1.0):
        raise ValueError('Alpha has to be between 0.0 to 1.0')
    if sigma < 0.0:
        raise ValueError('sigma must be positive')
    if sigma == 0.0:
        return -float(mu)
    if df <= 1:
        raise ValueError('df must be > 1 for ES to exist')
    
    z = t.ppf(alpha, df)
    pdf = t.pdf(z, df)
    tail_mean = mu - sigma * ((df + z*z) / (df - 1)) * (pdf / alpha)
    return -float(tail_mean)

returns = pd.read_csv("yinglun_song_fintech540/HW4/test7_2.csv")

df, mu, sigma = fit_tDist(returns)

out = pd.DataFrame(
    {
        "ES Absolute": [ES_tDist(df, mu, sigma)],
        "ES Diff from Mean": [ES_tDist(df, 0.0, sigma)],
    }
)
out.to_csv("yinglun_song_fintech540/HW4/testout8_5.csv", index=False)
