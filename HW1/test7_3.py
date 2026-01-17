import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

def t_regression_fit(y: np.ndarray, X: np.ndarray):
    
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if k != 3:
        raise ValueError(f"Expected 3 predictors for B1,B2,B3, but got {k}")

    # Add intercept column for Alpha
    X1 = np.column_stack([np.ones(n), X])  # (n x 4): [Alpha, B1, B2, B3]

    # OLS initial guess for betas
    beta0, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid0 = y - X1 @ beta0
    sigma0 = float(np.std(resid0, ddof=X1.shape[1]))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = 1.0

    # params vector: [Alpha, B1, B2, B3, log_sigma, log_nu_minus_2]
    def pack(alpha_betas, sigma, nu):
        return np.concatenate([alpha_betas, [np.log(sigma), np.log(nu - 2.0)]])

    def unpack(p):
        alpha_betas = p[:4]
        sigma = float(np.exp(p[4]))
        nu = float(2.0 + np.exp(p[5]))  # ensures nu > 2
        return alpha_betas, sigma, nu

    def nll(p):
        alpha_betas, sigma, nu = unpack(p)
        r = y - X1 @ alpha_betas
        # Student-t log-likelihood with scale sigma
        return -np.sum(stats.t.logpdf(r, df=nu, loc=0.0, scale=sigma))

    p0 = pack(beta0, sigma0, nu=10.0)

    res = minimize(nll, p0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    alpha_betas_hat, sigma_hat, nu_hat = unpack(res.x)

    Alpha, B1, B2, B3 = alpha_betas_hat.tolist()

    return {
        "mu": 0.0,
        "sigma": sigma_hat,
        "nu": nu_hat,
        "Alpha": Alpha,
        "B1": B1,
        "B2": B2,
        "B3": B3,
    }

def t_regression(filename: str):
    data = pd.read_csv(filename)

    # y = last column, X = first 3 columns (for B1,B2,B3)
    y = data.iloc[:, -1].to_numpy()
    X = data.iloc[:, :3].to_numpy()

    out = t_regression_fit(y, X)
    return pd.DataFrame([[out["mu"], out["sigma"], out["nu"], out["Alpha"], out["B1"], out["B2"], out["B3"]]],
                        columns=["mu", "sigma", "nu", "Alpha", "B1", "B2", "B3"])

print(t_regression("yinglun_song_fintech540/HW1/test7_3.csv"))
