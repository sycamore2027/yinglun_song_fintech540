import numpy as np
import pandas as pd
from pathlib import Path


def near_psd_higham(cov_matrix, iters=100):
    cov = np.asarray(cov_matrix, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_matrix must be a square matrix.")

    cov = (cov + cov.T) / 2.0
    std = np.sqrt(np.diag(cov))
    if np.any(std <= 0):
        raise ValueError("cov_matrix must have strictly positive diagonal values.")

    d_inv = np.diag(1.0 / std)
    corr = d_inv @ cov @ d_inv
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)

    y = corr.copy()
    for _ in range(iters):
        eigenvalues, eigenvectors = np.linalg.eigh(y)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        x = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        y = (x + x.T) / 2.0
        np.fill_diagonal(y, 1.0)

    d = np.diag(std)
    cov_fixed = d @ y @ d
    return (cov_fixed + cov_fixed.T) / 2.0


def simulateNormal(n_samples, cov_matrix, rng):
    cov = np.asarray(cov_matrix, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_matrix must be a square matrix.")

    cov = (cov + cov.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    transform = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    z = rng.standard_normal((n_samples, cov.shape[0]))
    return z @ transform.T


seed = 4
rng = np.random.default_rng(seed)

base_dir = Path(__file__).resolve().parent
cin_df = pd.read_csv(base_dir / "test5_3.csv")
cin = cin_df.to_numpy()

cin_fixed = near_psd_higham(cin)
sims = simulateNormal(100000, cin_fixed, rng)
cout = np.cov(sims, rowvar=False)

columns = [f"x{i + 1}" for i in range(cout.shape[1])]
pd.DataFrame(cout, columns=columns).to_csv(base_dir / "testout_5.4.csv", index=False)
