import numpy as np
import pandas as pd
from pathlib import Path


def simulateNormal(n_samples, cov_matrix, rng):
    cov = np.asarray(cov_matrix, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_matrix must be a square matrix.")

    cov = (cov + cov.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    floor = max(1e-10, 1e-8 * np.max(np.abs(eigenvalues)))
    eigenvalues = np.maximum(eigenvalues, floor)
    cov_pd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    cov_pd = (cov_pd + cov_pd.T) / 2.0
    transform = np.linalg.cholesky(cov_pd)

    z = rng.standard_normal((n_samples, cov.shape[0]))
    return z @ transform.T


seed = 4
rng = np.random.default_rng(seed)

base_dir = Path(__file__).resolve().parent
cin_df = pd.read_csv(base_dir / "test5_1.csv")
cin = cin_df.to_numpy()

sims = simulateNormal(100000, cin, rng)
cout = np.cov(sims, rowvar=False)

columns = [f"x{i + 1}" for i in range(cout.shape[1])]
pd.DataFrame(cout, columns=columns).to_csv(base_dir / "testout_5.1.csv", index=False)
