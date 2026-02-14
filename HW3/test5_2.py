import numpy as np
import pandas as pd
from pathlib import Path


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
input_path = base_dir / "test5_2.csv"
cin_df = pd.read_csv(input_path)
cin = cin_df.to_numpy()

sims = simulateNormal(100000, cin, rng)
cout = np.cov(sims, rowvar=False)

columns = [f"x{i + 1}" for i in range(cout.shape[1])]
pd.DataFrame(cout, columns=columns).to_csv(base_dir / "testout_5.2.csv", index=False)
