import numpy as np
import pandas as pd
from pathlib import Path


def simulate_pca(cov_matrix, n_samples, rng, pct_exp=1.0, mean=None):
    cov = np.asarray(cov_matrix, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_matrix must be a square matrix.")

    n = cov.shape[0]
    mu = np.zeros(n, dtype=float) if mean is None else np.asarray(mean, dtype=float)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    total_var = float(np.sum(eigenvalues))
    pos_idx = np.where(eigenvalues >= 1e-8)[0]

    if pct_exp < 1.0 and pos_idx.size > 0 and total_var > 0:
        cum = 0.0
        n_keep = 0
        for idx in pos_idx:
            cum += eigenvalues[idx] / total_var
            n_keep += 1
            if cum >= pct_exp:
                break
        pos_idx = pos_idx[:n_keep]

    vals = eigenvalues[pos_idx]
    vecs = eigenvectors[:, pos_idx]

    b = vecs @ np.diag(np.sqrt(vals))
    r = rng.standard_normal((vals.shape[0], n_samples))
    sims = (b @ r).T
    return sims + mu


seed = 4
rng = np.random.default_rng(seed)

base_dir = Path(__file__).resolve().parent
cin_df = pd.read_csv(base_dir / "test5_2.csv")
cin = cin_df.to_numpy()

sims = simulate_pca(cin, 100000, rng, pct_exp=1.0)
cout = np.cov(sims, rowvar=False)

columns = [f"x{i + 1}" for i in range(cout.shape[1])]
pd.DataFrame(cout, columns=columns).to_csv(base_dir / "testout_5.5.csv", index=False)
