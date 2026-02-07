import numpy as np
import pandas as pd

def safe_cholesky(cov, jitter=1e-10, max_tries=5):
    """
    Compute Cholesky factor of a near-PSD covariance matrix.
    Adds small diagonal jitter if needed.
    """
    cov = 0.5 * (cov + cov.T)  
    for i in range(max_tries):
        try:
            return np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov = cov + jitter * np.eye(cov.shape[0])
            jitter *= 10
    raise np.linalg.LinAlgError("Cholesky failed even after jitter.")

cin = pd.read_csv("HW2/test3_1_output.csv")
cov = cin.values.astype(float)
L = safe_cholesky(cov)
pd.DataFrame(L).to_csv("HW2/test4_1_output.csv", float_format="%.12g", index=False)