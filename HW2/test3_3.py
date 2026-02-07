import numpy as np
import pandas as pd

def higham_near_psd_corr(C, iters=100):
    Y = C.copy()
    for _ in range(iters):
        eigvals, eigvecs = np.linalg.eigh(Y)
        eigvals[eigvals < 0] = 0
        X = eigvecs @ np.diag(eigvals) @ eigvecs.T
        Y = X.copy()
        np.fill_diagonal(Y, 1.0)
    return Y

cin = pd.read_csv('HW2/test1_3_output.csv')
cols = cin.columns.copy()
cov= cin.values.astype(float)

std = np.sqrt(np.diag(cov))
D_inv = np.diag(1.0 / std)
corr = D_inv @ cov @ D_inv
np.fill_diagonal(corr, 1.0)

corr_fixed = higham_near_psd_corr(corr)
D = np.diag(std)
cov_fixed = D @ corr_fixed @ D
pd.DataFrame(cov_fixed).to_csv('HW2/test3_3_output.csv', float_format="%.12g", index=False)