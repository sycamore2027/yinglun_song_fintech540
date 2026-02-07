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

corr = pd.read_csv('HW2/test1_4_output.csv')
corr = corr.values.astype(float)
np.fill_diagonal(corr, 1.0)

corr_fixed = higham_near_psd_corr(corr)
pd.DataFrame(corr_fixed).to_csv('HW2/test3_4_output.csv', float_format="%.12g", index=False)