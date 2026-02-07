import numpy as np
import pandas as pd
    
def rebonato_jackel(matrix, tol = 1e-15):
    matrix = np.asarray(matrix, dtype = float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be square matrix")
    
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals_clipped = np.maximum(eigvals, tol)
    lambda_p = np.diag(eigvals_clipped)
    
    S = eigvecs
    row_variances  =(S**2) @ eigvals_clipped
    row_variances = np.maximum(row_variances, tol)
    
    t = 1.0  / np.sqrt(row_variances)
    T = np.diag(t)
    
    C_hat = T @ S @ lambda_p @ S.T @ T
    
    return C_hat

cin = pd.read_csv('HW2/test1_3_output.csv')
cols = cin.columns.copy()
cov = cin.values.astype(float)
std = np.sqrt(np.diag(cov))
D_inv = np.diag(1.0 / std)
corr = D_inv @ cov @ D_inv

corr_fixed = rebonato_jackel(corr)
D = np.diag(std)
cov_fixed = D @ corr_fixed @ D
df_out = pd.DataFrame(cov_fixed)
df_out.to_csv('HW2/test3_1_output.csv', float_format="%.12g", index=False)
    
    
    
    